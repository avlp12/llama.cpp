#include "llama-safetensors-glm5-next.h"
#include "llama-arch.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <cstring>
#include <functional>
#include <unistd.h>

namespace {
using json=nlohmann::ordered_json;
void check(bool condition,const char * message){if(!condition)throw std::runtime_error(message);}
void text(const std::filesystem::path & p,const json & j){std::ofstream(p)<<j.dump();}
void rejected(const std::function<void()> & f){bool caught=false;try{f();}catch(const std::exception&){caught=true;}check(caught,"negative input accepted");}
struct shard {
    json header=json::object();std::vector<uint8_t> data;
    void add(const std::string & n,const std::string & dtype,std::vector<uint64_t> shape,std::vector<uint8_t> bytes){
        header[n]={{"dtype",dtype},{"shape",shape},{"data_offsets",{data.size(),data.size()+bytes.size()}}};data.insert(data.end(),bytes.begin(),bytes.end());
    }
    void save(const std::filesystem::path & p){auto s=header.dump();uint64_t len=s.size();std::ofstream f(p,std::ios::binary);for(int i=0;i<8;++i)f.put(char(len>>(8*i)));f.write(s.data(),s.size());f.write((char*)data.data(),data.size());}
};
std::vector<uint8_t> f32(std::initializer_list<float> v){std::vector<uint8_t> b(v.size()*4);std::memcpy(b.data(),v.begin(),b.size());return b;}
}
int main(int argc, char ** argv){
    if (argc==3 && std::string(argv[1])=="--metadata") {
        try {
            const std::filesystem::path dir(argv[2]);
            auto c=llama_safetensors_read_json(dir/"config.json");
            llama_safetensors_glm5_next_importer loader(dir,c,llama_safetensors_io_mode::BUFFERED);
            auto * meta=loader.build_metadata();
            const auto & tc=c.at("text_config");
            const int first_moe=tc.at("first_k_dense_replace").get<int>();
            const std::string bias_name=LLM_TN(LLM_ARCH_GLM5_NEXT)(LLM_TENSOR_FFN_EXP_PROBS_B,"bias",first_moe).str();
            ggml_type type;std::array<int64_t,GGML_MAX_DIMS> shape;
            check(loader.describe(bias_name,type,shape)&&type==GGML_TYPE_F32&&shape[0]==tc.at("n_routed_experts").get<int64_t>(),"original router bias metadata mismatch");
            std::cout<<"GLM5_ROUTER_BIAS_VALID name="<<bias_name<<" experts="<<shape[0]<<"\n";
            std::cout<<"GLM5_METADATA_VALID keys="<<gguf_get_n_kv(meta)<<" tensors_hint="<<loader.tensor_capacity_hint()<<"\n";
            gguf_free(meta);return 0;
        } catch(const std::exception & e) { std::cerr<<e.what()<<"\n";return 1; }
    }
    std::filesystem::path root=std::filesystem::temp_directory_path()/("glm5-native-test-"+std::to_string(getpid()));
    try{
        std::filesystem::create_directories(root);
        json cfg={{"model_type","glm5_next"},{"quantization_config",{{"quant_method","exl3"},{"codebook","mcg"},{"bits",4},{"scope","glm53_routed_experts_only"}}},
            {"text_config",{{"num_hidden_layers",2},{"num_nextn_predict_layers",0},{"n_routed_experts",2},{"hidden_size",128},{"moe_intermediate_size",128},{"layer_types",{"linear_attention","deepseek_sparse_attention"}},{"hc_mult",4},{"mla_use_nope",true},{"qk_rope_head_dim",0},{"num_attention_heads",2},{"linear_attn_config",{{"num_heads",2}}},{"kv_lora_rank",2},{"qk_nope_head_dim",2},{"v_head_dim",2}}}};
        text(root/"generation_config.json",{{"eos_token_id",{1}}});text(root/"tokenizer.json",json::object());
        shard fixture;
        const std::string prefix="model.language_model.layers.0.";
        for(int e=0;e<2;++e){
            const auto m=prefix+"mlp.experts."+std::to_string(e)+".gate_proj";
            std::vector<uint8_t> trellis(8*8*64*2,uint8_t(e));std::vector<uint8_t> scales(256);
            for(size_t i=1;i<scales.size();i+=2)scales[i]=0x3c;
            fixture.add(m+".trellis","I16",{8,8,64},trellis);
            fixture.add(m+".mcg","I32",{1},{0xed,0x1f,0xac,0xcb});
            fixture.add(m+".suh","F16",{128},scales);fixture.add(m+".svh","F16",{128},scales);
        }
        fixture.add(prefix+"self_attn.A_log","F32",{2},f32({0,0.5f}));
        fixture.add(prefix+"self_attn.kv_b_proj.weight","F32",{8,2},f32({0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}));
        fixture.add(prefix+"self_attn.q_conv1d.weight","BF16",{2,1,2},{0,0x3f,0,0x40,0,0x41,0,0x42});
        fixture.add("model.language_model.norm.weight","BF16",{2},{0x80,0x3f,0,0x40});
        fixture.add("model.language_model.embed_tokens.weight","BF16",{2,2},{0x80,0x3f,0,0x40,0xab,0x4a,0x91,0xbe});
        fixture.add(prefix+"mlp.gate.e_score_correction_bias","F32",{2},f32({0.125f,-0.25f}));
        fixture.save(root/"model.safetensors");
        for(auto mode:{llama_safetensors_io_mode::BUFFERED,llama_safetensors_io_mode::MMAP}){
            llama_safetensors_glm5_next_importer loader(root,cfg,mode);ggml_type type;std::array<int64_t,4> ne;
            const std::string bias_name=LLM_TN(LLM_ARCH_GLM5_NEXT)(LLM_TENSOR_FFN_EXP_PROBS_B,"bias",0).str();
            check(loader.describe(bias_name,type,ne)&&type==GGML_TYPE_F32&&ne[0]==2,"canonical expert correction bias missing");
            check(loader.materialize(bias_name,type,8)==f32({0.125f,-0.25f}),"expert correction bias changed");
            check(loader.describe("blk.0.ffn_gate_exps.weight",type,ne),"expert description missing");
            check(type==ggml_exl3_type(4,1)&&ne==std::array<int64_t,4>{128,128,2,1},"MCG expert type/geometry changed");
            ggml_init_params ip={1024*1024,nullptr,true};auto *ctx=ggml_init(ip);auto *w=ggml_new_tensor_3d(ctx,type,128,128,2);
            auto *backend=ggml_backend_cpu_init();auto *buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);
            check(loader.load("blk.0.ffn_gate_exps.weight",w,true),"bounded expert load failed");
            std::vector<uint8_t> bytes(ggml_nbytes(w));ggml_backend_tensor_get(w,bytes.data(),0,bytes.size());
            check(bytes.size()==16384,"packed EXL3 byte count changed");
            for(size_t i=0;i<bytes.size();++i)check(bytes[i]==uint8_t(i>=8192),"expert payload changed");
            ggml_backend_buffer_free(buffer);ggml_backend_free(backend);ggml_free(ctx);
            for(const char * side:{"scale","input_scale"}){
                const std::string n="blk.0.ffn_gate_exps."+std::string(side);check(loader.describe(n,type,ne),"missing EXL3 side vector");
                check(type==GGML_TYPE_F16&&ne==std::array<int64_t,4>{128,2,1,1},"EXL3 scale precision/shape changed");
            }
            check(loader.describe("token_embd.weight",type,ne)&&type==GGML_TYPE_BF16,"plain BF16 matrix changed");
            auto emb=loader.materialize("token_embd.weight",type,8);check(emb==std::vector<uint8_t>({0x80,0x3f,0,0x40,0xab,0x4a,0x91,0xbe}),"embedding bytes changed");
            check(loader.describe("output_norm.weight",type,ne)&&type==GGML_TYPE_F32,"GGML norm widening contract changed");
            auto norm=loader.materialize("output_norm.weight",type,8);check(norm==f32({1,2}),"lossless norm widening changed values");
            check(loader.describe("blk.0.ssm_a",type,ne)&&type==GGML_TYPE_F32&&ne[0]==2,"A transform shape");
            auto a=loader.materialize("blk.0.ssm_a",type,8);float af[2];std::memcpy(af,a.data(),8);check(af[0]==-1&&std::abs(af[1]+std::exp(0.5f))<1e-6,"A_log transform differs");
            check(loader.describe("blk.0.attn_k_b.weight",type,ne)&&ne==std::array<int64_t,4>{2,2,2,1},"K absorption shape");
            auto k=loader.materialize("blk.0.attn_k_b.weight",type,32);check(k==f32({0,2,1,3,8,10,9,11}),"K absorption permutation");
            auto v=loader.materialize("blk.0.attn_v_b.weight",type,32);check(v==f32({4,5,6,7,12,13,14,15}),"V absorption slicing");
            check(loader.describe("blk.0.ssm_conv1d_q.weight",type,ne)&&type==GGML_TYPE_F32&&ne==std::array<int64_t,4>{2,1,2,1},"conv reshape");
            auto conv=loader.materialize("blk.0.ssm_conv1d_q.weight",type,16);check(conv==f32({0.5f,2,8,32}),"conv widening changed source values");
            check(!loader.describe("blk.999.attn_q.weight",type,ne),"out-of-range layer accepted");
            rejected([&]{loader.materialize("blk.0.attn_k_b.weight",GGML_TYPE_F16,16);});
        }
        auto wrong=cfg;wrong["quantization_config"]["codebook"]="mul1";rejected([&]{llama_safetensors_glm5_next_importer l(root,wrong);});
        wrong=cfg;wrong["text_config"]["hidden_size"]=256;rejected([&]{llama_safetensors_glm5_next_importer l(root,wrong);ggml_type t;std::array<int64_t,4> n;l.describe("blk.0.ffn_gate_exps.weight",t,n);});
        std::filesystem::remove_all(root);std::cout<<"GLM5 native EXL3 importer: buffered/mmap payload, scales, BF16, KDA/MLA transforms and negative gates passed\n";return 0;
    }catch(const std::exception & e){std::cerr<<e.what()<<'\n';std::filesystem::remove_all(root);return 1;}
}
