#include "llama-safetensors-glm5-next.h"
#include "llama-safetensors-metadata.h"
#include "llama-arch.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <regex>
#include <stdexcept>

namespace {
using role = llama_safetensors_quant_role;
const std::map<std::string, std::string> aliases = {
#include "llama-safetensors-glm5-next-map.inc"
    {"exp_probs_b.bias", "mlp.gate.e_score_correction_bias"},
};
void require(bool valid, const char * message) {
    if (!valid) throw std::runtime_error(std::string("GLM5-Next safetensors: ") + message);
}
}

bool llama_safetensors_glm5_next_importer::probe(const llama_safetensors_json & config) {
    const auto type = config.value("model_type", std::string());
    return type == "glm5_next" || type == "glm5_next_text";
}

llama_safetensors_glm5_next_importer::llama_safetensors_glm5_next_importer(
        const std::filesystem::path & root, llama_safetensors_json config, llama_safetensors_io_mode mode)
    : root_(root), config_(std::move(config)), text_(config_.contains("text_config") ? config_.at("text_config") : config_) {
    require(probe(config_), "unsupported model type");
    const auto & q = config_.at("quantization_config");
    require(q.at("quant_method") == "exl3" && q.at("codebook") == "mcg" && q.at("bits") == 4,
            "this B0 importer requires EXL3 MCG four-bit weights");
    require(q.at("scope") == "glm53_routed_experts_only", "wrong quantization scope");
    layers_ = text_.at("num_hidden_layers").get<uint32_t>();
    mtp_ = text_.value("num_nextn_predict_layers", 0U);
    experts_ = text_.at("n_routed_experts").get<uint32_t>();
    require(layers_ > 0 && layers_ + mtp_ <= 512 && experts_ > 0, "invalid layer/expert geometry");
    require(text_.at("layer_types").size() == layers_ && text_.at("hc_mult") == 4,
            "invalid hybrid or mHC geometry");
    require(text_.at("mla_use_nope").get<bool>() && text_.at("qk_rope_head_dim") == 0,
            "only NOPE MLA is supported");
    require(text_.at("linear_attn_config").at("num_heads") == text_.at("num_attention_heads"),
            "KDA and attention head counts differ");
    generation_ = llama_safetensors_read_json(root_ / "generation_config.json");
    tokenizer_ = llama_safetensors_read_tokenizer_json(root_ / "tokenizer.json");
    chat_template_ = llama_safetensors_read_optional_text(root_ / "chat_template.jinja");
    registry_ = llama_safetensors_registry::load(root_, mode);
    quant_ = std::make_unique<llama_safetensors_quant_adapters>(config_, registry_);
}

gguf_context * llama_safetensors_glm5_next_importer::build_metadata() const {
    llama_safetensors_metadata_sink sink;
    LLM_KV key(LLM_ARCH_GLM5_NEXT);
    const auto u = [&](llm_kv k, const char * field) { sink.set_u32(key(k), text_.at(field).get<uint32_t>()); };
    const auto f = [&](llm_kv k, const char * field) { sink.set_f32(key(k), text_.at(field).get<float>()); };
    sink.set_string("general.architecture", "glm5-next");
    sink.set_string("general.type", "model");
    sink.set_string("general.name", root_.filename().string());
    sink.set_u32("general.file_type", quant_->file_type());
    sink.set_u32(key(LLM_KV_BLOCK_COUNT), layers_ + mtp_);
    if (mtp_) sink.set_u32(key(LLM_KV_NEXTN_PREDICT_LAYERS), mtp_);
    llama_safetensors_emit_sampling_defaults(sink, generation_);
    u(LLM_KV_CONTEXT_LENGTH,"max_position_embeddings");
    u(LLM_KV_EMBEDDING_LENGTH,"hidden_size");
    u(LLM_KV_FEED_FORWARD_LENGTH,"intermediate_size");
    u(LLM_KV_ATTENTION_HEAD_COUNT,"num_attention_heads");
    u(LLM_KV_VOCAB_SIZE,"vocab_size");
    f(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,"rms_norm_eps");
    sink.set_f32(key(LLM_KV_ATTENTION_LAYERNORM_EPS),1e-6f);
    std::vector<uint32_t> kv;
    for (const auto & kind : text_.at("layer_types")) {
        require(kind == "linear_attention" || kind == "deepseek_sparse_attention", "unknown layer type");
        kv.push_back(kind == "linear_attention" ? 0 : 1);
    }
    kv.resize(layers_ + mtp_,1);
    sink.set_u32_array(key(LLM_KV_ATTENTION_HEAD_COUNT_KV),kv.data(),kv.size());
    const auto & lin = text_.at("linear_attn_config");
    sink.set_u32(key(LLM_KV_SSM_CONV_KERNEL),lin.at("short_conv_kernel_size").get<uint32_t>());
    sink.set_u32(key(LLM_KV_KDA_HEAD_DIM),lin.at("head_dim").get<uint32_t>());
    sink.set_f32(key(LLM_KV_KDA_GATE_LOWER_BOUND),lin.value("gate_lower_bound",-5.0f));
    u(LLM_KV_ATTENTION_Q_LORA_RANK,"q_lora_rank");u(LLM_KV_ATTENTION_KV_LORA_RANK,"kv_lora_rank");
    u(LLM_KV_ATTENTION_KEY_LENGTH,"kv_lora_rank");u(LLM_KV_ATTENTION_VALUE_LENGTH,"kv_lora_rank");
    u(LLM_KV_ATTENTION_KEY_LENGTH_MLA,"qk_nope_head_dim");u(LLM_KV_ATTENTION_VALUE_LENGTH_MLA,"v_head_dim");
    sink.set_u32(key(LLM_KV_ROPE_DIMENSION_COUNT),0);
    u(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT,"index_n_heads");u(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH,"index_head_dim");
    u(LLM_KV_ATTENTION_INDEXER_TOP_K,"index_topk");u(LLM_KV_ATTENTION_INDEXER_KPOOL,"index_kpool");
    sink.set_bool(key(LLM_KV_ATTENTION_INDEXER_KPOOL_SELECT_TAIL),text_.value("index_kpool_always_select_tail",true));
    std::vector<uint32_t> full(layers_,1);
    if (text_.contains("indexer_types")) {
        require(text_.at("indexer_types").size()==layers_,"indexer type count differs from layers");
        for (size_t i=0;i<layers_;++i) full[i]=text_.at("indexer_types")[i]=="full";
    }
    sink.set_u32_array(key(LLM_KV_ATTENTION_INDEXER_TYPES),full.data(),full.size());
    u(LLM_KV_HYPER_CONNECTION_COUNT,"hc_mult");u(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS,"hc_sinkhorn_iters");
    f(LLM_KV_HYPER_CONNECTION_EPSILON,"hc_eps");
    u(LLM_KV_LEADING_DENSE_BLOCK_COUNT,"first_k_dense_replace");u(LLM_KV_EXPERT_COUNT,"n_routed_experts");
    u(LLM_KV_EXPERT_USED_COUNT,"num_experts_per_tok");u(LLM_KV_EXPERT_SHARED_COUNT,"n_shared_experts");
    u(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,"moe_intermediate_size");f(LLM_KV_EXPERT_WEIGHTS_SCALE,"routed_scaling_factor");
    sink.set_bool(key(LLM_KV_EXPERT_WEIGHTS_NORM),text_.at("norm_topk_prob").get<bool>());
    const std::vector<float> clamps(layers_+mtp_,text_.value("swiglu_limit",10.0f));
    sink.set_f32_array(key(LLM_KV_SWIGLU_CLAMP_EXP),clamps.data(),clamps.size());
    sink.set_f32_array(key(LLM_KV_SWIGLU_CLAMP_SHEXP),clamps.data(),clamps.size());
    const auto token_id = [&](const std::string & content) {
        for (const auto & t : tokenizer_.at("added_tokens")) if(t.at("content")==content) return t.at("id").get<uint32_t>();
        throw std::runtime_error("missing GLM special token "+content);
    };
    llama_safetensors_emit_bpe_tokenizer(sink,tokenizer_,{"glm4",text_.at("vocab_size").get<uint32_t>(),token_id("[gMASK]"),
        llama_safetensors_first_token_id(generation_.at("eos_token_id"),"eos_token_id"),std::string("<|endoftext|>"),true,{},false},chat_template_);
    sink.set_u32("tokenizer.ggml.eot_token_id",token_id("<|user|>"));
    sink.set_u32("tokenizer.ggml.eom_token_id",token_id("<|observation|>"));
    sink.set_u32("tokenizer.ggml.unknown_token_id",token_id("<|endoftext|>"));
    return sink.release();
}

llama_safetensors_glm5_next_importer::binding llama_safetensors_glm5_next_importer::resolve(const std::string & target) const {
    binding out;
    std::string suffix=target,prefix;
    std::smatch match;
    static const std::regex block(R"(^blk\.([0-9]+)\.(.+)$)");
    if(std::regex_match(target,match,block)) {
        const uint64_t layer=std::stoull(match[1]);
        if(layer>=layers_+mtp_)return out;
        prefix="model.language_model.layers."+std::to_string(layer)+".";suffix=match[2];
    }
    static const std::regex expert(R"(^ffn_(gate|up|down)_exps\.(weight|scale|input_scale)$)");
    if(!prefix.empty() && std::regex_match(suffix,match,expert)) {
        const role kind=match[2]=="weight"?role::WEIGHT:match[2]=="scale"?role::WEIGHT_SCALE:role::INPUT_SCALE;
        for(uint32_t e=0;e<experts_;++e) {
            auto b=quant_->bind(prefix+"mlp.experts."+std::to_string(e)+"."+match[1].str()+"_proj",kind);
            require(b.has_value(),"missing routed expert or side tensor");
            if(kind==role::WEIGHT)require(b->target_type==ggml_exl3_type(4,1),"expert must use B0 MCG four-bit codebook");
            const int64_t hidden=text_.at("hidden_size"),inner=text_.at("moe_intermediate_size");
            const int64_t k=match[1]=="down"?inner:hidden,n=match[1]=="down"?hidden:inner;
            const std::vector<int64_t> expected=kind==role::WEIGHT?std::vector<int64_t>{k,n}:std::vector<int64_t>{kind==role::INPUT_SCALE?k:n};
            require(b->target_shape==expected,"expert geometry differs from GLM config");
            out.parts.push_back({b->primary,std::move(b)});
        }
        out.stack=true;return out;
    }
    if(!prefix.empty() && (suffix=="attn_k_b.weight" || suffix=="attn_v_b.weight")) {
        out.op=suffix=="attn_k_b.weight"?binding::MLA_K:binding::MLA_V;
        out.parts.push_back({prefix+"self_attn.kv_b_proj.weight",std::nullopt});return out;
    }
    const auto it=aliases.find(suffix);
    if(it==aliases.end())return out;
    const std::string source=prefix+it->second;
    if(!registry_.find(source))return out;
    out.parts.push_back({source,std::nullopt});
    if(suffix=="ssm_a")out.op=binding::EXP_A;
    out.force_f32=suffix.rfind("hc_",0)==0 || suffix.rfind("indexer_compressor_",0)==0 ||
        suffix.rfind("ssm_conv1d_",0)==0 || suffix=="ssm_dt.bias" || suffix=="exp_probs_b.bias";
    return out;
}

bool llama_safetensors_glm5_next_importer::describe(const std::string & target,ggml_type & type,std::array<int64_t,GGML_MAX_DIMS> & ne) const {
    const auto b=resolve(target);if(b.parts.empty())return false;
    if(!llama_safetensors_describe_tensor(registry_,b.parts[0],type,ne))return false;
    if(b.stack) {
        size_t dim=b.parts[0].quant->target_shape.size();require(dim<GGML_MAX_DIMS,"expert stack rank too high");
        for(const auto & p:b.parts)require(p.quant->target_type==type && p.quant->target_shape==b.parts[0].quant->target_shape,"mixed expert geometry");
        ne[dim]=experts_;
    } else if(b.op==binding::EXP_A) {
        type=GGML_TYPE_F32;ne={text_.at("num_attention_heads").get<int64_t>(),1,1,1};
    } else if(b.op==binding::MLA_K || b.op==binding::MLA_V) {
        const int64_t heads=text_.at("num_attention_heads"), k=text_.at("kv_lora_rank"),q=text_.at("qk_nope_head_dim"),v=text_.at("v_head_dim");
        require(ne[0]==k && ne[1]==heads*(q+v) && ne[2]==1 && ne[3]==1,"invalid MLA absorption source shape");
        require(type==GGML_TYPE_F32 || type==GGML_TYPE_F16 || type==GGML_TYPE_BF16,"unsupported MLA absorption dtype");
        ne=b.op==binding::MLA_K?std::array<int64_t,4>{q,k,heads,1}:std::array<int64_t,4>{k,v,heads,1};
    } else if(target.find("ssm_conv1d_")!=std::string::npos) {
        const auto * source=registry_.find(b.parts[0].source);require(source->shape.size()==2 || source->shape.size()==3,"invalid conv rank");
        const int64_t width=source->shape.front(),kernel=source->shape.back();
        require(source->shape.size()!=3 || source->shape[1]==1,"invalid conv middle dimension");ne={kernel,1,width,1};
    }
    if(b.force_f32)type=GGML_TYPE_F32;
    return true;
}
size_t llama_safetensors_glm5_next_importer::tensor_capacity_hint() const {return std::max<size_t>(2048,(layers_+mtp_)*80);}
void llama_safetensors_glm5_next_importer::bind(const std::string & name) const {
    for(const auto & p:resolve(name).parts)llama_safetensors_consume_tensor(*quant_,p);
}
bool llama_safetensors_glm5_next_importer::load(const std::string & name,ggml_tensor * dst,bool check) const {
    const auto b=resolve(name);if(b.parts.empty())return false;
    if(b.stack) {
        require(ggml_nbytes(dst)%b.parts.size()==0,"nonintegral expert plane");const size_t plane=ggml_nbytes(dst)/b.parts.size();
        for(size_t e=0;e<b.parts.size();++e) {
            auto bytes=quant_->finalize(*b.parts[e].quant,quant_->read(*b.parts[e].quant));
            require(bytes.size()==plane,"expert plane byte count differs");
            if(check)require(ggml_validate_row_data(dst->type,bytes.data(),bytes.size()),"invalid expert row data");
            ggml_backend_tensor_set(dst,bytes.data(),e*plane,plane);
        }return true;
    }
    return b.op==binding::NONE && !b.force_f32 && llama_safetensors_load_tensor_direct(registry_,b.parts[0],dst,check);
}
std::vector<uint8_t> llama_safetensors_glm5_next_importer::materialize(const std::string & name,ggml_type type,size_t size) const {
    const auto b=resolve(name);require(b.parts.size()==1 && !b.stack,"experts require bounded upload");
    const auto * src=registry_.find(b.parts[0].source);require(src!=nullptr,"missing source tensor");
    if(b.op==binding::EXP_A) {
        auto bytes=registry_.read(*src);
        if(src->dtype==llama_safetensors_dtype::BF16)bytes=llama_safetensors_bf16_to_f32(bytes);
        else if(src->dtype==llama_safetensors_dtype::F16)bytes=llama_safetensors_f16_to_f32(bytes);
        else require(src->dtype==llama_safetensors_dtype::F32,"invalid A_log dtype");
        require(type==GGML_TYPE_F32 && bytes.size()>=size && size==text_.at("num_attention_heads").get<size_t>()*4,"invalid A_log shape");
        bytes.resize(size);
        for(size_t i=0;i<size;i+=4){float x;std::memcpy(&x,bytes.data()+i,4);x=-std::exp(x);require(std::isfinite(x),"nonfinite A_log");std::memcpy(bytes.data()+i,&x,4);}return bytes;
    }
    if(b.op==binding::MLA_K || b.op==binding::MLA_V) {
        ggml_type source_type;std::array<int64_t,4> shape{};require(llama_safetensors_describe_tensor(registry_,b.parts[0],source_type,shape),"cannot describe MLA");
        require(type==source_type,"MLA transform must preserve dtype");
        const auto bytes=registry_.read(*src);const size_t width=ggml_type_size(type),h=text_.at("num_attention_heads"),k=text_.at("kv_lora_rank"),q=text_.at("qk_nope_head_dim"),v=text_.at("v_head_dim");
        const size_t n=b.op==binding::MLA_K?q:v;require(size==h*k*n*width && bytes.size()==h*(q+v)*k*width,"MLA byte count mismatch");std::vector<uint8_t> out(size);
        for(size_t head=0;head<h;++head)for(size_t row=0;row<n;++row)for(size_t col=0;col<k;++col){
            const size_t from=(head*(q+v)+row+(b.op==binding::MLA_V?q:0))*k+col;
            const size_t to=b.op==binding::MLA_K?(head*k+col)*q+row:(head*v+row)*k+col;
            std::memcpy(out.data()+to*width,bytes.data()+from*width,width);
        }return out;
    }
    return llama_safetensors_materialize_tensor(registry_,*quant_,b.parts[0],type,size);
}
void llama_safetensors_glm5_next_importer::validate_complete() const {quant_->validate_complete();}
