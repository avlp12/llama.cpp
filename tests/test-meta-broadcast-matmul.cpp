#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
struct config {int axis;};
static ggml_backend_meta_split_state split(const ggml_tensor * t, void * ptr) {
    if (!std::strcmp(t->name,"rhs")) return {ggml_backend_meta_split_axis(static_cast<config *>(ptr)->axis),{2,2},{1},1};
    return {GGML_BACKEND_SPLIT_AXIS_MIRRORED,{0},{1},1};
}
static void run(bool invalid=false) {
    const int axis=2;config c{axis};auto * reg=ggml_backend_cpu_reg();ggml_backend_dev_t devs[]={ggml_backend_reg_dev_get(reg,0),ggml_backend_reg_dev_get(reg,0)};
    auto * backend=ggml_backend_dev_init(ggml_backend_meta_device(devs,2,split,&c),nullptr);
    auto * weights=ggml_init({1024*1024,nullptr,true});auto * compute=ggml_init({1024*1024,nullptr,true});
    const int k=8,n=3,m=2,a2=invalid?4:1,a3=2,b2=4,b3=2;
    auto * a=ggml_new_tensor_4d(weights,GGML_TYPE_F32,k,n,a2,a3);ggml_set_name(a,"lhs");
    auto * b=ggml_new_tensor_4d(weights,GGML_TYPE_F32,k,m,b2,b3);ggml_set_name(b,"rhs");
    auto * mask=ggml_new_tensor_4d(weights,GGML_TYPE_F32,n,m,1,b3);ggml_set_name(mask,"mask");
    auto * y=ggml_mul_mat(compute,a,b);
    auto * soft=ggml_soft_max_ext(compute,y,mask,0.25f,0.0f);ggml_set_output(soft);
    ggml_set_output(y);auto * graph=ggml_new_graph_custom(compute,16,false);ggml_build_forward_expand(graph,soft);
    auto * buf=ggml_backend_alloc_ctx_tensors(weights,backend);ggml_backend_buffer_set_usage(buf,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    auto * allocator=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));GGML_ASSERT(ggml_gallocr_alloc_graph(allocator,graph));
    std::vector<float> av(ggml_nelements(a)),bv(ggml_nelements(b)),actual(ggml_nelements(y));
    for(size_t i=0;i<av.size();++i)av[i]=int(i%7)-3;
    for(size_t i=0;i<bv.size();++i)bv[i]=int(i%11)-5;
    ggml_backend_tensor_set(a,av.data(),0,av.size()*4);ggml_backend_tensor_set(b,bv.data(),0,bv.size()*4);
    std::vector<float> mv(ggml_nelements(mask));
    for(size_t i=0;i<mv.size();++i)mv[i]=-0.5f*float(i%3);
    ggml_backend_tensor_set(mask,mv.data(),0,mv.size()*4);
    GGML_ASSERT(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS);ggml_backend_tensor_get(y,actual.data(),0,actual.size()*4);
    for(int j3=0;j3<b3;++j3)for(int j2=0;j2<b2;++j2)for(int im=0;im<m;++im)for(int in=0;in<n;++in){
        float expected=0;for(int ik=0;ik<k;++ik)expected+=av[(((j3%a3)*a2+j2%a2)*n+in)*k+ik]*bv[((j3*b2+j2)*m+im)*k+ik];
        GGML_ASSERT(actual[((j3*b2+j2)*m+im)*n+in]==expected);
    }
    std::vector<float> probabilities(actual.size());ggml_backend_tensor_get(soft,probabilities.data(),0,probabilities.size()*4);
    for(int j3=0;j3<b3;++j3)for(int j2=0;j2<b2;++j2)for(int im=0;im<m;++im){
        double sum=0;std::vector<double> exps(n);
        for(int in=0;in<n;++in){exps[in]=std::exp(0.25*actual[((j3*b2+j2)*m+im)*n+in]+mv[(j3*m+im)*n+in]);sum+=exps[in];}
        for(int in=0;in<n;++in)GGML_ASSERT(std::abs(probabilities[((j3*b2+j2)*m+im)*n+in]-exps[in]/sum)<1e-6);
    }
    ggml_gallocr_free(allocator);ggml_backend_buffer_free(buf);ggml_free(compute);ggml_free(weights);ggml_backend_free(backend);
    std::printf("BROADCAST_MATMUL axis=%d exact and masked softmax PASS\n",axis);
}
int main(int argc,char **argv){if(argc==2&&!std::strcmp(argv[1],"--invalid-left-batch")){run(true);return 1;}run();}
