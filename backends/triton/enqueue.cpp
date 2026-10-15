#include "leanstream/triton_cuda.h"
#include <cerrno>
#include <algorithm>
#include <new>
namespace leanstream::runtime {
int TritonKernel::bind(CUfunction f,uint32_t g,uint32_t w,uint32_t s,
                      const uint64_t* v,int n,int e,int c){
    if(!f||!g||!w||n<1||n>32||!v||e<0||e>=n||c>=n||c< -1)return EINVAL;
    function=f;grid=g;warps=w;shared_bytes=s;count=n;epoch_index=e;claim_index=c;
    std::copy_n(v,n,values.data());
    for(int i=0;i<n;i++)arguments[i]=values.data()+i;
    return 0;
}
int TritonKernel::launch(CUstream stream,uint32_t epoch,uint32_t claim){
    if(!function)return EINVAL;
    values[epoch_index]=epoch;if(claim_index>=0)values[claim_index]=claim;
    return int(cuLaunchKernel(function,grid,1,1,32*warps,1,1,shared_bytes,stream,arguments.data(),nullptr));
}
int TritonBackend::bind(uint32_t id,uint32_t stage,CUfunction f,uint32_t grid,uint32_t warps,
                        uint32_t shared,const uint64_t* args,int n,int epoch,int claim){
    if(id>=4||stage>2||warps!=4)return EINVAL;
    if(stage==1&&grid!=kTritonCandidates[id].cta_count)return EINVAL;
    auto& kernel=stage==0?core_:stage==2?reduction_:consumers_[id];
    return kernel.bind(f,grid,warps,shared,args,n,epoch,claim);
}
int TritonBackend::enqueue(KernelRuntimeState state,uint32_t epoch,cudaStream_t stream){
    uint32_t id=selector_.select(state);
    if(id>=4||!consumers_[id].function)id=kTritonDefault.id;
    if(!epoch||!core_.function||!reduction_.function||!consumers_[id].function)return EINVAL;
    if(claim_>=0x7fffffffU)return EOVERFLOW;
    uint32_t claim=++claim_;
    int rc=core_.launch(static_cast<CUstream>(stream),epoch,claim);if(rc)return rc;
    rc=consumers_[id].launch(static_cast<CUstream>(stream),epoch,claim);if(rc)return rc;
    if(arm_final_prediction){rc=arm_final_prediction(arm_context);if(rc)return rc;}
    if(b_done&&cudaEventRecord(b_done,stream)!=cudaSuccess)return EIO;
    return reduction_.launch(static_cast<CUstream>(stream),epoch,claim);
}
int TritonBackend::submit(void* p,uint32_t layer,const void*,void*,void* stream,
                         Span<ExecDesc>,DeviceExecQueue,uint32_t epoch,ConsumerMode mode){
    if(mode!=ConsumerMode::Progressive)return EINVAL;
    auto& b=*static_cast<TritonBackend*>(p);auto state=b.state;state.layer=layer;
    return b.enqueue(state,epoch,static_cast<cudaStream_t>(stream));
}
bool TritonBackend::complete(void* p,uint32_t layer){auto& b=*static_cast<TritonBackend*>(p);return b.completed&&b.completed(b.lifecycle_context,layer);}
int TritonBackend::commit(void* p,uint32_t layer,Span<GroupId> ids){auto& b=*static_cast<TritonBackend*>(p);return b.install_cache?b.install_cache(b.lifecycle_context,layer,ids):ENOTSUP;}
ComputeBackend TritonBackend::view(){return {this,submit,complete,commit};}
}
extern "C" {
void* ls_triton_create(){return new(std::nothrow) leanstream::runtime::TritonBackend;}
void* ls_triton_create_table(const uint8_t* entries,uint32_t layers,uint32_t buckets){
    return new(std::nothrow) leanstream::runtime::TritonBackend({entries,layers,buckets});
}
void ls_triton_seed_claim(void* p,uint32_t initial){
    if(p)static_cast<leanstream::runtime::TritonBackend*>(p)->seed_claim(initial);
}
void ls_triton_destroy(void* p){delete static_cast<leanstream::runtime::TritonBackend*>(p);}
int ls_triton_bind(void* p,uint32_t candidate,uint32_t stage,void* f,uint32_t grid,uint32_t warps,uint32_t shared,
                   const uint64_t* args,int count,int epoch,int claim){
    return p?static_cast<leanstream::runtime::TritonBackend*>(p)->bind(candidate,stage,static_cast<CUfunction>(f),grid,warps,shared,args,count,epoch,claim):EINVAL;
}
int ls_triton_enqueue(void* p,uint32_t layer,uint32_t bucket,uint32_t epoch,void* stream){
    return p?static_cast<leanstream::runtime::TritonBackend*>(p)->enqueue({layer,bucket},epoch,static_cast<cudaStream_t>(stream)):EINVAL;
}
}
