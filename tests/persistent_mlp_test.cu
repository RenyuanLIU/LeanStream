#include "../kernels/streaming_mlp.cuh"
#include <cassert>
#include <cstdio>
#include <vector>
#include <thread>
#include <chrono>
#include <cmath>
#include <cerrno>
using namespace leanstream::runtime;
#define CUDA(x) do {auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));abort();}}while(0)
template<class T> T* device(size_t n){T* p;CUDA(cudaMalloc(&p,n*sizeof(T)));return p;}
template<class T> T* mapped(size_t n){T* p;CUDA(cudaHostAlloc(&p,n*sizeof(T),cudaHostAllocMapped));return p;}
struct Result {std::vector<float> output;};
static Result run(uint32_t count,bool progressive,bool delayed,bool shuffled,int quantized=1,bool injected_error=false,uint32_t workers=4) {
    constexpr uint32_t H=256,G=2;uint32_t all=count+1;size_t weight_bytes=size_t(all)*G*3*H/2;
    std::vector<uint8_t> w(weight_bytes,quantized==1?0x11:0xff);std::vector<__half> scale(all*G*3*H/128,__float2half(.05f)),x(H,__float2half(.1f));
    auto* dw=device<uint8_t>(weight_bytes);auto* staging=device<uint8_t>(weight_bytes);auto* ds=device<__half>(scale.size());auto* dx=device<__half>(H);
    CUDA(cudaMemcpy(dw,w.data(),weight_bytes,cudaMemcpyHostToDevice));CUDA(cudaMemcpy(staging,w.data(),weight_bytes,cudaMemcpyHostToDevice));CUDA(cudaMemcpy(ds,scale.data(),scale.size()*2,cudaMemcpyHostToDevice));CUDA(cudaMemcpy(dx,x.data(),H*2,cudaMemcpyHostToDevice));
    auto* desc=mapped<ExecDesc>(all);auto* indices=mapped<uint32_t>(all);auto* ctl=mapped<QueueControl>(1);
    auto* counters=mapped<uint32_t>(all);auto* starts=mapped<uint64_t>(all);
    *ctl=QueueControl{};for(uint32_t i=0;i<all;i++){desc[i]={i,i,i,i,SourceKind::Cache};counters[i]=0;starts[i]=0;}
    auto* da=device<ExecDesc>(1);CUDA(cudaMemcpy(da,desc,sizeof(ExecDesc),cudaMemcpyHostToDevice));
    // B table excludes A, but preserves deterministic dest slots 1..N.
    for(uint32_t i=0;i<count;i++)desc[i]={i+1,i+1,i+1,i+1,(i%2?SourceKind::Cache:SourceKind::Staging)};
    DeviceExecQueue q{desc,indices,ctl,count};ExecutionQueue queue(q);queue.reset(1);queue.seal(count);
    std::vector<uint32_t> order(count);for(uint32_t i=0;i<count;i++)order[i]=shuffled?count-1-i:i;
    uint32_t first=delayed?count/10:count;
    for(uint32_t i=0;i<first;i++)queue.publish(order[i]);
    auto* partial=device<float>(all*H);auto* y=device<__half>(H);
    MLPBuffers buffers{dx,dw,staging,ds,partial,counters,starts};cudaStream_t stream;CUDA(cudaStreamCreate(&stream));
    CUDA(enqueue_streaming_mlp({da,1},q,buffers,y,1,H,G,workers,progressive?ConsumerMode::Progressive:ConsumerMode::BatchCompat,stream));
    if(delayed){
        // Bounded publication phases: first subset must compute before later release.
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        if(progressive&&first){uint32_t done=0;for(uint32_t i=1;i<all;i++)done+=__atomic_load_n(counters+i,__ATOMIC_ACQUIRE);assert(done==first);assert(counters[0]==1);}
        if(injected_error)queue.fail(EIO);
        if(!injected_error)for(uint32_t i=first;i<count*3/10;i++)queue.publish(order[i]);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        if(!injected_error)for(uint32_t i=count*3/10;i<count;i++)queue.publish(order[i]);
    }
    cudaEvent_t finished;CUDA(cudaEventCreateWithFlags(&finished,cudaEventDisableTiming));CUDA(cudaEventRecord(finished,stream));
    auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
    while(cudaEventQuery(finished)==cudaErrorNotReady){assert(std::chrono::steady_clock::now()<deadline);std::this_thread::sleep_for(std::chrono::milliseconds(1));}
    CUDA(cudaEventSynchronize(finished));CUDA(cudaEventDestroy(finished));
    if(injected_error){assert(ctl->error==EIO);for(uint32_t i=1;i<all;i++)assert(counters[i]==(i>count-first));}
    Result returned;
    if(!injected_error){
    for(uint32_t i=0;i<all;i++)assert(counters[i]==1);
    uint64_t a_start=starts[0];for(uint32_t i=1;i<all;i++)assert(starts[i]>=a_start);
    std::vector<__half> out(H);CUDA(cudaMemcpy(out.data(),y,H*2,cudaMemcpyDeviceToHost));
    Result result;for(auto h:out){float v=__half2float(h);assert(std::isfinite(v));result.output.push_back(v);}
    // Independent dense tiny INT4=1 arithmetic sanity; half-rounded gate/up/SiLU.
    float gate=__half2float(__float2half(float(H)*__half2float(x[0])*__half2float(scale[0])*quantized));
    float act=__half2float(__float2half(__half2float(__float2half(gate/(1+expf(-gate))))*gate));
    float expected=__half2float(__float2half(all*G*act*__half2float(scale[0])*quantized));
    for(float v:result.output){if(fabs(v-expected)>=.002f)fprintf(stderr,"n=%u actual=%g expected=%g gate=%g act=%g\n",count,v,expected,gate,act);assert(fabs(v-expected)<.002f);}
    returned=result;
    }
    CUDA(cudaFree(dw));CUDA(cudaFree(staging));CUDA(cudaFree(ds));CUDA(cudaFree(dx));CUDA(cudaFree(da));CUDA(cudaFree(partial));CUDA(cudaFree(y));
    CUDA(cudaFreeHost(desc));CUDA(cudaFreeHost(indices));CUDA(cudaFreeHost(ctl));CUDA(cudaFreeHost(counters));CUDA(cudaFreeHost(starts));CUDA(cudaStreamDestroy(stream));
    return returned;
}
int main(){CUDA(cudaSetDevice(0));for(uint32_t workers:{1u,2u,4u,8u,16u})run(16,true,false,true,1,false,workers);run(37,true,true,true,1,true);for(uint32_t n:{0,1,2,16,37}){
    auto batch=run(n,false,false,false);auto fast=run(n,true,false,true);auto progressive=run(n,true,n>=4,true);
    assert(batch.output==fast.output&&batch.output==progressive.output);
    auto negative_batch=run(n,false,false,false,-1);auto negative_progressive=run(n,true,n>=4,true,-1);
    assert(negative_batch.output==negative_progressive.output);
 }puts("PASS: real INT4 MLP, BatchCompat=Progressive bit-exact, shuffled 10/20/70 publication, finite error exit, all-ready, zero/small B, exactly once, A precedes B");}
