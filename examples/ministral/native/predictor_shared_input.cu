#include <cuda_runtime.h>
#include <cuda/atomic>
#include <stdint.h>

__global__ void publish_epoch(uint32_t* flag,uint32_t epoch) {
    cuda::atomic_ref<uint32_t,cuda::thread_scope_system> ready(*flag);
    ready.store(epoch,cuda::memory_order_release);
}
extern "C" cudaError_t si_signal(uint32_t* flag,uint32_t epoch,cudaStream_t stream) {
    publish_epoch<<<1,1,0,stream>>>(flag,epoch);
    return cudaGetLastError();
}

#include <cuda_fp16.h>
// One block, full 4096 values. This is RMSNorm/output, not a selected-index gather.
__global__ void mapped_rmsnorm(const __half* input,const __half* weight,float eps,
                               __half* output,uint32_t* flag,uint32_t epoch,int width,uint32_t* bell) {
    __shared__ float sums[256];
    float sum=0.f;
    for(int i=threadIdx.x;i<width;i+=256) {
        float x=__half2float(input[i]);sum+=x*x;
    }
    sums[threadIdx.x]=sum;__syncthreads();
    for(int width=128;width; width>>=1) {
        if(threadIdx.x<width)sums[threadIdx.x]+=sums[threadIdx.x+width];
        __syncthreads();
    }
    float inverse=rsqrtf(sums[0]/float(width)+eps);
    for(int i=threadIdx.x;i<width;i+=256) {
        // HF rounds normalized FP32 to half BEFORE multiplying the half weight.
        __half normalized=__float2half_rn(__half2float(input[i])*inverse);
        output[i]=__hmul(weight[i],normalized);
    }
    // Every writing thread flushes its writes to system scope, then thread 0 publishes.
    __threadfence_system();__syncthreads();
    if(threadIdx.x==0) {
        cuda::atomic_ref<uint32_t,cuda::thread_scope_system> ready(*flag);
        ready.store(epoch,cuda::memory_order_release);
        if(bell)cuda::atomic_ref<uint32_t,cuda::thread_scope_system>(*bell).fetch_add(1,cuda::memory_order_release);
    }
}
extern "C" cudaError_t si_norm(const void* input,const void* weight,float eps,void* output,
                               uint32_t* flag,uint32_t epoch,cudaStream_t stream,int width,uint32_t* bell) {
    mapped_rmsnorm<<<1,256,0,stream>>>(static_cast<const __half*>(input),static_cast<const __half*>(weight),
                                    eps,static_cast<__half*>(output),flag,epoch,width,bell);
    return cudaGetLastError();
}
