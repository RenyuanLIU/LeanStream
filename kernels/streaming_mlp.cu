#include <cuda_fp16.h>
#include "streaming_mlp.cuh"
namespace leanstream::runtime {
__device__ __forceinline__ uint32_t acquire(const uint32_t* p) {
    uint32_t v;asm volatile("ld.acquire.sys.global.u32 %0, [%1];":"=r"(v):"l"(p):"memory");return v;
}
__device__ __forceinline__ uint64_t clock_ns(){uint64_t t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t));return t;}
__device__ __forceinline__ const uint8_t* get_group_ptr(const ExecDesc& d,const MLPBuffers& b,uint32_t bytes) {
    return (d.source_kind==SourceKind::Cache?b.cache:b.staging)+size_t(d.source_slot)*bytes;
}
__device__ float weight(const uint8_t* ptr,const __half* scale,uint32_t c) {
    int q=(ptr[c/2]>>((c%2)*4))&15;if(q>=8)q-=16;
    // Identical dequantization rounding to production weight() helper.
    return __half2float(__float2half_rn(float(q)*__half2float(scale[c/128])));
}
__device__ float sum256(float x,float* scratch) {
    int tid=threadIdx.x;scratch[tid]=x;__syncthreads();
    for(int stride=128;stride;stride/=2){if(tid<stride)scratch[tid]+=scratch[tid+stride];__syncthreads();}
    float result=scratch[0];__syncthreads();return result;
}
__device__ void compute(const ExecDesc& d,const MLPBuffers& b,uint32_t width,uint32_t g,float* scratch) {
    if(threadIdx.x==0 && b.first_compute)b.first_compute[d.dest_slot]=clock_ns();
    const uint8_t* ptr=get_group_ptr(d,b,3*width/2*g);
    for(uint32_t c=threadIdx.x;c<width;c+=256)b.partial[size_t(d.dest_slot)*width+c]=0;
    __syncthreads();
    for(uint32_t neuron=0;neuron<g;neuron++) {
        const uint8_t* row=ptr+size_t(neuron)*3*width/2;
        const __half* scale=b.scales+(size_t(d.physical_group)*g+neuron)*3*width/128;
        float gate=0,up=0;
        for(uint32_t c=threadIdx.x;c<width;c+=256) {
            float x=__half2float(b.input[c]);gate+=weight(row,scale,c)*x;
            up+=weight(row+width/2,scale+width/128,c)*x;
        }
        float ga=__half2float(__float2half_rn(sum256(gate,scratch)));
        float u=__half2float(__float2half_rn(sum256(up,scratch)));
        float silu=__half2float(__float2half_rn(ga/(1+expf(-ga))));
        float h=__half2float(__float2half_rn(silu*u));
        for(uint32_t c=threadIdx.x;c<width;c+=256)
            b.partial[size_t(d.dest_slot)*width+c]+=h*weight(row+width,scale+2*width/128,c);
        __syncthreads();
    }
    if(threadIdx.x==0){if(b.executions)atomicAdd(b.executions+d.dest_slot,1);}
}
__global__ void core_kernel(const ExecDesc* core,uint32_t count,MLPBuffers b,uint32_t width,uint32_t g) {
    __shared__ float scratch[256];if(blockIdx.x<count)compute(core[blockIdx.x],b,width,g,scratch);
}
__global__ void consumer(DeviceExecQueue q,MLPBuffers b,uint32_t epoch,uint32_t width,uint32_t g,bool batch) {
    __shared__ float scratch[256];__shared__ uint32_t position,total,fast;
    if(threadIdx.x==0) {
        while(acquire(&q.control->epoch)!=epoch && !acquire(&q.control->error))__nanosleep(128);
        total=q.control->total_count;
        if(batch)while(acquire(&q.control->batch_epoch)!=epoch&&!acquire(&q.control->error))__nanosleep(128);
        uint32_t proposed=acquire(&q.control->ready_count)==total?2:1;
        atomicCAS(&q.control->consumer_path,0,proposed);
        fast=atomicAdd(&q.control->consumer_path,0)==2;
    }
    __syncthreads();
    if(fast) { // all-ready static stripes: no claim atomics or empty polling
        for(uint32_t pos=blockIdx.x;pos<total;pos+=gridDim.x)
            compute(q.descriptors[q.ready_indices[pos]],b,width,g,scratch);
        return;
    }
    for(;;) {
        if(threadIdx.x==0) {
            position=total;
            for(;;) {
                if(acquire(&q.control->error))break;
                uint32_t ready=acquire(&q.control->ready_count);
                uint32_t head=atomicAdd(&q.control->claim_count,0);
                if(head>=total)break;
                if(head<ready && atomicCAS(&q.control->claim_count,head,head+1)==head){position=head;break;}
                __nanosleep(128);
            }
        }
        __syncthreads();if(position>=total)return;
        compute(q.descriptors[q.ready_indices[position]],b,width,g,scratch);
        __syncthreads();
    }
}
__global__ void reduction(DeviceExecQueue q,MLPBuffers b,__half* out,uint32_t core_count,uint32_t width) {
    uint32_t c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=width)return;
    float sum=0;for(uint32_t i=0;i<core_count+q.control->total_count;i++)sum+=b.partial[size_t(i)*width+c];
    out[c]=__float2half_rn(sum);
}
cudaError_t enqueue_streaming_mlp(Span<ExecDesc> a,DeviceExecQueue q,MLPBuffers b,__half* output,
    uint32_t epoch,uint32_t width,uint32_t g,uint32_t workers,ConsumerMode mode,cudaStream_t stream) {
    if(!epoch||!width||width%256||!g||!workers||!q.control||!q.descriptors||!q.ready_indices)return cudaErrorInvalidValue;
    if(a.size)core_kernel<<<a.size,256,0,stream>>>(a.data,a.size,b,width,g);
    // Conservative Auto until platform-matched production no-regression gate passes.
    consumer<<<workers,256,0,stream>>>(q,b,epoch,width,g,mode!=ConsumerMode::Progressive);
    reduction<<<(width+255)/256,256,0,stream>>>(q,b,output,a.size,width);
    return cudaGetLastError();
}
}
