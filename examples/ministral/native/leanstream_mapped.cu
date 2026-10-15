// Persistent mapped payloads. GPU-only caches/intermediates are never allocated here.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda/atomic>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <cstdio>
#include "leanstream_control.h"
struct Mapped {
    void *host=nullptr,*device=nullptr;
    size_t bytes=0;
    ~Mapped(){if(host)cudaFreeHost(host);}
    cudaError_t alloc(size_t n){bytes=n;auto e=cudaHostAlloc(&host,n,cudaHostAllocMapped);
        if(e==cudaSuccess)e=cudaHostGetDevicePointer(&device,host,0);
        if(e==cudaSuccess)std::memset(host,0,n);return e;}
};
#ifdef STREAM_IO_V2
static constexpr int packed_slots=7168;
#else
static constexpr int packed_slots=256;
#endif
struct Payloads {
    int group_size=2,active=1434,bytes=12288,hidden_size=4096,groups=7168,fill_size=215,max_exec=1434;
    Mapped hidden[2],importance,flags,packed,metadata;
#ifdef STREAM_IO_V2
    Mapped stream_states;
#endif
    uint32_t epochs[3]{};
    uint32_t* notify_device=nullptr; // Optional wake hint; mapped epochs own visibility.
};
static thread_local char error[192];
static int fail(const char* s){std::snprintf(error,sizeof(error),"%s",s);return -1;}
static int checked(cudaError_t e){return e==cudaSuccess?0:fail(cudaGetErrorString(e));}
__device__ void publish(uint32_t* flag,uint32_t epoch){
    __threadfence_system();__syncthreads();
    if(threadIdx.x==0)cuda::atomic_ref<uint32_t,cuda::thread_scope_system>(*flag).store(epoch,cuda::memory_order_release);
}
__global__ void hidden_output(const __half* in,__half* out,uint32_t* flag,uint32_t epoch,int width,uint32_t* bell){
    for(int i=threadIdx.x;i<width;i+=blockDim.x)out[i]=in[i];publish(flag,epoch);
    if(bell&&threadIdx.x==0)cuda::atomic_ref<uint32_t,cuda::thread_scope_system>(*bell).fetch_add(1,cuda::memory_order_release);
}
__global__ void importance_output(const __half* h,float* out,uint32_t* flag,uint32_t epoch,int g=2,int count=1434){
    for(int i=threadIdx.x;i<count;i+=blockDim.x){float total=0;for(int j=0;j<g;j++)total+=fabsf(__half2float(h[g*i+j]));out[i]=total;}
    publish(flag,epoch);
}
__global__ void importance_output_variable(const __half* h,float* out,uint32_t* flag,uint32_t epoch,
                                           const int32_t* exec_count,int g,int max_exec){
    int count=*exec_count;
    if(count<0||count>max_exec)return;
    for(int i=threadIdx.x;i<count;i+=blockDim.x){float total=0;for(int j=0;j<g;j++)total+=fabsf(__half2float(h[g*i+j]));out[i]=total;}
    publish(flag,epoch);
}
// Metadata records: staging slot, physical group, destination cache group.
__global__ void mapped_dequant(const uint8_t* packed,const int32_t* meta,int n,const __half* scales,__half* cache,int width){
    int item=blockIdx.x; if(item>=n)return;
    int source=meta[3*item],physical=meta[3*item+1],dest=meta[3*item+2];
    for(int i=threadIdx.x;i<2*3*width;i+=blockDim.x){
        int member=i/(3*width),projection=(i/width)%3,col=i%width;
        uint8_t b=packed[size_t(source)*(3*width)+member*(3*width/2)+projection*(width/2)+col/2];
        int q=(col&1)?b>>4:b&15;if(q>=8)q-=16;
        float scale=__half2float(scales[((physical*2+member)*3+projection)*(width/128)+col/128]);
        cache[((dest*2+member)*3+projection)*width+col]=__float2half_rn(float(q)*scale);
    }
}
extern "C" {
const char* lm_error(){return error;}
void lm_destroy(void* p){delete static_cast<Payloads*>(p);}
void* lm_create(){
    int d=0,map=0,integrated=0,atomic=0;cudaGetDevice(&d);
    cudaDeviceGetAttribute(&map,cudaDevAttrCanMapHostMemory,d);
    cudaDeviceGetAttribute(&integrated,cudaDevAttrIntegrated,d);
    cudaDeviceGetAttribute(&atomic,cudaDevAttrHostNativeAtomicSupported,d);
    if(!map||!integrated||!atomic){fail("Mapped release/acquire runtime requires integrated GPU with host native atomics");return nullptr;}
    auto* p=new Payloads;
    if(checked(p->hidden[0].alloc(8192))||checked(p->hidden[1].alloc(8192))||
       checked(p->importance.alloc(1434*4))||checked(p->flags.alloc(3*64))||
       checked(p->packed.alloc(packed_slots*12288))||checked(p->metadata.alloc(1434*3*4))){delete p;return nullptr;}
#ifdef STREAM_IO_V2
    if(checked(p->stream_states.alloc((2*7168+1)*sizeof(uint32_t)))){delete p;return nullptr;}
#endif
    return p;
}
int lm_configure(void* ptr,int g){
    auto& p=*static_cast<Payloads*>(ptr);if(g<2||g>4||p.epochs[0]||p.epochs[1]||p.epochs[2])return fail("Invalid mapped configuration");
    int bytes=g*6144;
    if(bytes!=p.bytes){if(checked(cudaFreeHost(p.packed.host)))return -1;p.packed.host=nullptr;p.packed.device=nullptr;
        if(checked(p.packed.alloc(packed_slots*bytes)))return -1;}
    p.group_size=g;p.active=2868/g;p.bytes=bytes;return 0;
}
int lm_configure_model(void* ptr,int h,int intermediate,int groups,int active,int fill){
    auto& p=*static_cast<Payloads*>(ptr);
    if(p.epochs[0]||p.epochs[1]||p.epochs[2]||h<128||h>4096||h%128||
       intermediate!=groups*2||groups<1||groups>11520||active<1||active>2304||
       fill<1||fill>346)return fail("Invalid mapped model geometry");
    auto resize=[&](Mapped& m,size_t bytes){
        if(m.bytes==bytes)return 0;
        if(checked(cudaFreeHost(m.host)))return -1;
        m.host=m.device=nullptr;return checked(m.alloc(bytes));
    };
    if(resize(p.hidden[0],size_t(h)*2)||resize(p.hidden[1],size_t(h)*2)||
       resize(p.importance,size_t(active)*4)||resize(p.packed,size_t(groups)*3*h)||
       resize(p.metadata,size_t(active)*3*4))return -1;
#ifdef STREAM_IO_V2
    if(resize(p.stream_states,size_t(2*groups+1)*sizeof(uint32_t)))return -1;
#endif
    p.group_size=2;p.active=active;p.max_exec=active;p.bytes=3*h;p.hidden_size=h;p.groups=groups;p.fill_size=fill;
    return 0;
}
int lm_no_waste_config(void* ptr,int max_exec){
    auto& p=*static_cast<Payloads*>(ptr);
    if(p.epochs[2]||max_exec<p.active||max_exec>p.groups)return fail("Invalid no-waste importance capacity");
    if(checked(cudaFreeHost(p.importance.host)))return -1;
    p.importance.host=p.importance.device=nullptr;
    if(checked(p.importance.alloc(size_t(max_exec)*sizeof(float))))return -1;
    p.max_exec=max_exec;return 0;
}
void* lm_pointer(void* ptr,int kind,int device){
    auto& p=*static_cast<Payloads*>(ptr);Mapped* m=nullptr;
    switch(kind){case 0:case 1:m=&p.hidden[kind];break;case 2:m=&p.importance;break;
        case 3:m=&p.packed;break;case 4:m=&p.metadata;break;
#ifdef STREAM_IO_V2
        case 5:m=&p.stream_states;break;
#endif
        default:return nullptr;}
    return device?m->device:m->host;
}
int lm_wait(void* ptr,int slot){
    auto& p=*static_cast<Payloads*>(ptr);if(slot<0||slot>2)return fail("Invalid epoch slot");
    auto* flag=static_cast<uint32_t*>(p.flags.host)+16*slot;
    auto begin=std::chrono::steady_clock::now();unsigned spins=0;
    while(__atomic_load_n(flag,__ATOMIC_ACQUIRE)!=p.epochs[slot]){
        if((++spins&1023)==0 && std::chrono::steady_clock::now()-begin>std::chrono::seconds(10))return fail("Mapped epoch timeout");
#ifdef __aarch64__
        asm volatile("yield" ::: "memory");
#endif
    }return 0;
}
// Nonblocking, system-acquire probe for the ready-driven predictor scheduler.
extern "C" const uint32_t* lm_ready_flag(void* ptr,int slot){
    if(!ptr||slot<0||slot>1)return nullptr;
    auto& p=*static_cast<Payloads*>(ptr);
    return static_cast<const uint32_t*>(p.flags.host)+16*slot;
}
extern "C" void lm_notify_doorbell(void* ptr,uint32_t* device){
    if(ptr)static_cast<Payloads*>(ptr)->notify_device=device;
}
extern "C" uint32_t lm_expected_epoch(void* ptr,int slot){
    if(!ptr||slot<0||slot>1)return 0;
    return static_cast<Payloads*>(ptr)->epochs[slot];
}
int lm_hidden(void* ptr,const void* input,int slot,void* stream){
    auto& p=*static_cast<Payloads*>(ptr);if(slot<0||slot>1)return fail("Invalid hidden slot");
    if(lm_wait(ptr,slot)||p.epochs[slot]==UINT32_MAX)return fail("Hidden epoch ownership");
    hidden_output<<<1,256,0,reinterpret_cast<cudaStream_t>(stream)>>>(static_cast<const __half*>(input),
        static_cast<__half*>(p.hidden[slot].device),static_cast<uint32_t*>(p.flags.device)+16*slot,++p.epochs[slot],p.hidden_size,p.notify_device);
    return checked(cudaGetLastError());
}
int lm_importance(void* ptr,const void* h,void* stream){
    auto& p=*static_cast<Payloads*>(ptr);if(lm_wait(ptr,2)||p.epochs[2]==UINT32_MAX)return fail("Importance epoch ownership");
    importance_output<<<1,256,0,reinterpret_cast<cudaStream_t>(stream)>>>(static_cast<const __half*>(h),
        static_cast<float*>(p.importance.device),static_cast<uint32_t*>(p.flags.device)+32,++p.epochs[2],p.group_size,p.active);
    return checked(cudaGetLastError());
}
// Bank reuse has already been proven by its maintenance epoch. Fail rather
// than hiding an unexpected main-thread producer wait in the async runtime.
int lm_hidden_nowait(void* ptr,const void* input,int slot,void* stream){
    auto& p=*static_cast<Payloads*>(ptr);if(slot<0||slot>1)return fail("Invalid hidden slot");
    auto* flag=static_cast<uint32_t*>(p.flags.host)+16*slot;
    if(__atomic_load_n(flag,__ATOMIC_ACQUIRE)!=p.epochs[slot]||p.epochs[slot]==UINT32_MAX)return fail("Outstanding hidden bank");
    hidden_output<<<1,256,0,reinterpret_cast<cudaStream_t>(stream)>>>(static_cast<const __half*>(input),
        static_cast<__half*>(p.hidden[slot].device),static_cast<uint32_t*>(p.flags.device)+16*slot,++p.epochs[slot],p.hidden_size,p.notify_device);
    return checked(cudaGetLastError());
}
int lm_importance_nowait(void* ptr,const void* h,void* stream){
    auto& p=*static_cast<Payloads*>(ptr);auto* flag=static_cast<uint32_t*>(p.flags.host)+32;
    if(__atomic_load_n(flag,__ATOMIC_ACQUIRE)!=p.epochs[2]||p.epochs[2]==UINT32_MAX)return fail("Outstanding importance bank");
    importance_output<<<1,256,0,reinterpret_cast<cudaStream_t>(stream)>>>(static_cast<const __half*>(h),
        static_cast<float*>(p.importance.device),static_cast<uint32_t*>(p.flags.device)+32,++p.epochs[2],p.group_size,p.active);
    return checked(cudaGetLastError());
}
int lm_importance_no_waste_nowait(void* ptr,const void* h,const int32_t* exec_count,void* stream){
    auto& p=*static_cast<Payloads*>(ptr);auto* flag=static_cast<uint32_t*>(p.flags.host)+32;
    if(p.max_exec<=p.active||!exec_count||
       __atomic_load_n(flag,__ATOMIC_ACQUIRE)!=p.epochs[2]||p.epochs[2]==UINT32_MAX)
        return fail("Outstanding/invalid no-waste importance bank");
    importance_output_variable<<<1,256,0,reinterpret_cast<cudaStream_t>(stream)>>>(
        static_cast<const __half*>(h),static_cast<float*>(p.importance.device),
        static_cast<uint32_t*>(p.flags.device)+32,++p.epochs[2],exec_count,p.group_size,p.max_exec);
    return checked(cudaGetLastError());
}
int lm_stream_synchronize(void* stream){return checked(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)));}
int lm_keep(void* ptr,void* cache){if(lm_wait(ptr,2))return -1;
    return lc_cache_update_keep(cache,static_cast<const float*>(static_cast<Payloads*>(ptr)->importance.host));}
int lm_dequant(void* ptr,int n,const void* scales,void* cache,void* stream){
    auto& p=*static_cast<Payloads*>(ptr);
    if(n<0||n>p.active)return fail("Invalid dequant count");if(!n)return 0;
    mapped_dequant<<<n,256,0,reinterpret_cast<cudaStream_t>(stream)>>>(static_cast<const uint8_t*>(p.packed.device),
        static_cast<const int32_t*>(p.metadata.device),n,static_cast<const __half*>(scales),static_cast<__half*>(cache),p.hidden_size);
    return checked(cudaGetLastError());
}
// Mathematical validation on discrete GPUs: no host atomic claims, same exact kernels.
int lm_test_hidden(const void* input,void* output,void* flag,void* stream){
    hidden_output<<<1,256,0,reinterpret_cast<cudaStream_t>(stream)>>>(static_cast<const __half*>(input),static_cast<__half*>(output),static_cast<uint32_t*>(flag),1,4096,nullptr);
    return checked(cudaGetLastError());
}
int lm_test_importance(const void* input,void* output,void* flag,void* stream){
    importance_output<<<1,256,0,reinterpret_cast<cudaStream_t>(stream)>>>(static_cast<const __half*>(input),static_cast<float*>(output),static_cast<uint32_t*>(flag),1);
    return checked(cudaGetLastError());
}
}
struct DLDevice{int type,id;};struct DLType{uint8_t code,bits;uint16_t lanes;};
struct DLTensor{void* data;DLDevice device;int ndim;DLType dtype;int64_t* shape;int64_t* strides;uint64_t offset;};
struct DLManaged{DLTensor tensor;void* ctx;void(*deleter)(DLManaged*);};
struct View{DLManaged tensor;int64_t shape[3]={1,1,4096};};
extern "C" void* lm_hidden_dlpack(void* ptr,int slot){
    if(slot<0||slot>1)return nullptr;int d=0;cudaGetDevice(&d);auto* v=new View;
    v->shape[2]=static_cast<Payloads*>(ptr)->hidden_size;
    v->tensor.tensor={lm_pointer(ptr,slot,1),{2,d},3,{2,16,1},v->shape,nullptr,0};
    v->tensor.ctx=v;v->tensor.deleter=[](DLManaged* t){delete static_cast<View*>(t->ctx);};return &v->tensor;
}

// Packed cache maintenance only; no quantization/dequantization or host/device memcpy.
__global__ void packed_update(const uint8_t* source,const int32_t* meta,int n,uint8_t* cache,int bytes){
    int k=blockIdx.x;if(k>=n)return;
    int src=meta[k*3],dst=meta[k*3+2];
    if(dst<0)return; // Async fixed-capacity launch; unused descriptors are sentinel-marked by P2.
    for(int j=threadIdx.x;j<bytes;j+=blockDim.x)cache[dst*bytes+j]=source[src*bytes+j];
}
extern "C" int lm_packed_update(void* ptr,int n,void* cache,void* stream){
    auto& p=*static_cast<Payloads*>(ptr);
    if(n<0||n>p.active)return fail("Invalid packed update count");if(!n)return 0;
    packed_update<<<n,256,0,reinterpret_cast<cudaStream_t>(stream)>>>(static_cast<const uint8_t*>(p.packed.device),
        static_cast<const int32_t*>(p.metadata.device),n,static_cast<uint8_t*>(cache),p.bytes);
    return checked(cudaGetLastError());
}
struct FlatView{DLManaged tensor;int64_t shape;};
extern "C" void* lm_packed_dlpack(void* ptr){
    int d=0;cudaGetDevice(&d);auto* v=new FlatView;auto& p=*static_cast<Payloads*>(ptr);
    v->shape=(p.groups==7168&&packed_slots==256?256:p.groups)*p.bytes;
    v->tensor.tensor={lm_pointer(ptr,3,1),{2,d},1,{1,8,1},&v->shape,nullptr,0};
    v->tensor.ctx=v;v->tensor.deleter=[](DLManaged* t){delete static_cast<FlatView*>(t->ctx);};return &v->tensor;
}

// Speculative midpoint only: exact half residual addition, then existing RMS
// rounding semantics, into otherwise-unused mapped hidden bank 1.
__global__ void mid_norm_output(const __half* residual,const float* partial,const __half* weight,float eps,
                               __half* out,uint32_t* flag,uint32_t epoch){
    __shared__ float sums[256];float sum=0.f;
    for(int i=threadIdx.x;i<4096;i+=256){float x=__half2float(__float2half_rn(__half2float(residual[i])+partial[i]));sum+=x*x;}
    sums[threadIdx.x]=sum;__syncthreads();
    for(int width=128;width;width>>=1){if(threadIdx.x<width)sums[threadIdx.x]+=sums[threadIdx.x+width];__syncthreads();}
    float inverse=rsqrtf(sums[0]/4096.f+eps);
    for(int i=threadIdx.x;i<4096;i+=256){float x=__half2float(__float2half_rn(__half2float(residual[i])+partial[i]));
        out[i]=__hmul(weight[i],__float2half_rn(x*inverse));}
    publish(flag,epoch);
}
extern "C" int lm_mid_norm(void* ptr,const void* residual,const void* partial,const void* weight,float eps,void* stream){
    auto& p=*static_cast<Payloads*>(ptr);auto* flag=static_cast<uint32_t*>(p.flags.host)+16;
    if(__atomic_load_n(flag,__ATOMIC_ACQUIRE)!=p.epochs[1]||p.epochs[1]==UINT32_MAX)return fail("Outstanding midpoint bank");
    mid_norm_output<<<1,256,0,reinterpret_cast<cudaStream_t>(stream)>>>(static_cast<const __half*>(residual),static_cast<const float*>(partial),
        static_cast<const __half*>(weight),eps,static_cast<__half*>(p.hidden[1].device),static_cast<uint32_t*>(p.flags.device)+16,++p.epochs[1]);
    return checked(cudaGetLastError());
}

// Reserve the next producer-side hidden epoch; previous producer must be complete.
// Called on the model thread before enqueue, never waits or changes GPU-only data.
extern "C" uint32_t lm_hidden_reserve(void* ptr){
    auto& p=*static_cast<Payloads*>(ptr);auto* flag=static_cast<uint32_t*>(p.flags.host);
    if(__atomic_load_n(flag,__ATOMIC_ACQUIRE)!=p.epochs[0]||p.epochs[0]==UINT32_MAX){fail("Outstanding producer hidden bank");return 0;}
    __atomic_store_n(flag+1,0u,__ATOMIC_RELEASE);
    return ++p.epochs[0];
}
extern "C" void* lm_hidden_flag_dlpack(void* ptr){
    auto& p=*static_cast<Payloads*>(ptr);int d=0;cudaGetDevice(&d);auto* v=new FlatView;v->shape=2;
    v->tensor.tensor={p.flags.device,{2,d},1,{0,32,1},&v->shape,nullptr,0};
    v->tensor.ctx=v;v->tensor.deleter=[](DLManaged* t){delete static_cast<FlatView*>(t->ctx);};return &v->tensor;
}

// Opt-in direct-GQA checkpoint. Bank 1 and its adjacent arrival counter are
// unused by the normal P1/P2 path. Reserve only after the previous epoch has
// completed; the GPU stores all 32 head payloads and the last CTA publishes.
extern "C" uint32_t lm_mid_checkpoint_reserve(void* ptr){
    auto& p=*static_cast<Payloads*>(ptr);auto* flag=static_cast<uint32_t*>(p.flags.host)+16;
    if(p.epochs[1]==UINT32_MAX||__atomic_load_n(flag,__ATOMIC_ACQUIRE)!=p.epochs[1]){
        fail("Outstanding midpoint checkpoint bank");return 0;
    }
    __atomic_store_n(flag+1,0u,__ATOMIC_RELEASE);
    return ++p.epochs[1];
}
extern "C" void* lm_mid_checkpoint_flags_dlpack(void* ptr){
    auto& p=*static_cast<Payloads*>(ptr);int d=0;cudaGetDevice(&d);auto* v=new FlatView;v->shape=2;
    v->tensor.tensor={static_cast<uint32_t*>(p.flags.device)+16,{2,d},1,{0,32,1},&v->shape,nullptr,0};
    v->tensor.ctx=v;v->tensor.deleter=[](DLManaged* t){delete static_cast<FlatView*>(t->ctx);};return &v->tensor;
}
extern "C" void* lm_notify_doorbell_dlpack(void* ptr){
    auto& p=*static_cast<Payloads*>(ptr);if(!p.notify_device)return nullptr;
    int d=0;cudaGetDevice(&d);auto* v=new FlatView;v->shape=1;
    v->tensor.tensor={p.notify_device,{2,d},1,{0,32,1},&v->shape,nullptr,0};
    v->tensor.ctx=v;v->tensor.deleter=[](DLManaged* t){delete static_cast<FlatView*>(t->ctx);};return &v->tensor;
}
