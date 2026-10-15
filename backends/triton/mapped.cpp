// Host-only mapped allocation / release-acquire helper. GPU computation uses Triton.
#include <cuda_runtime_api.h>
#include <cstdint>
#include <cstring>
#include <chrono>
struct Buffer {void* host=nullptr;void* device=nullptr;size_t bytes=0;};
extern "C" {
void tm_free(void* p){if(!p)return;auto* b=static_cast<Buffer*>(p);if(b->host)cudaFreeHost(b->host);delete b;}
void* tm_alloc(uint64_t bytes){
    int d=0,map=0,integrated=0,atomic=0;cudaGetDevice(&d);
    cudaDeviceGetAttribute(&map,cudaDevAttrCanMapHostMemory,d);cudaDeviceGetAttribute(&integrated,cudaDevAttrIntegrated,d);
    cudaDeviceGetAttribute(&atomic,cudaDevAttrHostNativeAtomicSupported,d);
    if(!bytes||!map||!integrated||!atomic)return nullptr;
    auto* b=new Buffer;b->bytes=bytes;
    if(cudaHostAlloc(&b->host,bytes,cudaHostAllocMapped)!=cudaSuccess || cudaHostGetDevicePointer(&b->device,b->host,0)!=cudaSuccess){tm_free(b);return nullptr;}
    std::memset(b->host,0,bytes);return b;
}
void* tm_host(void* p){return static_cast<Buffer*>(p)->host;}
void* tm_device(void* p){return static_cast<Buffer*>(p)->device;}
void tm_release(void* p,int index,uint32_t epoch){__atomic_store_n(static_cast<uint32_t*>(static_cast<Buffer*>(p)->host)+index,epoch,__ATOMIC_RELEASE);}
int tm_acquire_all(void* p,int count,uint32_t epoch,int timeout_ms){
    auto* b=static_cast<Buffer*>(p);if(count<0||size_t(count)*4>b->bytes)return -1;
    auto start=std::chrono::steady_clock::now();unsigned spins=0;
    for(int i=0;i<count;i++)while(__atomic_load_n(static_cast<uint32_t*>(b->host)+i,__ATOMIC_ACQUIRE)!=epoch){
        if((++spins&1023)==0 && std::chrono::steady_clock::now()-start>std::chrono::milliseconds(timeout_ms))return -2;
#if defined(__aarch64__)
        asm volatile("yield" ::: "memory");
#endif
    }return 0;
}
}
struct Device{int type,id;};struct Type{uint8_t code,bits;uint16_t lanes;};
struct Tensor{void* data;Device device;int ndim;Type dtype;int64_t* shape;int64_t* strides;uint64_t offset;};
struct Managed{Tensor tensor;void* ctx;void(*deleter)(Managed*);};
struct View{Managed tensor;int64_t shape;};
extern "C" void* tm_dlpack(void* p,int code,int bits,int64_t count){
    auto* b=static_cast<Buffer*>(p);if(count<=0||uint64_t(count)*bits/8>b->bytes)return nullptr;
    int dev=0;cudaGetDevice(&dev);auto* v=new View;v->shape=count;
    v->tensor.tensor={b->device,{2,dev},1,{uint8_t(code),uint8_t(bits),1},&v->shape,nullptr,0};
    v->tensor.ctx=v;v->tensor.deleter=[](Managed* m){delete static_cast<View*>(m->ctx);};return &v->tensor;
}
