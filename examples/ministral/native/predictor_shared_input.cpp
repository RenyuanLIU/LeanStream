// Single persistent slot: one outstanding producer; CPU finishes before reuse.
#include <cuda_runtime_api.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <chrono>
#include <dlfcn.h>
#include <new>
#include <limits>
#include <algorithm>
#ifdef __aarch64__
#include <arm_neon.h>
#endif
extern "C" cudaError_t si_signal(uint32_t*,uint32_t,cudaStream_t);
extern "C" cudaError_t si_norm(const void*,const void*,float,void*,uint32_t*,uint32_t,cudaStream_t,int,uint32_t*);
using Create=void*(*)(int,int,int,int,int);
using Destroy=void(*)(void*);
using Predict=int(*)(void*,const void*,const float*,const void*,float,const int32_t*,int,int,int32_t*,float*,float*,float*);
using PredictSubset=int(*)(void*,const void*,const float*,const void*,float,const int32_t*,int,int,int32_t*);
struct alignas(64) Shared {
    uint16_t *host=nullptr,*device=nullptr;
    uint32_t *flag=nullptr,*device_flag=nullptr;
    uint32_t* notify_device=nullptr;
    uint32_t epoch=0;
    cudaEvent_t profile_start=nullptr,profile_end=nullptr;
    alignas(64) int8_t quant[1664]{};
    float scale=1;double last_ready_ms=0,last_predict_start=0,last_predict_end=0;
    void *library=nullptr,*predictor=nullptr,*ranked_predictor=nullptr;
    int rows=7168, keep_size=1219, fill_size=215,hidden_size=4096,stride=1664,active_dim=1638;
    const float* projection=nullptr;bool capture_scores=false;
    alignas(64) float scores[11520]{},projected[1664]{};int32_t exclusions[1959]{};
    Predict predict=nullptr;Destroy destroy=nullptr;
    PredictSubset predict_subset=nullptr;

};
static thread_local char error_text[192]{};
static int fail(const char* text,int code=-1) {std::snprintf(error_text,sizeof(error_text),"%s",text);return code;}
static double now_ms() {
    return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
static float half_value(uint16_t value) {
#ifdef __aarch64__
    __fp16 half;std::memcpy(&half,&value,2);return float(half);
#else
    int e=(value>>10)&31,m=value&1023;
    float v=e==0?std::ldexp(float(m),-24):e==31?(m?NAN:INFINITY):std::ldexp(float(1024+m),e-25);
    return value&32768?-v:v;
#endif
}
extern "C" {
const char* si_error(){return error_text;}
void si_destroy(void* pointer) {
    auto* s=static_cast<Shared*>(pointer);if(!s)return;
    if(s->predictor)s->destroy(s->predictor);
    if(s->ranked_predictor)s->destroy(s->ranked_predictor);
    if(s->library)dlclose(s->library);
    if(s->host)cudaFreeHost(s->host);
    if(s->flag)cudaFreeHost(s->flag);
    delete s;
}
void* si_create(const char* library) {
    int dev=0,canmap=0,integrated=0,major=0,minor=0,atomic=0;
    cudaError_t rc=cudaGetDevice(&dev);
    if(rc!=cudaSuccess){fail(cudaGetErrorString(rc));return nullptr;}
    cudaDeviceGetAttribute(&canmap,cudaDevAttrCanMapHostMemory,dev);
    cudaDeviceGetAttribute(&integrated,cudaDevAttrIntegrated,dev);
    cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,dev);
    cudaDeviceGetAttribute(&minor,cudaDevAttrComputeCapabilityMinor,dev);
    cudaDeviceGetAttribute(&atomic,cudaDevAttrHostNativeAtomicSupported,dev);
    if(!canmap || !integrated || major*10+minor<72 || !atomic) {
        fail("Requires integrated Tegra >=7.2, mapped memory, host-native system atomics");return nullptr;
    }
    auto* s=new Shared;
    rc=cudaHostAlloc(reinterpret_cast<void**>(&s->host),8192,cudaHostAllocMapped);
    if(rc==cudaSuccess)rc=cudaHostGetDevicePointer(reinterpret_cast<void**>(&s->device),s->host,0);
    if(rc==cudaSuccess)rc=cudaHostAlloc(reinterpret_cast<void**>(&s->flag),64,cudaHostAllocMapped);
    if(rc==cudaSuccess)rc=cudaHostGetDevicePointer(reinterpret_cast<void**>(&s->device_flag),s->flag,0);
    if(rc!=cudaSuccess){fail(cudaGetErrorString(rc));si_destroy(s);return nullptr;}
    std::memset(s->host,0,8192);__atomic_store_n(s->flag,0u,__ATOMIC_RELEASE);
    if(!__atomic_is_lock_free(sizeof(uint32_t),s->flag)){fail("Host epoch is not lock-free");si_destroy(s);return nullptr;}
    s->library=dlopen(library,RTLD_NOW|RTLD_LOCAL);
    if(!s->library){fail(dlerror());si_destroy(s);return nullptr;}
    auto create=reinterpret_cast<Create>(dlsym(s->library,"ls_create"));
    s->destroy=reinterpret_cast<Destroy>(dlsym(s->library,"ls_destroy"));
    s->predict=reinterpret_cast<Predict>(dlsym(s->library,"ls_predict"));
    s->predict_subset=reinterpret_cast<PredictSubset>(dlsym(s->library,"ls_predict_subset"));
    if(!create || !s->destroy || !s->predict){fail("Missing W4A8 ABI");si_destroy(s);return nullptr;}
    s->predictor=create(7168,1664,215,12,4);
    if(!s->predictor){fail("12-core W4A8 context failed");si_destroy(s);return nullptr;}
    return s;
}
int si_configure(void* ptr,int g,int capture){
    auto& s=*static_cast<Shared*>(ptr);if(g<2||g>4||s.epoch||s.ranked_predictor)return fail("Invalid predictor configuration");
    s.rows=g==2?7168:g==3?4779:3584;s.keep_size=g==2?1219:g==3?812:609;s.fill_size=2868/g-s.keep_size;
    auto create=reinterpret_cast<Create>(dlsym(s.library,"ls_create"));
    void* next=create(s.rows,1664,s.fill_size,12,4);if(!next)return fail("Group-native context failed");
    s.destroy(s.predictor);s.predictor=next;s.capture_scores=capture;return 0;
}
int si_configure_model(void* ptr,int h,int rows,int keep,int fill,int stride){
    auto& s=*static_cast<Shared*>(ptr);
    if(s.epoch||s.ranked_predictor||h<128||h>4096||h%128||rows<1||rows>11520||
       keep<1||keep>1958||fill<1||fill>346||stride<1||stride>1664||stride%32)
        return fail("Invalid projected predictor geometry");
    auto create=reinterpret_cast<Create>(dlsym(s.library,"ls_create"));
    void* next=create(rows,stride,fill,12,4);
    if(!next)return fail("Model W4A8 context failed");
    s.destroy(s.predictor);s.predictor=next;s.rows=rows;s.keep_size=keep;s.fill_size=fill;
    s.hidden_size=h;s.stride=s.active_dim=stride;s.projection=nullptr;return 0;
}
int si_set_projection(void* ptr,const float* projection,int stride){
    auto& s=*static_cast<Shared*>(ptr);
    if(!projection||stride!=s.active_dim)return fail("Invalid per-layer projection");
    s.projection=projection;return 0;
}
const float* si_scores(void* ptr){return static_cast<Shared*>(ptr)->scores;}
void si_capture_scores(void* ptr,int enabled){static_cast<Shared*>(ptr)->capture_scores=enabled!=0;}
static const int32_t* exclusions(Shared& s,const int32_t* keep){
    if(s.rows!=4779)return keep;
    std::copy_n(keep,s.keep_size,s.exclusions);s.exclusions[s.keep_size]=4778;return s.exclusions;
}
// Initialize the existing OpenMP predictor team on its persistent control master.
// No new arithmetic/context representation; never called in timed decode.
int si_worker_init(void* pointer) {
    auto& s=*static_cast<Shared*>(pointer);
    auto create=reinterpret_cast<Create>(dlsym(s.library,"ls_create"));
    void* temporary=create(s.rows,s.stride,s.fill_size,12,4);
    if(!temporary)return fail("Control-worker predictor initialization failed");
    s.destroy(temporary);return 0;
}
int si_enable_ranking(void* pointer){
    auto& s=*static_cast<Shared*>(pointer);
    auto create=reinterpret_cast<Create>(dlsym(s.library,"ls_create"));
    if(!s.ranked_predictor)s.ranked_predictor=create(s.rows,s.stride,512,12,4);
    return s.ranked_predictor?0:fail("Continuation predictor context failed");
}
void* si_host(void* p){return static_cast<Shared*>(p)->host;}
void* si_device(void* p){return static_cast<Shared*>(p)->device;}
void* si_quant(void* p){return static_cast<Shared*>(p)->quant;}
double si_ready_ms(void* p){return static_cast<Shared*>(p)->last_ready_ms;}
float si_scale(void* p){return static_cast<Shared*>(p)->scale;}
uint32_t si_epoch(void* p){return __atomic_load_n(static_cast<Shared*>(p)->flag,__ATOMIC_ACQUIRE);}
const uint32_t* si_ready_flag(void* p){return p?static_cast<Shared*>(p)->flag:nullptr;}
void si_notify_doorbell(void* p,uint32_t* device){if(p)static_cast<Shared*>(p)->notify_device=device;}
uint32_t si_expected_epoch(void* p){return p?static_cast<Shared*>(p)->epoch:0;}
// No half gather buffer: directly read selected elements of the full vector twice.
int si_prepare_array(const uint16_t* input,const int32_t* features,int8_t* quant,float* scale) {
    float maximum=0;
    for(int i=0;i<1638;++i) {
        if(features[i]<0 || features[i]>=4096)return fail("Feature index out of range");
        float x=half_value(input[features[i]]);
        if(!std::isfinite(x))return fail("Nonfinite MLP input");
        maximum=std::fmax(maximum,std::fabs(x));
    }
    *scale=maximum==0?1.f:maximum/127.f;
    for(int i=0;i<1638;++i) {
        float q=std::nearbyint(half_value(input[features[i]])/(*scale));
        quant[i]=static_cast<int8_t>(std::fmax(-127.f,std::fmin(127.f,q)));
    }
    return 0;
}
int si_prepare(void* pointer,const uint16_t* input,const int32_t* features) {
    auto& s=*static_cast<Shared*>(pointer);
    if(s.active_dim!=1664 && !s.projection){
        if(!features||s.active_dim<1||s.active_dim>s.stride)return fail("Invalid compact feature configuration");
        float maximum=0;
        for(int i=0;i<s.active_dim;i++){
            if(features[i]<0||features[i]>=s.hidden_size)return fail("Compact feature index out of range");
            float x=half_value(input[features[i]]);
            if(!std::isfinite(x))return fail("Nonfinite compact predictor input");
            maximum=std::fmax(maximum,std::fabs(x));
        }
        s.scale=maximum==0?1.f:maximum/127.f;
        std::fill_n(s.quant,s.stride,0);
        for(int i=0;i<s.active_dim;i++){
            float q=std::nearbyint(half_value(input[features[i]])/s.scale);
            s.quant[i]=static_cast<int8_t>(std::fmax(-127.f,std::fmin(127.f,q)));
        }
        return 0;
    }
    if(!s.projection)return si_prepare_array(input,features,s.quant,&s.scale);
    std::fill_n(s.projected,s.active_dim,0.f);
    for(int i=0;i<s.hidden_size;i++){
        float x=half_value(input[i]);if(!std::isfinite(x))return fail("Nonfinite projected input");
        const float* row=s.projection+size_t(i)*s.active_dim;
        for(int d=0;d<s.active_dim;d++)s.projected[d]+=x*row[d];
    }
    float maximum=0.f;
    for(int d=0;d<s.active_dim;d++)maximum=std::fmax(maximum,std::fabs(s.projected[d]));
    s.scale=maximum==0.f?1.f:maximum/127.f;
    for(int d=0;d<s.active_dim;d++){
        float q=std::nearbyint(s.projected[d]/s.scale);
        s.quant[d]=static_cast<int8_t>(std::fmax(-127.f,std::fmin(127.f,q)));
    }
    return 0;
}
int si_set_active_features(void* pointer,int count){
    if(!pointer)return fail("Missing compact predictor context");
    auto& s=*static_cast<Shared*>(pointer);
    if(count<1||count>s.stride)return fail("Invalid compact active feature count");
    s.active_dim=count;return 0;
}
// Same validated predictor/quantizer, for already acquire-visible mapped layer input (P1).
extern "C" int si_predict_ready(void* pointer,const uint16_t* input,const int32_t* features,
        const void* weights,const float* scales,const int32_t* keep,int32_t* output,double* times){
    auto& s=*static_cast<Shared*>(pointer);double begin=now_ms();
    if(si_prepare(pointer,input,features))return -1;
    double prepared=now_ms();
    int rc=s.predict(s.predictor,weights,scales,s.quant,s.scale,exclusions(s,keep),s.keep_size+(s.rows==4779),0,output,nullptr,nullptr,s.capture_scores?s.scores:nullptr);
    double end=now_ms();s.last_predict_start=begin;s.last_predict_end=end;times[0]=prepared-begin;times[1]=end-prepared;times[2]=end-begin;
    return rc?fail("W4A8 prediction failed",rc):0;
}
// Same feature preparation and W4A8 arithmetic, restricted to caller-owned
// logical row IDs. Only speculative P1/Mid may call this entry point.
int si_predict_subset(void* pointer,const uint16_t* input,const int32_t* features,
        const void* weights,const float* scales,const int32_t* ids,int count,int take,
        int32_t* output,double* times){
    auto& s=*static_cast<Shared*>(pointer);double begin=now_ms();
    if(!s.predict_subset||!ids||!output||count<take||take<1||count>s.rows)return fail("Invalid subset predictor");
    if(si_prepare(pointer,input,features))return -1;
    double prepared=now_ms();
    int rc=s.predict_subset(s.predictor,weights,scales,s.quant,s.scale,ids,count,take,output);
    double end=now_ms();times[0]=prepared-begin;times[1]=end-prepared;times[2]=end-begin;
    return rc?fail("Subset W4A8 prediction failed",rc):0;
}
int si_predict_ranked(void* pointer,const uint16_t* input,const int32_t* features,
        const void* weights,const float* scales,const int32_t* keep,int32_t* first,double* times,int32_t* ranking){
    auto& s=*static_cast<Shared*>(pointer);double begin=now_ms();
    if(si_prepare(pointer,input,features))return -1;double prepared=now_ms();
    if(!s.ranked_predictor)return fail("Continuation context not initialized");
    // Existing backend API: one top-512 selection, not a second score-copy/sort pass.
    int rc=s.predict(s.ranked_predictor,weights,scales,s.quant,s.scale,exclusions(s,keep),s.keep_size+(s.rows==4779),0,ranking,nullptr,nullptr,nullptr);
    double end=now_ms();s.last_predict_start=begin;s.last_predict_end=end;times[0]=prepared-begin;times[1]=end-prepared;times[2]=end-begin;
    if(rc)return fail("Ranked prediction failed",rc);
    std::copy(ranking,ranking+s.fill_size,first);return 0;
}
// Optional event instrumentation: no change to producer/flag/predictor semantics.
void si_profile_events(void* pointer,void* start,void* end) {
    auto& s=*static_cast<Shared*>(pointer);
    s.profile_start=reinterpret_cast<cudaEvent_t>(start);
    s.profile_end=reinterpret_cast<cudaEvent_t>(end);
}
// Split runtime: enqueue the unchanged Mode-B producer, then let the host enqueue
// downstream kernels before it acquire-waits and runs P2. si_run remains unchanged.
int si_start_norm(void* pointer,const void* input,const void* weight,float eps,void* stream_pointer) {
    auto& s=*static_cast<Shared*>(pointer);
    if(s.epoch==UINT32_MAX || si_epoch(pointer)!=s.epoch)return fail("Epoch ownership violation");
    auto stream=reinterpret_cast<cudaStream_t>(stream_pointer);
    if(s.profile_start && cudaEventRecord(s.profile_start,stream)!=cudaSuccess)return fail("Norm start event");
    auto rc=si_norm(input,weight,eps,s.device,s.device_flag,++s.epoch,stream,s.hidden_size,s.notify_device);
    if(rc!=cudaSuccess)return fail(cudaGetErrorString(rc));
    if(s.profile_end && cudaEventRecord(s.profile_end,stream)!=cudaSuccess)return fail("Norm end event");
    return 0;
}
int si_finish_norm(void* pointer,const int32_t* features,const void* weights,const float* scales,
                   const int32_t* keep,int32_t* output,double* times) {
    auto& s=*static_cast<Shared*>(pointer);double begin=now_ms();
    while(__atomic_load_n(s.flag,__ATOMIC_ACQUIRE)!=s.epoch) {
        if(now_ms()-begin>10000)return fail("Split RMS epoch timeout");
#ifdef __aarch64__
        asm volatile("yield" ::: "memory");
#endif
    }
    s.last_ready_ms=now_ms();times[0]=s.last_ready_ms-begin;
    return si_predict_ready(pointer,s.host,features,weights,scales,keep,output,times+1);
}
int si_finish_norm_ready(void* pointer,const int32_t* features,const void* weights,const float* scales,
                         const int32_t* keep,int32_t* output,double* times){
    auto& s=*static_cast<Shared*>(pointer);
    if(__atomic_load_n(s.flag,__ATOMIC_ACQUIRE)!=s.epoch)return fail("P2 payload not READY");
    s.last_ready_ms=now_ms();times[0]=0;
    return si_predict_ready(pointer,s.host,features,weights,scales,keep,output,times+1);
}

int si_run(void* pointer,int mode,const void* input,const void* norm_weight,float eps,
           void* stream_pointer,const int32_t* features,const void* weights,const float* scales,
           const int32_t* keep,int32_t* output,double* times,float* scores) {
    auto& s=*static_cast<Shared*>(pointer);
    if(mode!=0 && mode!=1)return fail("Invalid handoff mode");
    if(s.epoch==UINT32_MAX || si_epoch(pointer)!=s.epoch)return fail("Epoch ownership violation");
    uint32_t expected=++s.epoch;
    auto stream=reinterpret_cast<cudaStream_t>(stream_pointer);
    double begin=now_ms();
    cudaError_t rc;
    if(s.profile_start) {rc=cudaEventRecord(s.profile_start,stream);if(rc!=cudaSuccess)return fail(cudaGetErrorString(rc));}
    if(mode==0) {
        rc=cudaMemcpyAsync(s.host,input,size_t(s.hidden_size)*2,cudaMemcpyDeviceToHost,stream);
        if(rc==cudaSuccess)rc=si_signal(s.device_flag,expected,stream);
    } else {
        rc=si_norm(input,norm_weight,eps,s.device,s.device_flag,expected,stream,s.hidden_size,s.notify_device);
    }
    if(rc!=cudaSuccess)return fail(cudaGetErrorString(rc));
    if(s.profile_end) {rc=cudaEventRecord(s.profile_end,stream);if(rc!=cudaSuccess)return fail(cudaGetErrorString(rc));}
    unsigned spins=0;
    while(__atomic_load_n(s.flag,__ATOMIC_ACQUIRE)!=expected) {
        if((++spins&1023)==0 && now_ms()-begin>10000)return fail("Epoch timeout (10 seconds)");
#ifdef __aarch64__
        asm volatile("yield" ::: "memory");
#endif
    }
    double ready=now_ms();s.last_ready_ms=ready;
    if(si_prepare(pointer,s.host,features))return -1;
    double prepared=now_ms();
    int status=s.predict(s.predictor,weights,scales,s.quant,s.scale,keep,s.keep_size,0,output,nullptr,nullptr,scores);
    double finish=now_ms();
    times[0]=ready-begin;times[1]=prepared-ready;times[2]=finish-prepared;
    times[3]=finish-ready;times[4]=finish-begin;
    return status?fail("W4A8 prediction failed",status):0;
}
}

struct DLDevice {int device_type;int device_id;};
struct DLDataType {uint8_t code,bits;uint16_t lanes;};
struct DLTensor {void* data;DLDevice device;int ndim;DLDataType dtype;int64_t* shape;int64_t* strides;uint64_t byte_offset;};
struct DLManagedTensor {DLTensor dl_tensor;void* manager_ctx;void(*deleter)(DLManagedTensor*);};
struct MappedView {DLManagedTensor tensor;int64_t shape[3]={1,1,4096};};
extern "C" void* si_dlpack(void* pointer) {
    auto& s=*static_cast<Shared*>(pointer);int device=0;cudaGetDevice(&device);
    auto* view=new MappedView;
    view->shape[2]=s.hidden_size;
    view->tensor.dl_tensor={s.device,{2,device},3,{2,16,1},view->shape,nullptr,0};
    view->tensor.manager_ctx=view;
    view->tensor.deleter=[](DLManagedTensor* t){delete static_cast<MappedView*>(t->manager_ctx);};
    return &view->tensor;
}

extern "C" double si_predict_start_ms(void* p){return static_cast<Shared*>(p)->last_predict_start;}
extern "C" double si_predict_end_ms(void* p){return static_cast<Shared*>(p)->last_predict_end;}

extern "C" uint32_t si_reserve_norm(void* ptr){auto& s=*static_cast<Shared*>(ptr);if(s.epoch==UINT32_MAX||si_epoch(ptr)!=s.epoch){fail("Prepared RMS ownership violation");return 0;}return ++s.epoch;}
extern "C" uint32_t* si_device_ready_flag(void* ptr){return static_cast<Shared*>(ptr)->device_flag;}

extern "C" void* si_flag_dlpack(void* pointer){auto& s=*static_cast<Shared*>(pointer);int device=0;cudaGetDevice(&device);auto* view=new MappedView;view->shape[0]=1;view->tensor.dl_tensor={s.device_flag,{2,device},1,{0,32,1},view->shape,nullptr,0};view->tensor.manager_ctx=view;view->tensor.deleter=[](DLManagedTensor* t){delete static_cast<MappedView*>(t->manager_ctx);};return &view->tensor;}
