// Optional Python lifecycle bridge. Scheduling and publication belong to Runtime.
#include "leanstream/runtime.h"
#include "leanstream/triton_cuda.h"
#include <cuda_runtime.h>
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>
#include <atomic>
#include <stdexcept>
#ifdef LEANSTREAM_TORCH_TESTING
#include <condition_variable>
#include <mutex>
static std::mutex test_mutex;static std::condition_variable test_changed;
static bool test_hold=false;static std::atomic<uint64_t> test_entered{0},test_bytes{0};
extern "C" ssize_t stream_test_pread(int fd,void* out,size_t bytes,off_t offset){
    {std::unique_lock<std::mutex> lock(test_mutex);++test_entered;test_changed.notify_all();test_changed.wait(lock,[]{return !test_hold;});}
    ssize_t result=::pread(fd,out,bytes,offset);if(result>0)test_bytes+=result;return result;
}
#endif
using namespace leanstream::runtime;
extern "C" {void* tm_alloc(uint64_t);void tm_free(void*);void* tm_host(void*);void* tm_device(void*);}
struct Bridge {
    int fd=-1; uint32_t groups; uint64_t bytes; bool active=false,submitted=false,event_recorded=false,completion_proved=false;
    std::vector<GroupLayout> layout;std::vector<SlotId> slots;std::vector<GroupId> resident,core;
    void* buffers[7]{}; // staging, states, descriptors, indices, control, core, counts
    TritonBackend gpu;std::unique_ptr<Runtime> runtime;LayerStep step;
    cudaEvent_t done=nullptr;cudaStream_t stream=nullptr;std::thread retirement;
    bool test_fail_after_submit=false;std::atomic<int> error{0};uint64_t last_requests=0,last_generation=0;
    ~Bridge(){if(retirement.joinable())retirement.join();runtime.reset();for(auto p:buffers)tm_free(p);if(done)cudaEventDestroy(done);if(fd>=0)::close(fd);}
    void join(){if(retirement.joinable())retirement.join();}
    static int enqueue(void* p,uint32_t layer,const void*,void*,void* stream,Span<ExecDesc>,DeviceExecQueue,uint32_t epoch,ConsumerMode){
        auto& b=*static_cast<Bridge*>(p);b.submitted=true;b.stream=static_cast<cudaStream_t>(stream);
        int rc=b.gpu.enqueue({layer,0},epoch,b.stream);
        // Even partial launch failure keeps stream and allocations live for abort.
        #ifdef LEANSTREAM_TORCH_TESTING
        if(!rc&&b.test_fail_after_submit){b.test_fail_after_submit=false;return EIO;}
#endif
        if(!rc){rc=int(cudaEventRecord(b.done,b.stream));b.event_recorded=!rc;}return rc;
    }
    static bool complete(void* p,uint32_t){auto& b=*static_cast<Bridge*>(p);return b.completion_proved||(b.event_recorded&&cudaEventQuery(b.done)==cudaSuccess);}
    static int defer(void* p,uint32_t,Span<GroupId> ids,uint32_t epoch,void (*retired)(void*,uint32_t),void* ctx){
        auto& b=*static_cast<Bridge*>(p);
        if(ids.size!=b.resident.size())return EINVAL;
        for(size_t i=0;i<ids.size;i++)if(ids[i]!=b.resident[i])return EINVAL;
        try {b.retirement=std::thread([&b,epoch,retired,ctx]{
            int rc=int(cudaEventSynchronize(b.done));
            if(rc){b.error=rc;return;} // Do not retire borrowed buffers without proof.
            retired(ctx,epoch);
        });}catch(...){return ENOMEM;}return 0;
    }
};
extern "C" {
void* lt_create(const char* path,uint32_t groups,uint32_t bytes,const GroupLayout* layout,const int32_t* resident,uint32_t nr,const int32_t* core,uint32_t nc,uint32_t qd){
    try {
        if(!path||!groups||!bytes||bytes%512||!layout||!nr||!nc||nc>nr||nr>groups||!qd||qd>24)return nullptr;
        auto b=std::make_unique<Bridge>();b->groups=groups;b->bytes=bytes;
        b->fd=open(path,O_RDONLY|O_DIRECT);if(b->fd<0)return nullptr;
        b->layout.assign(layout,layout+groups);b->resident.assign(resident,resident+nr);b->core.assign(core,core+nc);b->slots.assign(groups,-1);
        for(uint32_t i=0;i<nr;i++){int id=resident[i];if(id<0||uint32_t(id)>=groups||b->slots[id]>=0)return nullptr;b->slots[id]=i;}
        for(auto id:b->core)if(id<0||uint32_t(id)>=groups||b->slots[id]<0)return nullptr;
        uint64_t sizes[]={uint64_t(groups)*bytes,uint64_t(groups+1)*4,uint64_t(groups)*20,uint64_t(groups)*4,128,uint64_t(nc)*20,uint64_t(groups)*4};
        for(int i=0;i<7;i++){b->buffers[i]=tm_alloc(sizes[i]);if(!b->buffers[i])return nullptr;}
        if(cudaEventCreateWithFlags(&b->done,cudaEventDisableTiming)!=cudaSuccess)return nullptr;
        ComputeBackend backend{};backend.context=b.get();backend.enqueue=Bridge::enqueue;backend.complete=Bridge::complete;backend.defer_commit=Bridge::defer;
        b->runtime=std::make_unique<Runtime>(ManagerConfig{groups,qd,bytes,bytes,512,ConsumerMode::Progressive},backend);
        return b.release();
    } catch(...){return nullptr;}
}
#ifdef LEANSTREAM_TORCH_TESTING
void lt_test_hold(int enabled){std::lock_guard<std::mutex> lock(test_mutex);test_hold=enabled;if(enabled){test_entered=0;test_bytes=0;}test_changed.notify_all();}
uint64_t lt_test_entered(){return test_entered;}
uint64_t lt_test_bytes(){return test_bytes;}
int lt_test_inflight(void* p,int g){return static_cast<Bridge*>(p)->runtime->weights().group_state(g).io_state==IOState::Inflight;}
void lt_test_fail_after_submit(void* p){static_cast<Bridge*>(p)->test_fail_after_submit=true;}
#endif
void* lt_buffer(void* p,int index){return p&&index>=0&&index<7?static_cast<Bridge*>(p)->buffers[index]:nullptr;}
int lt_bind(void* p,uint32_t candidate,uint32_t stage,void* function,uint32_t grid,uint32_t warps,uint32_t shared,const uint64_t* args,int count,int epoch,int claim){
    if(!p)return EINVAL;auto& b=*static_cast<Bridge*>(p);if(b.active)return EBUSY;
    return b.gpu.bind(candidate,stage,static_cast<CUfunction>(function),grid,warps,shared,args,count,epoch,claim);
}
int lt_seed(void* p,uint32_t claim){if(!p)return EINVAL;static_cast<Bridge*>(p)->gpu.seed_claim(claim);return 0;}
int lt_begin(void* p){
    if(!p)return EINVAL;auto& b=*static_cast<Bridge*>(p);b.join();if(b.error)return b.error;if(b.active)return EBUSY;
    LayerStorage s{};s.fd=b.fd;s.layout={b.layout.data(),b.layout.size()};s.cache_slots=b.slots.data();s.staging=tm_host(b.buffers[0]);s.io_states=static_cast<uint32_t*>(tm_host(b.buffers[1]));
    s.core={b.core.data(),b.core.size()};s.queue={static_cast<ExecDesc*>(tm_host(b.buffers[2])),static_cast<uint32_t*>(tm_host(b.buffers[3])),static_cast<QueueControl*>(tm_host(b.buffers[4])),b.groups};
    s.mapped_core_descriptors=static_cast<ExecDesc*>(tm_host(b.buffers[5]));s.staging_bytes=b.groups*b.bytes;
    int rc=b.runtime->begin_layer(0,s,&b.step);if(!rc){b.active=true;b.submitted=false;b.event_recorded=false;b.completion_proved=false;std::memset(tm_host(b.buffers[6]),0,b.groups*4);}return rc;
}
int lt_update(void* p,const int32_t* ids,uint32_t n,uint32_t budget){if(!p)return EINVAL;return static_cast<Bridge*>(p)->step.update_prediction({ids,n},{budget,0,0});}
int lt_enqueue(void* p,void* x,void* y,void* stream){if(!p)return EINVAL;return static_cast<Bridge*>(p)->step.enqueue_compute(x,y,stream);}
int lt_finalize(void* p,const int32_t* ids,uint32_t n){if(!p)return EINVAL;return static_cast<Bridge*>(p)->step.finalize({ids,n});}
int lt_wait_io(void* p,int ms){if(!p)return EINVAL;return static_cast<Bridge*>(p)->runtime->weights().wait_io(ms);}
uint64_t lt_requests(void* p){auto& b=*static_cast<Bridge*>(p);return b.active?b.runtime->weights().read_requests():b.last_requests;}
uint64_t lt_generation(void* p){auto& b=*static_cast<Bridge*>(p);return b.active?b.runtime->weights().prediction_generation():b.last_generation;}
int lt_commit(void* p){
    if(!p)return EINVAL;auto& b=*static_cast<Bridge*>(p);b.last_requests=b.runtime->weights().read_requests();b.last_generation=b.runtime->weights().prediction_generation();
    int rc=b.step.commit({b.resident.data(),b.resident.size()});if(!rc)b.active=false;return rc;
}
int lt_drain(void* p){if(!p)return EINVAL;auto& b=*static_cast<Bridge*>(p);b.join();if(b.error)return b.error;return b.runtime->drain();}
int lt_abort(void* p){
    if(!p)return EINVAL;auto& b=*static_cast<Bridge*>(p);b.join();if(b.error)return b.error;if(!b.active)return 0;
    int rc=b.step.abort();if(rc==EBUSY&&b.submitted){int e=int(cudaStreamSynchronize(b.stream));if(e)return e;b.completion_proved=true;rc=b.step.abort();}
    if(!rc)b.active=false;return rc;
}
int lt_close(void* p){if(!p)return 0;int rc=lt_abort(p);if(rc)return rc;rc=lt_drain(p);if(rc)return rc;delete static_cast<Bridge*>(p);return 0;}
}
