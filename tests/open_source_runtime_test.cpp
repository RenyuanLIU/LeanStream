#include "leanstream/runtime.h"
#include "../src/io/stream_io.h"
#include <cstdlib>
#include <unistd.h>
#include <sys/syscall.h>
#include <fcntl.h>
#include <cassert>
#include <cerrno>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <thread>
#include <vector>
#include <cstdio>
#include <atomic>
#include <functional>
#include <stdexcept>
using namespace leanstream::runtime;
static std::mutex gate_mutex;static std::condition_variable gate_cv;
static bool hold=false,fail_read=false,short_read=false;static int started=0;
extern "C" ssize_t stream_test_pread(int fd,void* p,size_t n,off_t o) {
    {std::unique_lock<std::mutex> lock(gate_mutex);++started;gate_cv.notify_all();gate_cv.wait(lock,[]{return !hold;});}
    if(fail_read){errno=EIO;return -1;}
    if(short_read)return 0;
    return syscall(SYS_pread64,fd,p,n,o);
}
static void block(){std::lock_guard<std::mutex> lock(gate_mutex);hold=true;started=0;}
static void release(){std::lock_guard<std::mutex> lock(gate_mutex);hold=false;gate_cv.notify_all();}
static void await_started(){std::unique_lock<std::mutex> lock(gate_mutex);assert(gate_cv.wait_for(lock,std::chrono::seconds(3),[]{return started>0;}));}
struct Fixture {
    static constexpr int N=12,B=512;
    int fd;char path[64]="./leanstream-public-api-XXXXXX";
    std::vector<unsigned char> bytes=std::vector<unsigned char>(N*B);
    void* arena=nullptr;SlotId cache[N];uint32_t states[N+1]{};
    GroupLayout layout[N];ExecDesc desc[N]{};uint32_t indices[N]{};QueueControl control{};
    Fixture(){fd=mkstemp(path);assert(fd>=0);assert(!posix_memalign(&arena,512,N*B));
        for(int i=0;i<N;i++){layout[i]={uint32_t(i),uint32_t(N-1-i),uint64_t((N-1-i)*B),B};cache[i]=-1;
            std::fill(bytes.begin()+i*B,bytes.begin()+(i+1)*B,i);}
        assert(write(fd,bytes.data(),bytes.size())==ssize_t(bytes.size()));assert(!fcntl(fd,F_SETFL,fcntl(fd,F_GETFL)|O_DIRECT));cache[0]=7;}
    ~Fixture(){close(fd);unlink(path);free(arena);}
    LayerStorage storage(Span<GroupId> core={}){return {fd,{layout,N},cache,arena,states,core,{desc,indices,&control,N}};}
    ManagerConfig config(){return {N,1,B,B,512,ConsumerMode::Progressive};}
};
static void prepared_storage() {
    Fixture f;WeightManager m(f.config());GroupId core[]={0};auto s=f.storage({core,1});s.layout_generation=1;s.plan_generation=1;
    assert(!m.prepare_layer(s));assert(m.epoch()==0&&m.read_requests()==0&&f.control.epoch==0);
    assert(!m.begin_layer(0,s));assert(m.prepare_layer(s)==EBUSY);GroupId final[]={0};assert(!m.finalize({final,1}));assert(!m.retire());
    assert(!m.begin_layer(0,s));assert(!m.finalize({final,1}));assert(!m.retire());
    s.plan_generation=2;core[0]=1;assert(m.prepare_layer(s)==EINVAL);core[0]=0;
    assert(!m.begin_layer(0,s));assert(!m.finalize({final,1}));assert(!m.retire());
}
static void registered_layout() {
    Fixture f;WeightManager manager(f.config());auto storage=f.storage();storage.layout_generation=1;
    assert(!manager.begin_layer(0,storage));assert(!manager.retire());
    assert(!manager.begin_layer(0,storage));assert(!manager.retire());
    storage.layout_generation=2;f.layout[3].logical_group=f.layout[2].logical_group;
    assert(manager.begin_layer(0,storage)==EINVAL);
    f.layout[3].logical_group=3;storage.layout_generation=1;
    assert(!manager.begin_layer(0,storage));assert(!manager.retire());
}
static void queue_semantics() {
    Fixture f;ExecutionQueue q(f.storage().queue);q.reset(7);
    q.seal(4);int order[]={2,0,3,1};for(auto i:order)q.publish(i);
    assert(f.control.ready_count==4&&f.control.batch_epoch==7);
    for(int i=0;i<4;i++)assert(f.indices[i]==uint32_t(order[i]));
    q.reset(8);q.seal(0);assert(f.control.batch_epoch==8&&f.control.ready_count==0);
}
static void updates_and_finalize() {
    Fixture f;WeightManager m(f.config());GroupId core[]={0};int rc=m.begin_layer(1,f.storage({core,1}));if(rc)fprintf(stderr,"begin rc=%d errno=%d arena=%p\n",rc,errno,f.arena);assert(!rc);
    block();GroupId p0[]={0,1,2,3,1};assert(!m.update_prediction({p0,5}));await_started();
    GroupId p1[]={1,4,5};for(int i=0;i<8;i++)assert(!m.update_prediction({p1,3}));
    assert(m.prediction_generation()==9);assert(m.begin_layer(2,f.storage({core,1}))==EBUSY);assert(m.group_state(1).io_state==IOState::Inflight);
    GroupId final[]={0,1,4,6};assert(!m.finalize({final,4}));assert(m.finalize({final,4})==EINVAL);
    release();assert(!m.wait_io(3000));assert(!m.retire());
    assert(f.control.total_count==3&&f.control.ready_count==3);
    bool seen[12]{};for(int i=0;i<3;i++){auto& d=f.desc[f.indices[i]];assert(!seen[d.logical_group]);seen[d.logical_group]=true;
        assert(d.logical_group==1||d.logical_group==4||d.logical_group==6);
        assert(d.physical_group==uint32_t(11-d.logical_group));assert(d.source_slot==d.logical_group);
        assert(((unsigned char*)f.arena)[d.source_slot*512]==d.physical_group);}
    assert(!seen[2]&&!seen[3]&&!seen[5]&&!seen[0]);
    assert(m.read_requests()==3); // final update drops all obsolete unsubmitted work
    assert(!m.begin_layer(2,f.storage({core,1})));assert(m.epoch()==2);
    assert(!m.finalize({core,1}));assert(f.control.total_count==0&&f.control.batch_epoch==2);assert(!m.retire());
}
static void ready_and_cache_reuse() {
    Fixture f;WeightManager m(f.config());assert(!m.begin_layer(0,f.storage()));
    GroupId id[]={3};assert(!m.update_prediction({id,1}));
    auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(3);
    while(m.group_state(3).io_state!=IOState::Ready){assert(std::chrono::steady_clock::now()<deadline);std::this_thread::yield();}
    GroupId many[300];std::fill_n(many,300,3);assert(!m.update_prediction({many,300}));
    GroupId final[]={0,3};assert(!m.finalize({final,2}));assert(!m.wait_io(3000));assert(!m.retire());
    assert(m.read_requests()==1&&f.control.ready_count==2);
    assert(f.desc[0].source_kind==SourceKind::Cache&&f.desc[0].source_slot==7);
}
struct FakeGPU {int launches=0;bool done=false;};
static int enqueue(void* p,uint32_t,const void*,void*,void*,Span<ExecDesc> a,DeviceExecQueue q,uint32_t e,ConsumerMode){auto& g=*(FakeGPU*)p;g.launches++;assert(a.size==1);assert(q.control->epoch!=e);return 0;}
static bool complete(void* p,uint32_t){return ((FakeGPU*)p)->done;}
static int commit(void*,uint32_t,Span<GroupId>){return 0;}
static int partial_enqueue_error(void* ptr,uint32_t,const void*,void*,void*,Span<ExecDesc>,DeviceExecQueue,uint32_t,ConsumerMode){((FakeGPU*)ptr)->launches++;return EIO;}
static void partial_enqueue_lifetime(){Fixture f;FakeGPU gpu;Runtime r(f.config(),{&gpu,partial_enqueue_error,complete,commit});LayerStep step;
    GroupId core[]={0};assert(!r.begin_layer(0,f.storage({core,1}),&step));assert(step.enqueue_compute(nullptr,nullptr,nullptr)==EIO);
    assert(step.abort()==EBUSY);gpu.done=true;assert(!step.abort());assert(gpu.launches==1);
}
static void api_schedule() {
    Fixture f;FakeGPU gpu;Runtime r(f.config(),{&gpu,enqueue,complete,commit});LayerStep step;
    GroupId core[]={0},pred[]={1,2},final[]={0,1,2};assert(!r.begin_layer(0,f.storage({core,1}),&step));
    assert(!step.update_prediction({pred,2}));assert(!step.enqueue_compute(nullptr,nullptr,nullptr));
    for(int i=0;i<4;i++){assert(!step.update_prediction({pred,2}));}
    assert(!step.finalize({final,3}));
    assert(step.update_prediction({pred,2})==EINVAL);assert(step.commit({final,3})==EBUSY);
    assert(!r.weights().wait_io(3000));gpu.done=true;assert(!step.commit({final,3}));assert(gpu.launches==1);
    assert(step.finalize({final,3})==EINVAL);
    LayerStep old=step;gpu.done=false;
    assert(!r.begin_layer(0,f.storage({core,1}),&step));assert(old.update_prediction({pred,2})==EINVAL);
    assert(!step.enqueue_compute(nullptr,nullptr,nullptr));assert(step.abort()==EBUSY);
    assert(f.control.error==ECANCELED);gpu.done=true;assert(!step.abort());
}
static void io_failure_and_recovery() {
    Fixture f;WeightManager m(f.config());assert(!m.begin_layer(0,f.storage()));
    fail_read=true;GroupId ids[]={1};assert(!m.finalize({ids,1}));
    assert(m.wait_io(3000)==EIO);assert(f.control.error==EIO);assert(m.retire()==EIO);
    fail_read=false;assert(!m.begin_layer(0,f.storage()));assert(f.control.error==0);
    assert(!m.finalize({ids,1}));assert(!m.wait_io(3000));assert(!m.retire());
    Fixture other;FakeGPU gpu;Runtime r(other.config(),{&gpu,enqueue,complete,commit});
    LayerStep step;GroupId core[]={0};assert(!r.begin_layer(0,other.storage({core,1}),&step));
    assert(!step.enqueue_compute(nullptr,nullptr,nullptr));assert(!step.finalize({core,1}));
    assert(step.abort()==EBUSY);assert(step.commit({core,1})==ECANCELED);
    gpu.done=true;assert(!step.abort());
}
struct DeferredGPU {
    void (*retired[2])(void*,uint32_t){};void* context[2]{};Span<GroupId> targets[2]{};uint32_t epoch[2]{};
};
static int deferred_enqueue(void*,uint32_t,const void*,void*,void*,Span<ExecDesc>,DeviceExecQueue,uint32_t,ConsumerMode){return 0;}
static bool deferred_complete(void*,uint32_t){return true;}
static int defer(void* ptr,uint32_t layer,Span<GroupId> ids,uint32_t epoch,void (*done)(void*,uint32_t),void* ctx) {
    auto& gpu=*(DeferredGPU*)ptr;gpu.retired[layer]=done;gpu.context[layer]=ctx;gpu.targets[layer]=ids;gpu.epoch[layer]=epoch;return 0;
}
static void banked_commit() {
    Fixture a,b;DeferredGPU gpu;Runtime r(a.config(),{&gpu,deferred_enqueue,deferred_complete,nullptr,defer},2);
    LayerStep first,second;GroupId ids[]={0};
    assert(!r.begin_layer(0,a.storage(),&first));assert(!first.enqueue_compute(nullptr,nullptr,nullptr));
    assert(!first.finalize({ids,1}));assert(!first.commit({ids,1}));ids[0]=9;
    assert(gpu.targets[0][0]==0); // copied before caller reuses policy buffer
    assert(r.begin_layer(0,a.storage(),&second)==EBUSY);
    assert(!r.begin_layer(1,b.storage(),&second));GroupId next[]={0};
    assert(!second.enqueue_compute(nullptr,nullptr,nullptr));assert(!second.finalize({next,1}));assert(!second.commit({next,1}));
    gpu.retired[1](gpu.context[1],gpu.epoch[1]);gpu.retired[0](gpu.context[0],gpu.epoch[0]);assert(!r.drain());
    assert(!r.begin_layer(0,a.storage(),&first));
    gpu.retired[0](gpu.context[0],gpu.epoch[0]); // delayed old retirement cannot free new bank
    assert(r.begin_layer(0,a.storage(),&second)==EBUSY);assert(!first.abort());
}
static std::function<void(int)> finalize_hook;
extern "C" void leanstream_test_finalize_hook(int point){if(finalize_hook)finalize_hook(point);}
struct ManualReady {
 Fixture& f;IOCompletionCallback cb=nullptr;void* ctx=nullptr;uint32_t epoch=0;
 explicit ManualReady(Fixture& f):f(f){}
 static int begin(void* p,IOCompletionCallback cb,void* ctx,uint32_t e){auto& b=*(ManualReady*)p;b.cb=cb;b.ctx=ctx;b.epoch=e;return 0;}
 static int update(void*,Span<GroupId>){return 0;}
 static int consume(void*,Span<GroupId>){return 0;}
 static int retire(void*){return 0;} // test joins all producers before retirement
 static SlotId ready(void* p,GroupId g){auto& b=*(ManualReady*)p;return __atomic_load_n(b.f.states+g,__ATOMIC_ACQUIRE)==STREAM_READY?g:-1;}
 IOBackend backend(){return {this,begin,update,consume,retire,ready};}
 void complete(GroupId g){std::fill_n((unsigned char*)f.arena+g*f.B,f.B,11-g);__atomic_store_n(f.states+g,STREAM_READY,__ATOMIC_RELEASE);cb(ctx,{epoch,g,g,0,1});}
};
static void handoff_races(){for(int point:{0,1,2}){
 Fixture f;ManualReady io(f);auto cfg=f.config();cfg.io_backend=io.backend();WeightManager m(cfg);
 assert(!m.begin_layer(0,f.storage()));f.states[3]=STREAM_INFLIGHT;std::thread producer;
 finalize_hook=[&](int at){if(at!=point)return;
  if(point==1){ // callback blocked by registration's producer lock; state already READY
   std::fill_n((unsigned char*)f.arena+3*f.B,f.B,8);__atomic_store_n(f.states+3,STREAM_READY,__ATOMIC_RELEASE);
   static std::atomic<bool> started{false};started=false;producer=std::thread([&]{started=true;io.cb(io.ctx,{io.epoch,3,3,0,1});});
   while(!started.load())std::this_thread::yield(); // no callback join under the producer lock
  }else io.complete(3);
 };
 GroupId final[]={3};assert(!m.finalize({final,1}));finalize_hook={};if(producer.joinable())producer.join();
 io.cb(io.ctx,{io.epoch,3,3,0,1}); // duplicate delivery must not republish
 assert(f.control.ready_count==1&&f.indices[0]==0&&f.desc[0].logical_group==3);
 assert(!m.retire());
}}
static void concurrent_prefix(){Fixture f;ManualReady io(f);auto cfg=f.config();cfg.io_backend=io.backend();WeightManager m(cfg);assert(!m.begin_layer(0,f.storage()));
 GroupId final[11];for(int i=0;i<11;i++)final[i]=i+1;assert(!m.finalize({final,11}));
 std::atomic<bool> go{false},done{false};std::thread observer([&]{while(!done.load()){
  auto count=__atomic_load_n(&f.control.ready_count,__ATOMIC_ACQUIRE);bool seen[12]{};assert(count<=11);
  for(uint32_t pos=0;pos<count;pos++){uint32_t idx=f.indices[pos];assert(idx<11&&!seen[idx]);seen[idx]=true;auto d=f.desc[idx];assert(d.logical_group==idx+1&&d.source_slot==idx+1);assert(((unsigned char*)f.arena)[d.source_slot*512]==11-d.logical_group);}
 }});
 std::vector<std::thread> producers;for(int g=1;g<12;g++)producers.emplace_back([&,g]{while(!go.load())std::this_thread::yield();io.complete(g);io.cb(io.ctx,{io.epoch,g,g,0,1});});
 go=true;for(auto& t:producers)t.join();done=true;observer.join();assert(f.control.ready_count==11);assert(!m.retire());
}
static void failed_finalize_is_terminal(){Fixture f;FakeGPU gpu;Runtime r(f.config(),{&gpu,enqueue,complete,commit});LayerStep step;GroupId core[]={0},bad[]={1};
 assert(!r.begin_layer(0,f.storage({core,1}),&step));assert(!step.enqueue_compute(nullptr,nullptr,nullptr));
 assert(step.finalize({bad,1})==EINVAL);assert(__atomic_load_n(&f.control.error,__ATOMIC_ACQUIRE)==EINVAL);assert(step.abort()==EBUSY);gpu.done=true;assert(!step.abort());
 assert(!r.begin_layer(0,f.storage({core,1}),&step));assert(!step.enqueue_compute(nullptr,nullptr,nullptr));assert(!step.finalize({core,1}));assert(!step.commit({core,1}));
}
static void capacity_contract(){auto bad=Fixture().config();bad.group_count=0xffffffffu;bool rejected=false;try{WeightManager invalid(bad);}catch(const std::invalid_argument&){rejected=true;}assert(rejected);Fixture f;WeightManager m(f.config());auto s=f.storage();s.staging_bytes=Fixture::N*Fixture::B-1;assert(m.begin_layer(0,s)==ENOSPC);s.staging_bytes=Fixture::N*Fixture::B;assert(!m.begin_layer(0,s));assert(!m.retire());}
static void short_read_failure(){Fixture f;WeightManager m(f.config());short_read=true;assert(!m.begin_layer(0,f.storage()));GroupId final[]={3};assert(!m.finalize({final,1}));assert(m.wait_io(3000)==EIO);assert(f.control.error==EIO);assert(m.retire()==EIO);short_read=false;assert(!m.begin_layer(0,f.storage()));assert(!m.finalize({final,1}));assert(!m.wait_io(3000));assert(!m.retire());}
static void prediction_budget(){Fixture f;WeightManager m(f.config());assert(!m.begin_layer(0,f.storage()));block();GroupId ids[]={0,1,2,3};assert(!m.update_prediction({ids,4},{1,m.epoch(),17}));await_started();release();assert(m.update_prediction({ids,4},{1,m.epoch()+1,18})==ESTALE);GroupId final[]={0,1};assert(!m.finalize({final,2}));assert(!m.wait_io(3000));assert(m.prediction_generation()==1);assert(!m.retire());assert(m.read_requests()==1);}
int main(){prediction_budget();handoff_races();concurrent_prefix();failed_finalize_is_terminal();capacity_contract();short_read_failure();prepared_storage();partial_enqueue_lifetime();registered_layout();queue_semantics();updates_and_finalize();ready_and_cache_reuse();api_schedule();io_failure_and_recovery();banked_commit();puts("PASS: queue, dynamic priority, cache/ready/inflight reuse, exact final membership, epoch reset, API lifecycle");}
