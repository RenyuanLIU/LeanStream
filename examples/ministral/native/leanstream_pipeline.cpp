#include "leanstream/triton_cuda.h"
#include <cuda.h>
// Coarse native orchestration; no Python group arrays in online decode.
#include "leanstream_control.h"
#include "attention_priority.h"
#include "priority_io.h"
#ifdef STREAM_IO_V2
#include "leanstream_stream_io.h"
extern "C" int pio_stream_profile(void*,StreamProfile*,StreamUpdateProfile*,int);
#endif
#include "execution_queue.h"
#include "leanstream/execution_queue.h"
#include "leanstream/runtime.h"
#include "ready_id_queue.h"
#include <memory>
#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <unistd.h>
#include <fcntl.h>
#include <sched.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <string>
#include <atomic>
#include <cuda_runtime.h>
#include <pthread.h>
#include <time.h>
#ifdef LEANSTREAM_NVTX_PROFILE
#include <nvToolsExt.h>
struct LeanstreamNvtxScope {
    explicit LeanstreamNvtxScope(const char* name){nvtxRangePushA(name);}
    ~LeanstreamNvtxScope(){nvtxRangePop();}
};
#endif
extern "C" {
void* lm_create();int lm_configure(void*,int);int lm_configure_model(void*,int,int,int,int,int);
int si_configure(void*,int,int);int si_configure_model(void*,int,int,int,int,int);
int si_set_projection(void*,const float*,int);const float* si_scores(void*);void si_capture_scores(void*,int);void lm_destroy(void*);void* lm_pointer(void*,int,int);int lm_wait(void*,int);
int lm_dequant(void*,int,const void*,void*,void*);int lm_keep(void*,void*);
double si_ready_ms(void*);int si_worker_init(void*);int si_enable_ranking(void*);
int si_start_norm(void*,const void*,const void*,float,void*);
int si_finish_norm(void*,const int32_t*,const void*,const float*,const int32_t*,int32_t*,double*);
int si_finish_norm_ready(void*,const int32_t*,const void*,const float*,const int32_t*,int32_t*,double*);
const uint32_t* si_ready_flag(void*);uint32_t si_expected_epoch(void*);
const uint32_t* lm_ready_flag(void*,int);uint32_t lm_expected_epoch(void*,int);
void lm_notify_doorbell(void*,uint32_t*);void si_notify_doorbell(void*,uint32_t*);
int lm_packed_update(void*,int,void*,void*);
int lm_no_waste_config(void*,int);
int lm_stream_synchronize(void*);
void* si_create(const char*);void si_destroy(void*);const char* si_error();const char* lm_error();
int si_predict_ranked(void*,const uint16_t*,const int32_t*,const void*,const float*,const int32_t*,int32_t*,double*,int32_t*);
int si_predict_ready(void*,const uint16_t*,const int32_t*,const void*,const float*,const int32_t*,int32_t*,double*);
int si_predict_subset(void*,const uint16_t*,const int32_t*,const void*,const float*,const int32_t*,int,int,int32_t*,double*);
int si_set_active_features(void*,int);
int si_run(void*,int,const void*,const void*,float,void*,const int32_t*,const void*,const float*,const int32_t*,int32_t*,double*,float*);
}
using namespace leanstream;
static double ms(){return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();}
static uint64_t ns(){return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}
static thread_local char error[256];
static int fail(const char* s){std::snprintf(error,sizeof(error),"%s",s);std::fprintf(stderr,"LeanStream native failure: %s\n",error);return -1;}
struct Layer {
    CacheState cache;int fd=-1;
    const void* weights=nullptr;const float* scales=nullptr;const int32_t* features=nullptr;
    const void* mid_weights=nullptr;const float* mid_scales=nullptr;const int32_t* mid_features=nullptr;
    const void* lite_weights=nullptr;const float* lite_scales=nullptr;const int32_t* lite_features=nullptr;
    const void* device_scales=nullptr;void* rows=nullptr;
    std::array<int64_t,MAX_GROUPS> offsets{};
    const float* projection=nullptr;int projection_stride=0;
    const int32_t* speculative_hot=nullptr;int speculative_hot_count=0;
};
static constexpr int max_layers=34;
static constexpr int no_waste_max_extra=32+128+215;
static constexpr int no_waste_max_optional=215+no_waste_max_extra;
static constexpr int no_waste_max_exec=1434+no_waste_max_extra;
struct ReadyArm {int state=0,hidden=0,layer=-1;uint64_t token=0;const uint32_t* flag=nullptr;uint32_t epoch=0;};
static void ready_unregister(void*);
extern "C" double si_predict_start_ms(void*);
extern "C" double si_predict_end_ms(void*);
#ifdef STREAM_IO_V2
static constexpr int staging_slots=MAX_GROUPS;
extern "C" int pio_stream_end(void*);
extern "C" void* pio_stream_create(int,const int64_t*,int,uint64_t,void*,uint64_t,void*);
extern "C" int pio_stream_consume(void*,const int32_t*,int,uint64_t,uint64_t*,int*);
#else
static constexpr int staging_slots=256;
#endif
struct Pipeline;
static int public_update(Pipeline&,const int32_t*,int,bool);
static int public_begin(Pipeline&);
static int public_commit(Pipeline&,const int32_t*,int);
static int public_prepare(Pipeline&);
static void public_shutdown(Pipeline&);
extern "C" int lp_control_arm(void*,int,int);
extern "C" int lp_control_submit(void*,int,int);
struct PublicPlan {
    leanstream::runtime::DeviceExecQueue view{};
    std::unique_ptr<leanstream::runtime::ExecutionQueue> queue;
    std::array<int32_t,MAX_GROUPS> index{};
    std::array<uint8_t,MAX_GROUPS> seen{};
    std::mutex mutex,begin_mutex; bool armed=false, progressive=true;
    std::unique_ptr<leanstream::runtime::Runtime> api;
    leanstream::runtime::LayerStep step;
    std::atomic<bool> active{false};
    std::vector<leanstream::runtime::GroupLayout> layout;
    std::vector<int32_t> cache_snapshot,destinations,final_ids,reference_b;
    leanstream::runtime::IOCompletionCallback callback=nullptr;void* callback_context=nullptr;uint32_t callback_epoch=0;
    bool commit_prediction=false,final_committed=false;uint64_t published_ns=0;int outstanding=0;
    int (*compute)(void*)=nullptr;void* compute_context=nullptr;cudaEvent_t done_event=nullptr;
    uint32_t expected_gpu_epoch=0;uint32_t* compat_ready=nullptr;uint32_t* reduction_done=nullptr;int reduction_programs=0;cudaStream_t compute_stream=nullptr;bool compute_error=false;
    leanstream::runtime::TritonBackend gpu;bool cuda_bound=false;uint32_t bound_stages=0;
    double api_times[5]{};int layout_layer=-1;bool prepared=false;uint64_t storage_generation=0;int prepared_layer=-1;
    bool verify=getenv("LEANSTREAM_PUBLIC_VALIDATE")&&atoi(getenv("LEANSTREAM_PUBLIC_VALIDATE"));
    void reset(uint32_t epoch){std::lock_guard<std::mutex> lock(mutex);if(api){expected_gpu_epoch=epoch;return;}armed=false;if(progressive){index.fill(-1);seen.fill(0);}queue->reset(epoch);}
    void complete(const Completion& c){if(api){if(callback)callback(callback_context,{callback_epoch,c.group,c.slot,c.error,c.sequence});if(c.error&&compat_ready)__atomic_store_n(compat_ready,~callback_epoch,__ATOMIC_RELEASE);return;}std::lock_guard<std::mutex> lock(mutex);
        if(!armed)return;if(c.error){queue->fail(c.error);return;}
        int idx=index[c.group];if(idx<0||seen[c.group])return;
        if(c.slot!=c.group){queue->fail(EINVAL);return;}
        seen[c.group]=1;queue->publish(idx);
    }
};
struct Pipeline {
    std::unique_ptr<PublicPlan> public_plan;
    bool diagnostic_trace=false;std::array<uint64_t,32> control_trace{};
    void stamp(int phase,int index){if(diagnostic_trace)control_trace[(phase-1)*16+index]=ns();}
    int group_size=2,groups=7168,keep_size=1219,fill_size=215,group_bytes=12288;
    int hidden_size=4096,intermediate_size=14336,layer_count=32;bool stable=false,model_configured=false;
    std::array<Layer,max_layers> layers;
    std::array<int32_t,CPU_SETSIZE> io_cpus{};int io_cpu_count=0;
    void *mapped=nullptr,*shared=nullptr,*lite_shared=nullptr,*tiny_mid_shared=nullptr,*io=nullptr;
    uint64_t generation=0;std::atomic<int> current{-1};
    std::array<int32_t,MAX_FILL> fill1{},fill2{},fill_mid{};
    std::atomic<uint64_t> submitted_token{0};
    std::atomic<uint64_t> p2_queued_token{0};
    std::atomic<uint64_t> continuation_token{0};
    std::array<uint64_t,3> priority_trace{};
    bool mid_publish_enabled=true; // Opt-in shadow control; set only at a drained boundary.
    uint64_t mid_generation=0;std::array<double,12> mid_times{};std::array<uint64_t,7> mid_trace{};
    int cheap_p1_size=0,cheap_mid_size=0,mid_count=0;
    int p0_budget=0,lite_budget=0,lite_features=0;
    int tiny_mid_features=0,p1_priority_count=0,mid_pending_count=0;
    int semantic_mid_rescue=0; // Opt-in full semantic ranking; bounded new I/O IDs.
    std::array<uint64_t,4> semantic_mid_metrics{}; // rescue, pending, published, stale-after-score
    std::array<double,24> relaxed_probe{}; // Diagnostic-only: three M/P candidate snapshots, eight values each.
    int relaxed_p2_m=0,relaxed_p2_p=0;
    std::array<double,4> relaxed_metrics{}; // actual substitutions, relative score loss, cache saved, selection ms
    // Diagnostic snapshot taken before strict P2 changes the I/O generation.
    // 0 EMPTY, 1 CACHE, 2 valid READY, 3 INFLIGHT. Session-local mapped
    // stream states provide the acquire; no completed payload is copied here.
    std::array<int32_t,MAX_GROUPS> no_waste_state{},no_waste_cache{};
    std::array<int32_t,MAX_KEEP> no_waste_keep{};
    std::array<int32_t,MAX_FILL> no_waste_strict_fill{};
    std::array<float,MAX_GROUPS> no_waste_scores{};
    std::array<int32_t,10> no_waste_counts{};
    int no_waste_policy=0,no_waste_new_count=0,no_waste_optional_count=0,no_waste_exec_count=0;
    int32_t* no_waste_control=nullptr;void* no_waste_maintenance_stream=nullptr;void* no_waste_packed_cache=nullptr;
    std::array<int32_t,MAX_FILL> no_waste_new{};
    std::array<int32_t,no_waste_max_exec> no_waste_exec_ids{};
    std::array<double,12> no_waste_metrics{};
    std::array<int32_t,128> p1_priority{},mid_before{},mid_after{};
    std::array<int32_t,64> no_waste_p0_at_p2{};
    std::array<int32_t,128> no_waste_p1_at_p2{};
    // Diagnostic only: pending, top-16/32/64 P1/Mid final-required coverage,
    // start/end/publish, and P2-ready-while-active counter.
    std::array<uint64_t,12> tiny_mid_metrics{};
    int p0_count=0;
    std::array<int32_t,64> p0_rank{};
    std::array<uint64_t,8> early_trace{}; // P0/P1 publication, requested counts, generations, P2 publication.
    std::array<int32_t,2048> spec_pool{};
    std::array<uint8_t,MAX_GROUPS> spec_marks{};
    std::array<uint64_t,MAX_GROUPS> completed_generation{};
    std::array<int32_t,512> continuation{},speculative{};
    int final_qd=1,pre_qd=1,pool_qd=1;bool speculate=true;uint64_t idle_at_p1=0;double idle_before_p2_ms=0;
    std::array<int32_t,MAX_GROUPS> host_slot{};
    std::array<uint64_t,staging_slots> sequences{};
    std::array<uint64_t,MAX_GROUPS> completed_ns{};
    uint64_t p2_published=0,p2_generation=0;
    std::array<int32_t,staging_slots> owners{};
    std::array<uint8_t,MAX_GROUPS> wanted{};
    std::array<Completion,staging_slots> completions;
    // p1quant,w4,total,publish; p2quant,w4,total,publish; plan,wait,keep;
    // p1 requested,completed before P2,final ready at P2,missing,fill overlap,waste,io calls;
    std::array<double,24> times{};
    std::string predictor_path;
    std::atomic<bool> maintenance_pending{false};
    std::mutex maintenance_mutex;std::condition_variable maintenance_cv;
    std::string maintenance_error;bool prebuilt_a=false,async_control=false;
    bool ready_driven=false,p1_prepared=false,io_prebound=false;
    int ready_notification_mode=0; // 0: mapped-epoch scan; 1: one adaptive GPU doorbell.
    uint32_t p1_epoch=0,mid_epoch=0,p2_epoch=0;
    uint64_t p1_io_begin_ns=0,p1_io_end_ns=0;
    std::array<ReadyArm,7> predictor_arms{};
    bool checkpoint_publishing=false,prediction_input_enabled=true;uint32_t input_io_budget=UINT32_MAX;
    uint32_t checkpoint_mask=0;std::array<const uint16_t*,3> checkpoint_data{};std::array<const uint32_t*,3> checkpoint_flags{};
    std::array<uint32_t,3> checkpoint_budget{};std::array<uint64_t,15> checkpoint_stats{};std::array<int32_t,MAX_FILL> checkpoint_rank{};
    std::array<std::array<uint64_t,6>,7> ready_trace{};
    std::array<uint64_t,4> final_io_state{}; // Diagnostic snapshot before next-session prebind.
    std::mutex control_mutex;std::condition_variable control_cv;int control_jobs=0;std::string control_error;
    uint64_t main_waits=0;double main_wait_ms=0;
    bool legacy_p1=false; // A/B only: production never waits for worker start.
    int32_t *desc_a=nullptr,*desc_b=nullptr;uint32_t* split_ready=nullptr;uint32_t split_epoch=0;
    std::unique_ptr<ExecutionQueue> execution;
    std::unique_ptr<ReadyClient> ready_ids;uint64_t ready_epoch=0;
    // CPU steady_clock ns: last required pread, worker wake, descriptors ready,
    // ready-store before/after, immutable metadata ready.
    std::array<uint64_t,6> ready_times{};
    std::array<int32_t,MAX_GROUPS> missing_index{};
    std::array<int32_t,MAX_FILL> missing_b{};
    std::array<std::atomic<uint8_t>,MAX_FILL> final_seen{};
    std::atomic<bool> ready_armed{false};std::atomic<int> final_remaining{0};
    std::atomic<uint64_t> final_last_io{0};
    std::thread final_worker;std::mutex final_mutex;std::condition_variable final_cv;
    bool final_job=false,final_busy=false,final_stop=false;std::string final_error;
    ~Pipeline(){
        {std::unique_lock<std::mutex> lock(maintenance_mutex);maintenance_cv.wait(lock,[&]{return !maintenance_pending.load();});}
        {std::unique_lock<std::mutex> lock(control_mutex);control_cv.wait(lock,[&]{return control_jobs==0;});}
        {std::lock_guard<std::mutex> lock(final_mutex);final_stop=true;final_cv.notify_all();}
        if(final_worker.joinable())final_worker.join();
        if(public_plan&&public_plan->api)public_shutdown(*this);
        if(io)pio_close(io);if(ready_ids)ready_ids->wait_idle();if(tiny_mid_shared)si_destroy(tiny_mid_shared);if(lite_shared)si_destroy(lite_shared);if(shared)si_destroy(shared);if(mapped)lm_destroy(mapped);
        for(auto& l:layers)if(l.fd>=0)::close(l.fd);}
    int drain(bool discard,bool count_waste=true){
        int n=pio_poll(io,completions.data(),staging_slots);if(n<0)return fail("Completion queue error");
        for(int i=0;i<n;i++){
            const auto& c=completions[i];if(c.error||c.bytes!=group_bytes)return fail("SSD short/failed request");
            if(owners[c.slot]>=0 || host_slot[c.group]>=0)return fail("Duplicate staging ownership");
            owners[c.slot]=c.group;sequences[c.slot]=c.sequence;host_slot[c.group]=c.slot;
            times[17]++;completed_ns[c.group]=c.complete_ns;completed_generation[c.group]=c.generation;
            if(p2_published && c.selected_ns>=p2_published && c.generation!=p2_generation)
                return fail("Stale future I/O start");
        }
#ifdef STREAM_IO_V2
        discard=false; // Retain every successful block until the session retires.
#endif
        if(discard)for(int slot=0;slot<256;slot++)if(owners[slot]>=0&&!wanted[owners[slot]]){
            int g=owners[slot];if(pio_release(io,slot,sequences[slot]))return fail("Speculative release failed");
            owners[slot]=-1;host_slot[g]=-1;if(count_waste)times[16]++;
        }return 0;
    }
    int finish_io(){
        if(!io)return 0;
        // Called AFTER mapped importance acquire, which follows all cache/MLP GPU consumers.
#ifdef STREAM_IO_V2
        if(pio_stream_end(io))return fail("Streaming session retirement failed");
#else
        if(pio_wait_idle(io,10000))return fail("I/O quiescence failed");
#endif
        if(ready_ids)ready_ids->wait_idle();
        if(drain(true))return -1;wanted.fill(0);
#ifdef STREAM_IO_V2
        owners.fill(-1);host_slot.fill(-1);
#endif
        return drain(true,false);
    }
    int publish(const int32_t* ids,int n,bool final=false){uint64_t ns;int active;
        stamp(final?2:1,final?7:6);
#ifdef STREAM_IO_V2
        int rc;
        if(public_plan&&public_plan->api){rc=public_update(*this,ids,n,final);ns=public_plan->published_ns;active=public_plan->outstanding;}
        else rc=final?pio_stream_consume(io,ids,n,++generation,&ns,&active):pio_update(io,ids,n,++generation,n,&ns,&active);
#else
        int rc=pio_update(io,ids,n,++generation,n,&ns,&active);
#endif
        if(rc)return fail("Priority update failed");
        stamp(final?2:1,final?8:7);
        priority_trace[final?2:(priority_trace[0]?1:0)]=generation;
        if(final){p2_published=ns;p2_generation=generation;times[19]=active;early_trace[6]=ns;}
        return 0;}
};
static int await_final(Pipeline& p){
    auto& l=p.layers[p.current];const auto& plan=l.cache.plan();
    double start=ms();if(p.drain(true))return -1;
    for(;;){
        bool ready=true;for(int k=0;k<plan.missing_count;k++)if(p.host_slot[plan.missing[k]]<0){ready=false;break;}
        if(ready)break;
        if(pio_wait_completion(p.io,10000))return fail("Final P2 host-ready timeout");
        if(p.drain(true))return -1;
    }
    p.times[9]=ms()-start;
    auto* meta=static_cast<int32_t*>(lm_pointer(p.mapped,4,0));
    for(int k=0;k<plan.missing_count;k++){
        int g=plan.missing[k];meta[3*k]=p.host_slot[g];meta[3*k+1]=int(l.offsets[g]/p.group_bytes);meta[3*k+2]=plan.destinations[k];
    }
    return 0;
}
static int prepare_b(Pipeline& p){
    auto& l=p.layers[p.current];const auto& plan=l.cache.plan();
    p.missing_index.fill(-1);
    for(int k=0;k<plan.missing_count;k++)p.missing_index[plan.missing[k]]=k;
    auto* meta=static_cast<int32_t*>(lm_pointer(p.mapped,4,0));
    if(p.async_control)for(int k=0;k<p.fill_size;k++)meta[3*k+2]=-1;
    for(int k=0;k<p.fill_size;k++){
        int g=p.fill2[k],dest=l.cache.cache_resident(g)?l.cache.view(1)[g]:-1,source=dest,kind=0;
        if(dest<0){
            int j=p.missing_index[g];
            if(j<0||j>=plan.missing_count||plan.missing[j]!=g)return fail("Invalid final missing index");
            dest=plan.destinations[j];source=-1;kind=1;p.missing_b[j]=k;
            meta[3*j]=-1;meta[3*j+1]=int(l.offsets[g]/p.group_bytes);meta[3*j+2]=dest;
        }
        int32_t* d=p.desc_b+5*k;d[0]=g;d[1]=int(l.offsets[g]/p.group_bytes);d[2]=source;d[3]=dest;d[4]=kind;
    }
    p.ready_times[5]=ns();return 0;
}
// Completion-driven publication: each target is claimed exactly once, including
// requests already completed when P2 installs the immutable final target.
static void publish_ready(Pipeline& p){
    p.ready_times[0]=p.final_last_io.load(std::memory_order_acquire);
    p.ready_times[2]=ns();p.ready_times[3]=ns();
    __atomic_store_n(p.split_ready,p.split_epoch,__ATOMIC_RELEASE);
    p.ready_times[4]=ns();
    // This mutex is AFTER publication, and only wakes the accounting worker.
    {std::lock_guard<std::mutex> lock(p.final_mutex);p.final_cv.notify_all();}
}
static void organize_ready(void* ptr,int group,uint64_t epoch){
    auto& p=*static_cast<Pipeline*>(ptr);if(epoch!=p.ready_ids->epoch())return;
    int slot=-1;uint64_t ready=0;auto& cache=p.layers[p.current].cache;
    if(cache.cache_resident(group)){slot=cache.view(1)[group];ready=ns();}
    else if(pio_peek(p.io,group,&slot,&ready)||!ready)return;
    if(p.execution->complete(group,slot,ready,static_cast<int32_t*>(lm_pointer(p.mapped,4,0)))){
        p.ready_ids->mark_visible();
        p.final_remaining.store(0,std::memory_order_release);
        std::lock_guard<std::mutex> lock(p.final_mutex);p.final_cv.notify_all();
    }
}
static ReadyOrganizer& ready_organizer(){static ReadyOrganizer organizer;return organizer;}
static void final_completion(void* ptr,const Completion* c){
    auto& p=*static_cast<Pipeline*>(ptr);
    if(p.public_plan&&(p.public_plan->progressive||p.public_plan->api))p.public_plan->complete(*c);
    if(p.ready_ids){
        if(c->error){std::lock_guard<std::mutex> lock(p.final_mutex);p.final_error="ReadyID pread failed";p.final_cv.notify_all();return;}
        // Session retirement drains callbacks before ready_epoch advances.
        p.ready_ids->publish(c->group,p.ready_epoch,c->complete_ns,3);return;
    }
    if(p.execution){
        if(c->error){std::lock_guard<std::mutex> lock(p.final_mutex);p.final_error="Execution-queue pread failed";p.final_cv.notify_all();return;}
        if(p.execution->complete(c->group,c->slot,c->complete_ns,static_cast<int32_t*>(lm_pointer(p.mapped,4,0)))){
            p.final_remaining.store(0,std::memory_order_release);
            std::lock_guard<std::mutex> lock(p.final_mutex);p.final_cv.notify_all();
        }
        return;
    }
    if(!p.ready_armed.load(std::memory_order_acquire))return;
    int j=p.missing_index[c->group];if(j<0)return;
    if(c->error){std::lock_guard<std::mutex> lock(p.final_mutex);p.final_error="Final pread failed";p.final_cv.notify_all();return;}
    uint8_t unseen=0;if(!p.final_seen[j].compare_exchange_strong(unseen,1,std::memory_order_acq_rel))return;
    auto* meta=static_cast<int32_t*>(lm_pointer(p.mapped,4,0));
    meta[3*j]=c->slot;p.desc_b[5*p.missing_b[j]+2]=c->slot;
    uint64_t last=p.final_last_io.load(std::memory_order_relaxed);
    while(last<c->complete_ns&&!p.final_last_io.compare_exchange_weak(last,c->complete_ns,std::memory_order_relaxed)){}
    // The acq_rel decrement chains all target payload/descriptor writes before release.
    if(p.final_remaining.fetch_sub(1,std::memory_order_acq_rel)==1)publish_ready(p);
}
static int arm_final(Pipeline& p){
    const auto& plan=p.layers[p.current].cache.plan();
    for(auto& seen:p.final_seen)seen.store(0,std::memory_order_relaxed);
    p.final_last_io.store(0);p.final_remaining.store(plan.missing_count);
    p.ready_armed.store(true,std::memory_order_release);
    if(!plan.missing_count){publish_ready(p);return 0;}
    for(int j=0;j<plan.missing_count;j++){
        Completion c{};c.group=plan.missing[j];
        if(pio_peek(p.io,c.group,&c.slot,&c.complete_ns))return fail("Final completion reconciliation failed");
        if(c.complete_ns)final_completion(&p,&c);
    }
    return 0;
}
static int publish_final(Pipeline& p){
    const auto& plan=p.layers[p.current].cache.plan();
    auto* meta=static_cast<int32_t*>(lm_pointer(p.mapped,4,0));
    uint64_t start=ns();p.ready_times[1]=start;
    for(;;){
        bool bytes=true,descriptors=true;uint64_t last=0;
        for(int j=0;j<plan.missing_count;j++){
            int32_t slot;uint64_t complete;
            if(pio_peek(p.io,plan.missing[j],&slot,&complete))return fail("Final snapshot failed");
            if(slot<0)descriptors=false;
            else if(meta[3*j]<0){ // Patch only the source slot; everything else was prepared by P2.
                meta[3*j]=slot;p.desc_b[5*p.missing_b[j]+2]=slot;
            }
            if(!complete)bytes=false;else last=std::max(last,complete);
        }
        if(descriptors&&!p.ready_times[2])p.ready_times[2]=ns();
        if(bytes&&descriptors){
            p.ready_times[0]=last;p.ready_times[3]=ns();
            __atomic_store_n(p.split_ready,p.split_epoch,__ATOMIC_RELEASE);
            p.ready_times[4]=ns();p.times[9]=double(p.ready_times[4]-start)/1e6;
            return 0; // No completion draining, ownership bookkeeping, or mutex before publication.
        }
        // Release obsolete speculative slots only while final bytes are still unavailable.
        // Successful final snapshots remain owned until the existing GPU completion path.
        if(!descriptors && p.drain(true))return -1;
        if(ns()-start>10000000000ull)return fail("Final payload timeout");
#ifdef __aarch64__
        asm volatile("yield" ::: "memory");
#else
        std::this_thread::yield();
#endif
    }
}
extern "C" int pio_stream_begin_current(void*);
extern "C" int pio_stream_speculation_ceiling(void*,uint64_t);
static leanstream::runtime::LayerStorage public_storage(Pipeline& p,bool refresh,uint64_t generation){
    auto& q=*p.public_plan;auto& l=p.layers[p.current];
    if(q.layout_layer!=p.current){
        for(int g=0;g<p.groups;g++)q.layout[g]={uint32_t(g),uint32_t(l.offsets[g]/p.group_bytes),uint64_t(l.offsets[g]),uint32_t(p.group_bytes)};
        q.layout_layer=p.current;
    }
    if(refresh){
        std::memcpy(q.cache_snapshot.data(),l.cache.view(1),size_t(p.groups)*sizeof(int32_t));
        std::memcpy(q.destinations.data(),q.cache_snapshot.data(),size_t(p.groups)*sizeof(int32_t));
    }
    return {l.fd,{q.layout.data(),size_t(p.groups)},q.cache_snapshot.data(),
        lm_pointer(p.mapped,3,0),static_cast<uint32_t*>(lm_pointer(p.mapped,5,0)),
        {l.cache.view(2),size_t(p.keep_size)},q.view,nullptr,q.destinations.data(),uint64_t(p.current+1),generation};
}
static int public_prepare(Pipeline& p){
    auto& q=*p.public_plan;if(q.active)return EBUSY;q.prepared=false;
    auto storage=public_storage(p,true,++q.storage_generation);
    int rc=q.api->weights().prepare_layer(storage);if(!rc){q.prepared_layer=p.current;q.prepared=true;}return rc;
}
static int public_begin(Pipeline& p){
    auto& q=*p.public_plan;if(q.active.load(std::memory_order_acquire))return 0;
    std::lock_guard<std::mutex> lock(q.begin_mutex);if(q.active)return 0;
    if(p.current<0||!p.io)return EAGAIN;
    std::fill(q.api_times,q.api_times+5,0.);uint64_t begin_ns=ns();
    bool prepared=q.prepared&&q.prepared_layer==p.current;
    auto storage=public_storage(p,!prepared,prepared?q.storage_generation:0);
    q.final_committed=false;int rc=q.api->begin_layer(0,storage,&q.step);
    if(!rc)q.active.store(true,std::memory_order_release);q.api_times[0]=double(ns()-begin_ns)/1e6;return rc;
}
static int public_io_begin(void* ptr,leanstream::runtime::IOCompletionCallback cb,void* ctx,uint32_t epoch){
    auto& p=*static_cast<Pipeline*>(ptr);auto& q=*p.public_plan;
    q.callback=cb;q.callback_context=ctx;q.callback_epoch=epoch;
    int rc=pio_stream_begin_current(p.io);
    if(!rc&&p.checkpoint_mask){rc=pio_stream_speculation_ceiling(p.io,160);}
    return rc;
}
static int public_io_update(void* ptr,leanstream::runtime::Span<int32_t> ids){
    auto& p=*static_cast<Pipeline*>(ptr);auto& q=*p.public_plan;int rc;
    if(q.commit_prediction){rc=pio_stream_consume(p.io,ids.data,ids.size,++p.generation,&q.published_ns,&q.outstanding);if(!rc)q.final_committed=true;}
    else rc=pio_update(p.io,ids.data,ids.size,++p.generation,ids.size,&q.published_ns,&q.outstanding);
    return rc;
}
static int public_io_consume(void* ptr,leanstream::runtime::Span<int32_t>){
    auto& q=*static_cast<Pipeline*>(ptr)->public_plan;
    // Current policy's exact final union consists only of already-admitted data
    // plus the committed KNEE rank. No new reads or priority change is needed.
    return q.final_committed?0:EINVAL;
}
static int public_io_retire(void* ptr){return pio_stream_end(static_cast<Pipeline*>(ptr)->io);}
static int32_t public_io_slot(void*,int32_t group){return group;} // stable stream arena
static int public_enqueue(void* ptr,uint32_t,const void*,void*,void* stream,leanstream::runtime::Span<leanstream::runtime::ExecDesc> core,
                          leanstream::runtime::DeviceExecQueue,uint32_t epoch,leanstream::runtime::ConsumerMode){
    auto& p=*static_cast<Pipeline*>(ptr);auto& q=*p.public_plan;
    if(epoch!=p.split_epoch||!q.compute)return EINVAL;
    if(q.verify&&std::memcmp(core.data,p.desc_a,core.size*sizeof(leanstream::runtime::ExecDesc)))return EINVAL;
    if(q.cuda_bound){
        q.gpu.arm_context=&p;
        q.gpu.arm_final_prediction=[](void* ctx){auto& pipeline=*static_cast<Pipeline*>(ctx);
            return pipeline.ready_driven?lp_control_arm(&pipeline,2,0):lp_control_submit(&pipeline,2,0);};
        return q.gpu.enqueue({0,0},epoch,static_cast<cudaStream_t>(stream));
    }
    return q.compute(q.compute_context);
}
static bool public_complete(void* ptr,uint32_t){auto& q=*static_cast<Pipeline*>(ptr)->public_plan;
    if(q.compute_error)return cudaStreamQuery(q.compute_stream)==cudaSuccess;
    if(q.reduction_done){for(int i=0;i<q.reduction_programs;i++)if(__atomic_load_n(q.reduction_done+i,__ATOMIC_ACQUIRE)!=q.expected_gpu_epoch)return false;return true;}
    return q.done_event&&cudaEventQuery(q.done_event)==cudaSuccess;
}
static int public_cache_commit(void*,uint32_t,leanstream::runtime::Span<int32_t>){return 0;} // original keeper installed bytes
static void public_cancel(void* ptr,uint32_t,uint32_t epoch){auto& p=*static_cast<Pipeline*>(ptr);__atomic_store_n(p.split_ready,~epoch,__ATOMIC_RELEASE);}
static int public_update(Pipeline& p,const int32_t* ids,int n,bool final){
    int rc=public_begin(p);if(rc)return rc;auto& q=*p.public_plan;
    uint64_t begin_ns=ns();q.commit_prediction=final;rc=(!final&&!p.checkpoint_publishing&&p.input_io_budget!=UINT32_MAX)?q.step.update_prediction({ids,size_t(n)},{p.input_io_budget,0,0}):q.step.update_prediction({ids,size_t(n)});q.commit_prediction=false;q.api_times[1]+=double(ns()-begin_ns)/1e6;return rc;
}
static int public_commit(Pipeline& p,const int32_t* ids,int n){
    auto& q=*p.public_plan;if(!q.active)return 0;
    // Existing background maintenance owns this wait; not a profiling sync or
    // foreground layer dependency. Reduction must retire before queue reuse.
    uint64_t wait_start=ns();while(!public_complete(&p,0)){
        if(ns()-wait_start>10000000000ull)return ETIMEDOUT;
        std::this_thread::yield();
    }
    uint64_t begin_ns=ns();int rc=q.step.commit({ids,size_t(n)});q.api_times[4]=double(ns()-begin_ns)/1e6;if(rc)std::fprintf(stderr,"public_commit rc=%d queue_error=%u layer=%d\n",rc,q.view.control->error,int(p.current));if(!rc)q.active.store(false,std::memory_order_release);return rc;
}
static void public_shutdown(Pipeline& p){
    auto& q=*p.public_plan;if(!q.active)return;q.api->weights().cancel();
    __atomic_store_n(p.split_ready,~q.callback_epoch,__ATOMIC_RELEASE);
    if(q.expected_gpu_epoch){uint64_t start=ns();while(!public_complete(&p,0)){
        if(ns()-start>10000000000ull){std::fprintf(stderr,"Public GPU retirement timeout: refusing to free live storage\n");std::abort();}
        std::this_thread::yield();
    }}
    q.step.abort();q.active=false;
}
static int prepare_public_plan(Pipeline& p){
#ifdef STREAM_IO_V2
    auto& q=*p.public_plan;auto& l=p.layers[p.current];
    const auto* states=static_cast<const uint32_t*>(lm_pointer(p.mapped,5,0));
    std::array<uint8_t,MAX_GROUPS> keep{},required{};
    for(int j=0;j<p.keep_size;j++)keep[l.cache.view(2)[j]]=1;
    for(int j=0;j<p.no_waste_new_count;j++)required[p.no_waste_new[j]]=1;
    std::lock_guard<std::mutex> lock(q.mutex);int count=0,extra=0;
    if(q.api){q.final_ids.clear();for(int j=0;j<p.keep_size;j++)q.final_ids.push_back(l.cache.view(2)[j]);}
    auto add=[&](int g,int source,int dest,int kind){
        if(count>=int(q.view.capacity))return -1;
        if(q.api){q.final_ids.push_back(g);q.destinations[g]=dest;++count;return 0;}
        q.view.descriptors[count]={uint32_t(g),uint32_t(l.offsets[g]/p.group_bytes),uint32_t(source),uint32_t(dest),leanstream::runtime::SourceKind(kind)};
        q.index[g]=count++;return 0;
    };
    for(int slot=0;slot<l.cache.capacity;slot++){
        int g=l.cache.view(0)[slot];if(!keep[g]&&add(g,slot,slot,0))return fail("Public cache descriptor overflow");
    }
    // consume() replaced the unsubmitted rank. Only already-admitted reads and
    // the committed final IDs can become READY from this point onward. This
    // union is exactly the old post-quiescence executable staged set.
    for(int g=0;g<p.groups;g++)if(!l.cache.cache_resident(g)){
        uint32_t state=__atomic_load_n(states+g,__ATOMIC_ACQUIRE);
        if(state==STREAM_ERROR)return fail("Public final IO error");
        if(required[g]||state==STREAM_READY||state==STREAM_INFLIGHT)
            if(add(g,g,l.cache.capacity+extra++,1))return fail("Public staged descriptor overflow");
    }
    // Importance/reduction consumers also need the current executable count.
    // Publish it BEFORE releasing the plan epoch; never leave previous-token
    // counts visible while progressive B can finish ahead of legacy sealing.
    __atomic_store_n(p.no_waste_control,count,__ATOMIC_RELEASE);
    __atomic_store_n(p.no_waste_control+1,p.keep_size+count,__ATOMIC_RELEASE);
    if(q.api){uint64_t begin_ns=ns();int rc=q.step.finalize({q.final_ids.data(),q.final_ids.size()});q.api_times[3]=double(ns()-begin_ns)/1e6;if(rc)return fail("Public API finalization failed");return 0;}
    q.queue->seal(count);q.armed=true;
    for(int idx=0;idx<count;idx++){
        auto& d=q.view.descriptors[idx];int g=d.logical_group;
        if(d.source_kind==leanstream::runtime::SourceKind::Cache||__atomic_load_n(states+g,__ATOMIC_ACQUIRE)==STREAM_READY){q.seen[g]=1;q.queue->publish(idx);}
    }
    return 0;
#else
    return fail("Public plan requires streaming IO backend");
#endif
}
static int publish_no_waste(Pipeline& p){
#ifndef STREAM_IO_V2
    return fail("No-waste requires stream_io_v2");
#else
    if(p.public_plan && (p.public_plan->progressive||p.public_plan->api) && prepare_public_plan(p))return -1;
    if(pio_wait_idle(p.io,10000))return fail("No-waste final I/O did not quiesce");
    // The public planner already built exact immutable GPU descriptors. The
    // legacy descriptor reconstruction is replay-only; bookkeeping consumes
    // the same final logical-to-destination map without constructing twice.
    if(p.public_plan&&p.public_plan->api&&!p.public_plan->verify){
        auto& q=*p.public_plan;int total=int(q.final_ids.size()),optional=total-p.keep_size;
        if(__atomic_load_n(&q.view.control->ready_count,__ATOMIC_ACQUIRE)!=uint32_t(optional))return fail("Public final queue incomplete");
        for(int g:q.final_ids)p.no_waste_exec_ids[q.destinations[g]]=g;
        p.no_waste_optional_count=optional;p.no_waste_exec_count=total;
        p.no_waste_metrics[3]=total-p.layers[p.current].cache.capacity;
        p.no_waste_metrics[4]=optional;p.no_waste_metrics[5]=total;
        p.ready_times[2]=ns();
        __atomic_store_n(p.no_waste_control,optional,__ATOMIC_RELEASE);
        __atomic_store_n(p.no_waste_control+1,total,__ATOMIC_RELEASE);
        p.ready_times[3]=ns();__atomic_store_n(p.split_ready,p.split_epoch,__ATOMIC_RELEASE);p.ready_times[4]=ns();
        return 0;
    }
    struct RestoreDesc {Pipeline& p;int32_t* saved;~RestoreDesc(){p.desc_b=saved;}} restore{p,p.desc_b};
    if(p.public_plan&&p.public_plan->api&&!p.public_plan->verify)p.desc_b=p.public_plan->reference_b.data();
    auto& l=p.layers[p.current];
    const auto* states=static_cast<const uint32_t*>(lm_pointer(p.mapped,5,0));
    std::array<uint8_t,MAX_GROUPS> keep{};
    for(int j=0;j<p.keep_size;j++){
        int g=l.cache.view(2)[j];if(g<0||g>=p.groups)return fail("No-waste KEEP invalid");
        keep[g]=1;
    }
    int optional=0,staged=0;
    if(p.no_waste_policy>=4){
        // Diagnostic modes: execute exactly strict S (M1), or the exact
        // superset S union all valid loaded groups (M2/M3). The P2 read
        // budget for these modes fetches every missing strict group.
        std::array<uint8_t,MAX_GROUPS> selected{};
        std::array<int32_t,MAX_FILL> free_dest{};int free_count=0;
        for(int slot=0;slot<p.keep_size+p.fill_size;slot++){
            int g=l.cache.view(0)[slot];
            if(keep[g])p.no_waste_exec_ids[slot]=g;
            else free_dest[free_count++]=slot;
        }
        if(free_count!=p.fill_size)return fail("Diagnostic non-KEEP cache slots invalid");
        for(int j=0;j<p.keep_size;j++)selected[l.cache.view(2)[j]]=1;
        for(int j=0;j<p.fill_size;j++){
            int g=p.fill2[j];
            if(g<0||g>=p.groups||selected[g])return fail("Diagnostic strict fill invalid");
            selected[g]=1;
            int slot=l.cache.view(1)[g];int source=0;
            if(slot<0){
                uint64_t complete=0;
                if(__atomic_load_n(states+g,__ATOMIC_ACQUIRE)!=STREAM_READY||
                   pio_peek(p.io,g,&slot,&complete)||slot!=g||!complete)
                    return fail("Diagnostic strict fill source not READY");
                source=1;
            }
            int dest=free_dest[j];
            p.no_waste_exec_ids[dest]=g;
            int32_t* d=p.desc_b+5*optional++;
            d[0]=g;d[1]=int(l.offsets[g]/p.group_bytes);d[2]=slot;d[3]=dest;d[4]=source;
        }
        if(p.no_waste_policy>=5){
            auto append=[&](int g,int slot,int source)->int{
                if(selected[g])return 0;
                if(optional>=no_waste_max_optional||staged>=no_waste_max_extra)
                    return fail("Diagnostic superset capacity exceeded");
                selected[g]=1;int dest=p.keep_size+p.fill_size+staged++;
                p.no_waste_exec_ids[dest]=g;
                int32_t* d=p.desc_b+5*optional++;
                d[0]=g;d[1]=int(l.offsets[g]/p.group_bytes);d[2]=slot;d[3]=dest;d[4]=source;
                return 0;
            };
            for(int slot=0;slot<p.keep_size+p.fill_size;slot++){
                int g=l.cache.view(0)[slot];if(append(g,slot,0))return -1;
            }
            for(int g=0;g<p.groups;g++){
                if(selected[g]||__atomic_load_n(states+g,__ATOMIC_ACQUIRE)!=STREAM_READY)continue;
                int slot=-1;uint64_t complete=0;
                if(pio_peek(p.io,g,&slot,&complete)||slot!=g||!complete)
                    return fail("Diagnostic superset staged source invalid");
                if(append(g,slot,1))return -1;
            }
        }
        p.no_waste_optional_count=optional;
        p.no_waste_exec_count=p.keep_size+optional;
        std::array<uint8_t,no_waste_max_exec> seen_dest{};
        for(int j=0;j<p.keep_size;j++){
            int32_t* d=p.desc_a+5*j;int dest=d[3];
            if(dest<0||dest>=p.no_waste_exec_count||seen_dest[dest]++||
               p.no_waste_exec_ids[dest]!=d[0])
                return fail("Diagnostic KEEP destination identity invalid");
        }
        for(int j=0;j<optional;j++){
            int32_t* d=p.desc_b+5*j;int dest=d[3];
            if(dest<0||dest>=p.no_waste_exec_count||seen_dest[dest]++||
               p.no_waste_exec_ids[dest]!=d[0])
                return fail("Diagnostic B destination identity invalid");
        }
        p.no_waste_metrics[3]=staged;p.no_waste_metrics[4]=optional;
        p.no_waste_metrics[5]=p.no_waste_exec_count;
        p.ready_times[2]=ns();
        __atomic_store_n(p.no_waste_control,optional,__ATOMIC_RELEASE);
        __atomic_store_n(p.no_waste_control+1,p.no_waste_exec_count,__ATOMIC_RELEASE);
        p.ready_times[3]=ns();
        __atomic_store_n(p.split_ready,p.split_epoch,__ATOMIC_RELEASE);
        p.ready_times[4]=ns();
        return 0;
    }
    for(int slot=0;slot<p.keep_size+p.fill_size;slot++){
        int g=l.cache.view(0)[slot];
        if(g<0||g>=p.groups)return fail("No-waste cache slot invalid");
        p.no_waste_exec_ids[slot]=g;
        if(keep[g])continue;
        int32_t* d=p.desc_b+5*optional++;
        d[0]=g;d[1]=int(l.offsets[g]/p.group_bytes);d[2]=slot;d[3]=slot;d[4]=0;
    }
    if(optional!=p.fill_size)return fail("No-waste old optional count invalid");
    for(int g=0;g<p.groups;g++){
        if(l.cache.cache_resident(g)||__atomic_load_n(states+g,__ATOMIC_ACQUIRE)!=STREAM_READY)continue;
        int slot=-1;uint64_t complete=0;
        if(pio_peek(p.io,g,&slot,&complete)||slot!=g||!complete)
            return fail("No-waste staged READY ownership invalid");
        if(optional>=no_waste_max_optional||staged>=no_waste_max_extra)
            return fail("No-waste execution capacity exceeded");
        int dest=p.keep_size+p.fill_size+staged++;
        p.no_waste_exec_ids[dest]=g;
        int32_t* d=p.desc_b+5*optional++;
        d[0]=g;d[1]=int(l.offsets[g]/p.group_bytes);d[2]=slot;d[3]=dest;d[4]=1;
    }
    for(int j=0;j<p.no_waste_new_count;j++){
        int g=p.no_waste_new[j];
        if(__atomic_load_n(states+g,__ATOMIC_ACQUIRE)!=STREAM_READY)
            return fail("No-waste final P2 extra not READY");
    }
    p.no_waste_optional_count=optional;
    p.no_waste_exec_count=p.keep_size+p.fill_size+staged;
    if(p.no_waste_exec_count>no_waste_max_exec)return fail("No-waste EXEC overflow");
    p.no_waste_metrics[3]=staged;p.no_waste_metrics[4]=optional;
    p.no_waste_metrics[5]=p.no_waste_exec_count;
    if(p.public_plan){
        auto& q=*p.public_plan;std::lock_guard<std::mutex> lock(q.mutex);
        if(!q.progressive&&!q.api){q.queue->seal(optional);q.queue->publish_all();}
        if(q.verify && (q.view.control->total_count!=uint32_t(optional)||
           std::memcmp(q.view.descriptors,p.desc_b,size_t(optional)*5*sizeof(int32_t))))
            return fail("Public final descriptors differ from production replay");
        if(q.verify && __atomic_load_n(&q.view.control->ready_count,__ATOMIC_ACQUIRE)!=uint32_t(optional))return fail("Public queue incomplete");
    }
    p.ready_times[2]=ns();
    __atomic_store_n(p.no_waste_control,optional,__ATOMIC_RELEASE);
    __atomic_store_n(p.no_waste_control+1,p.no_waste_exec_count,__ATOMIC_RELEASE);
    p.ready_times[3]=ns();
    __atomic_store_n(p.split_ready,p.split_epoch,__ATOMIC_RELEASE);
    p.ready_times[4]=ns();
    return 0;
#endif
}
static int plan_no_waste(Pipeline& p){
#ifndef STREAM_IO_V2
    return fail("No-waste requires stream_io_v2");
#else
    auto& l=p.layers[p.current];
    if(!p.no_waste_control||!p.async_control||p.execution||p.ready_ids)
        return fail("No-waste requires bound asynchronous fixed-grid execution");
    const auto* states=static_cast<const uint32_t*>(lm_pointer(p.mapped,5,0));
    const float* scores=si_scores(p.shared);
    int strict_new=0;
    for(int j=0;j<p.fill_size;j++){
        int g=p.fill2[j];
        if(!l.cache.cache_resident(g)&&
           __atomic_load_n(states+g,__ATOMIC_ACQUIRE)==STREAM_EMPTY)++strict_new;
    }
    std::array<int32_t,MAX_GROUPS> order{};int n=0;
    for(int g=0;g<p.groups;g++)if(!l.cache.cache_resident(g)&&
       __atomic_load_n(states+g,__ATOMIC_ACQUIRE)==STREAM_EMPTY)order[n++]=g;
    if(strict_new>n)return fail("No-waste strict read budget invalid");
    auto better=[&](int a,int b){return scores[a]>scores[b]||(scores[a]==scores[b]&&a<b);};
    if(strict_new&&strict_new<n)std::nth_element(order.begin(),order.begin()+strict_new,order.begin()+n,better);
    std::sort(order.begin(),order.begin()+strict_new,better);
    double total=0;for(int j=0;j<strict_new;j++)total+=std::max(0.f,scores[order[j]]);
    double cumulative=0,best=-1;int knee=0;
    for(int j=0;j<strict_new;j++){
        cumulative+=std::max(0.f,scores[order[j]]);
        double distance=total>0?cumulative/total-double(j+1)/strict_new:0;
        if(distance>best){best=distance;knee=j+1;}
    }
    int r=p.no_waste_policy==1?knee:p.no_waste_policy==2?std::min(strict_new,2*knee):strict_new;
    if(r<0||r>p.fill_size)return fail("No-waste P2 read count invalid");
    if(p.no_waste_policy>=4){
        r=0;
        for(int j=0;j<p.fill_size;j++){
            int g=p.fill2[j];
            if(!l.cache.cache_resident(g)&&__atomic_load_n(states+g,__ATOMIC_ACQUIRE)==STREAM_EMPTY)
                p.no_waste_new[r++]=g;
        }
    }else std::copy_n(order.begin(),r,p.no_waste_new.begin());
    p.no_waste_new_count=r;
    p.no_waste_metrics[0]=strict_new;p.no_waste_metrics[1]=knee;p.no_waste_metrics[2]=r;
    p.times[14]=strict_new;
    if(p.publish(p.no_waste_new.data(),r,true))return -1;
    return 0;
#endif
}
static int plan_p2(Pipeline& p){
    auto& l=p.layers[p.current];
    if(p.async_control){Stats stats{};if(pio_stats(p.io,&stats))return fail("P2 idle snapshot");p.idle_before_p2_ms=double(stats.idle_ns-p.idle_at_p1)/1e6;}
    uint64_t p2_ready_ns=uint64_t(si_ready_ms(p.shared)*1.e6);
    if(p.drain(false))return -1;
#ifdef STREAM_IO_V2
    if(p.diagnostic_trace && p.groups==7168 && p.fill_size==215 && p.io){
        auto* states=static_cast<const uint32_t*>(lm_pointer(p.mapped,5,0));
        const float* scores=si_scores(p.shared);
        int cache_count=0,ready_count=0,inflight_count=0;
        for(int g=0;g<p.groups;g++){
            p.no_waste_cache[g]=-1;
            if(l.cache.cache_resident(g)){
                p.no_waste_state[g]=1;p.no_waste_cache[g]=l.cache.view(1)[g];++cache_count;
            }else{
                uint32_t state=__atomic_load_n(states+g,__ATOMIC_ACQUIRE);
                p.no_waste_state[g]=state==STREAM_READY?2:state==STREAM_INFLIGHT?3:0;
                ready_count+=p.no_waste_state[g]==2;
                inflight_count+=p.no_waste_state[g]==3;
            }
            p.no_waste_scores[g]=scores[g];
        }
        std::copy_n(l.cache.view(2),p.keep_size,p.no_waste_keep.begin());
        std::copy_n(p.fill2.begin(),p.fill_size,p.no_waste_strict_fill.begin());
        std::copy_n(p.p0_rank.begin(),int(p.early_trace[2]),p.no_waste_p0_at_p2.begin());
        std::copy_n(p.p1_priority.begin(),p.p1_priority_count,p.no_waste_p1_at_p2.begin());
        int missing=0,strict_ready=0,strict_inflight=0;
        for(int j=0;j<p.fill_size;j++){
            int state=p.no_waste_state[p.fill2[j]];
            missing+=state!=1;strict_ready+=state==2;strict_inflight+=state==3;
        }
        p.no_waste_counts={p.keep_size,cache_count-p.keep_size,ready_count,inflight_count,
                           missing,int(p.early_trace[2]),p.p1_priority_count,p.group_bytes,strict_ready,strict_inflight};
    }
#endif
    if(p.diagnostic_trace && p.fill_size==215 && p.groups==7168){
        const float* scores=si_scores(p.shared);
        std::array<uint8_t,MAX_GROUPS> excluded{};
        for(int j=0;j<p.keep_size;j++)excluded[l.cache.view(2)[j]]=1;
        std::array<int32_t,MAX_GROUPS> order{};int count=0;
        for(int g=0;g<p.groups;g++)if(!excluded[g])order[count++]=g;
        auto better=[&](int a,int b){return scores[a]>scores[b]||(scores[a]==scores[b]&&a<b);};
        std::partial_sort(order.begin(),order.begin()+279,order.begin()+count,better);
        auto state=[&](int g){
            if(l.cache.cache_resident(g))return 2;
            int32_t slot=-1;uint64_t ready=0;
            if(pio_peek(p.io,g,&slot,&ready)==0&&ready)return 1;
            return 0;
        };
        for(int variant=0;variant<3;variant++){
            int m=variant==0?231:variant==1?247:279;
            int mandatory=variant==2?160:192;
            std::array<uint8_t,MAX_GROUPS> chosen{};
            int n=0,added=0;double strict_mass=0,chosen_mass=0;
            for(int j=0;j<p.fill_size;j++)strict_mass+=scores[order[j]];
            for(int j=0;j<mandatory;j++){chosen[order[j]]=1;++n;}
            for(int j=mandatory;j<m&&n<p.fill_size;j++)if(state(order[j])>0){chosen[order[j]]=1;++n;}
            for(int j=0;j<m&&n<p.fill_size;j++)if(!chosen[order[j]]){chosen[order[j]]=1;++n;}
            int strict_ready=0,strict_cache=0,chosen_ready=0,chosen_cache=0;
            for(int j=0;j<p.fill_size;j++){
                int s=state(order[j]);strict_ready+=s==1;strict_cache+=s==2;
                if(!chosen[order[j]])++added;
            }
            for(int j=0;j<m;j++)if(chosen[order[j]]){
                int s=state(order[j]);chosen_ready+=s==1;chosen_cache+=s==2;
                chosen_mass+=scores[order[j]];
            }
            auto* out=p.relaxed_probe.data()+variant*8;
            out[0]=m;out[1]=mandatory;out[2]=added;out[3]=strict_cache;
            out[4]=strict_ready;out[5]=chosen_cache;out[6]=chosen_ready;
            out[7]=(strict_mass-chosen_mass)/std::max(strict_mass,1e-9);
        }
    }
    if(p.relaxed_p2_m){
        double begin=ms();const float* scores=si_scores(p.shared);
        std::array<uint8_t,MAX_GROUPS> excluded{},chosen{};
        for(int j=0;j<p.keep_size;j++)excluded[l.cache.view(2)[j]]=1;
        std::array<int32_t,MAX_GROUPS> order{};int count=0;
        for(int g=0;g<p.groups;g++)if(!excluded[g])order[count++]=g;
        auto better=[&](int a,int b){return scores[a]>scores[b]||(scores[a]==scores[b]&&a<b);};
        std::nth_element(order.begin(),order.begin()+p.relaxed_p2_m,order.begin()+count,better);
        std::sort(order.begin(),order.begin()+p.relaxed_p2_m,better);
        int n=0;
        for(int j=0;j<p.relaxed_p2_p;j++){p.fill2[n++]=order[j];chosen[order[j]]=1;}
        for(int j=p.relaxed_p2_p;j<p.relaxed_p2_m&&n<p.fill_size;j++)
            if(l.cache.cache_resident(order[j])){p.fill2[n++]=order[j];chosen[order[j]]=1;}
        for(int j=0;j<p.relaxed_p2_m&&n<p.fill_size;j++)if(!chosen[order[j]]){
            p.fill2[n++]=order[j];chosen[order[j]]=1;
        }
        if(n!=p.fill_size)return fail("Relaxed P2 failed exact fill count");
        int changed=0,saved=0;double strict_mass=0,chosen_mass=0;
        for(int j=0;j<p.fill_size;j++){
            int g=order[j];strict_mass+=scores[g];
            if(!chosen[g]){++changed;if(!l.cache.cache_resident(g))++saved;}
            chosen_mass+=scores[p.fill2[j]];
        }
        p.relaxed_metrics={double(changed),(strict_mass-chosen_mass)/std::max(strict_mass,1e-9),double(saved),ms()-begin};
    }
    p.times[12]=std::count_if(p.completed_ns.begin(),p.completed_ns.end(),[&](uint64_t ns){return ns && ns<=p2_ready_ns;});
    p.stamp(2,5);
    if(p.no_waste_policy)return plan_no_waste(p);
    double start=ms();if(l.cache.plan_final(p.fill2.data()))return fail("P2 plan failed");
    if(l.cache.stage_core(si_scores(p.shared)))return fail("Stable core plan failed");
    const auto& plan=l.cache.plan();p.wanted.fill(0);
    for(int k=0;k<plan.missing_count;k++)p.wanted[plan.missing[k]]=1;
    p.times[8]=ms()-start;p.times[14]=plan.missing_count;p.stamp(2,6);
    if(p.mid_generation){
        for(int j=0;j<p.mid_count;j++)for(int h:p.fill2)if(p.fill_mid[j]==h){p.mid_times[7]++;break;}
        for(int g=0;g<p.groups;g++)if(p.completed_ns[g]&&p.completed_generation[g]==p.mid_generation){
            p.mid_times[p.wanted[g]?8:9]++;}
        p.mid_times[10]=ms()-p.mid_times[11];
    }
    for(int k=0;k<plan.missing_count;k++)p.times[13]+=p.completed_ns[plan.missing[k]] && p.completed_ns[plan.missing[k]]<=p2_ready_ns;
    for(int i=0;i<p.fill_size;i++)for(int j=0;j<p.fill_size;j++)if(p.fill1[i]==p.fill2[j]){p.times[15]++;break;}
    // Install the immutable final order BEFORE releasing slots: each release
    // wakes readers. Otherwise the old P1 queue can refill freed slots with
    // obsolete completions and exhaust capacity while B waits for its final set.
    if(p.ready_ids){
        p.final_remaining.store(p.fill_size);
        {std::lock_guard<std::mutex> lock(p.final_mutex);p.final_busy=true;p.final_job=true;}
        // Build immutable metadata before releasing final membership. No scanner descriptor packing.
        p.execution->confirm(l.cache,p.fill2.data(),l.offsets.data(),p.group_bytes,
            static_cast<int32_t*>(lm_pointer(p.mapped,4,0)),[](int,int&,uint64_t&){},false);
        p.ready_ids->require(p.fill2.data(),p.fill_size,p.ready_epoch);
        for(int j=0;j<p.fill_size;j++){int g=p.fill2[j];
            int slot=-1;uint64_t ready=0;
            if(l.cache.cache_resident(g))p.ready_ids->publish(g,p.ready_epoch,ns(),1);
            else if(!pio_peek(p.io,g,&slot,&ready)&&ready)p.ready_ids->publish(g,p.ready_epoch,ready,2);
        }
    }
    start=ms();if(p.publish(plan.missing.data(),plan.missing_count,true))return -1;
#ifdef STREAM_IO_V2
    if(p.diagnostic_trace){
        StreamProfile profile{};
        std::array<StreamUpdateProfile,256> updates{};
        if(pio_stream_profile(p.io,&profile,updates.data(),updates.size())<0)return fail("Final I/O profile unavailable");
        p.final_io_state={profile.required_count,profile.already_ready,
                          profile.inflight_required,profile.empty_required};
        if(profile.required_count!=uint64_t(plan.missing_count)||
           profile.already_ready+profile.inflight_required+profile.empty_required!=profile.required_count)
            return fail("Final I/O classification mismatch");
    }
#endif
    if(p.async_control && p.drain(true))return -1;
    p.times[7]=ms()-start;
    if(p.async_control&&pio_parallel(p.io,p.final_qd))return fail("Final I/O parallelism");
    p.stamp(2,9);
    return 0;
}
extern "C" const double* lp_relaxed_probe(void* ptr){return static_cast<Pipeline*>(ptr)->relaxed_probe.data();}
extern "C" const double* lp_relaxed_metrics(void* ptr){return static_cast<Pipeline*>(ptr)->relaxed_metrics.data();}
extern "C" const int32_t* lp_no_waste_state(void* ptr){return static_cast<Pipeline*>(ptr)->no_waste_state.data();}
extern "C" const int32_t* lp_no_waste_cache(void* ptr){return static_cast<Pipeline*>(ptr)->no_waste_cache.data();}
extern "C" const int32_t* lp_no_waste_keep(void* ptr){return static_cast<Pipeline*>(ptr)->no_waste_keep.data();}
extern "C" const int32_t* lp_no_waste_strict_fill(void* ptr){return static_cast<Pipeline*>(ptr)->no_waste_strict_fill.data();}
extern "C" const float* lp_no_waste_scores(void* ptr){return static_cast<Pipeline*>(ptr)->no_waste_scores.data();}
extern "C" const int32_t* lp_no_waste_counts(void* ptr){return static_cast<Pipeline*>(ptr)->no_waste_counts.data();}
extern "C" const int32_t* lp_no_waste_p0_rank(void* ptr){return static_cast<Pipeline*>(ptr)->no_waste_p0_at_p2.data();}
extern "C" const int32_t* lp_no_waste_p1_priority(void* ptr){return static_cast<Pipeline*>(ptr)->no_waste_p1_at_p2.data();}
extern "C" const double* lp_no_waste_metrics(void* ptr){return static_cast<Pipeline*>(ptr)->no_waste_metrics.data();}
extern "C" const int32_t* lp_no_waste_exec_ids(void* ptr){return static_cast<Pipeline*>(ptr)->no_waste_exec_ids.data();}
extern "C" const int32_t* lp_no_waste_extra_ids(void* ptr){return static_cast<Pipeline*>(ptr)->no_waste_new.data();}
extern "C" int lp_relaxed_p2_config(void* ptr,int m,int mandatory){
    auto& p=*static_cast<Pipeline*>(ptr);
    if(p.current>=0||p.io||p.async_control||p.split_ready||
       !((m==0&&mandatory==0)||(p.groups==7168&&p.fill_size==215&&
       ((m==247&&mandatory==192)||(m==279&&mandatory==160)))))
        return fail("Relaxed P2 configuration must precede streaming setup");
    p.relaxed_p2_m=m;p.relaxed_p2_p=mandatory;
    si_capture_scores(p.shared,m!=0);return 0;
}
extern "C" int lp_no_waste_config(void* ptr,int policy){
#ifndef STREAM_IO_V2
    (void)ptr;(void)policy;return fail("No-waste requires stream_io_v2");
#else
    auto& p=*static_cast<Pipeline*>(ptr);
    if(p.current>=0||p.io||p.async_control||p.split_ready||
       (policy<1||policy>6)||p.groups!=7168||p.keep_size!=1219||
       p.fill_size!=215||p.group_size!=2||p.relaxed_p2_m||p.semantic_mid_rescue||
       p.p0_budget!=32||p.lite_budget!=128||p.cheap_mid_size)
        return fail("Invalid no-waste configuration");
    if(lm_no_waste_config(p.mapped,no_waste_max_exec))return fail(lm_error());
    p.no_waste_policy=policy;
    // No-waste ranks missing groups from P2 scores even when tracing is off.
    si_capture_scores(p.shared,1);return 0;
#endif
}
extern "C" int lp_no_waste_count_bind(void* ptr,int32_t* control){
    auto& p=*static_cast<Pipeline*>(ptr);
    if(!p.no_waste_policy||!p.split_ready||!control||p.no_waste_control)
        return fail("Invalid no-waste control binding");
    p.no_waste_control=control;control[0]=control[1]=0;return 0;
}
extern "C" {
const char* lp_error(){return error;}
void lp_destroy(void* p){if(static_cast<Pipeline*>(p)->ready_driven)ready_unregister(p);delete static_cast<Pipeline*>(p);}
void* lp_create(const char* predictor){
    auto* p=new Pipeline;p->predictor_path=predictor;p->host_slot.fill(-1);p->owners.fill(-1);
    p->mapped=lm_create();if(!p->mapped){fail(lm_error());delete p;return nullptr;}
    cpu_set_t original;CPU_ZERO(&original);
    if(sched_getaffinity(0,sizeof(original),&original)){fail("Read original CPU affinity failed");delete p;return nullptr;}
    p->shared=si_create(predictor);if(!p->shared){fail(si_error());delete p;return nullptr;}
    cpu_set_t master;CPU_ZERO(&master);
    if(sched_getaffinity(0,sizeof(master),&master)){fail("Read predictor master affinity failed");delete p;return nullptr;}
    for(int cpu=0;cpu<CPU_SETSIZE;cpu++)if(CPU_ISSET(cpu,&original)&&!CPU_ISSET(cpu,&master))p->io_cpus[p->io_cpu_count++]=cpu;
    if(!p->io_cpu_count){fail("No originally permitted CPU separate from predictor master");delete p;return nullptr;}
    return p;
}
void* lp_mapped(void* p){return static_cast<Pipeline*>(p)->mapped;}
void* lp_shared(void* p){return static_cast<Pipeline*>(p)->shared;}
void* lp_cache(void* p,int l){auto& bank=*static_cast<Pipeline*>(p);if(l<0||l>=bank.layer_count)return nullptr;return &bank.layers[l].cache;}
int lp_group_config(void* ptr,int g,int stable){
    auto& p=*static_cast<Pipeline*>(ptr);
    if(g<2||g>4||p.current>=0||p.io||p.async_control||p.split_ready)return fail("Late/invalid group configuration");
    for(auto& l:p.layers)if(l.fd>=0)return fail("Configure before binding");
    p.group_size=g;p.groups=g==2?7168:g==3?4779:3584;p.keep_size=g==2?1219:g==3?812:609;
    p.fill_size=2868/g-p.keep_size;p.group_bytes=g*6144;p.stable=stable;
    for(auto& l:p.layers)if(l.cache.configure(g,stable))return fail("Group cache configuration");
    if(lm_configure(p.mapped,g)||si_configure(p.shared,g,stable))return fail("Mapped/predictor group configuration");return 0;
}
int lp_model_config(void* ptr,int h,int intermediate,int layers,int stable){
#ifndef STREAM_IO_V2
    return fail("New-model G2 requires stable stream_io_v2 staging");
#else
    auto& p=*static_cast<Pipeline*>(ptr);
    if(p.current>=0||p.io||p.async_control||p.split_ready||
       h<128||h>4096||h%128||intermediate<2||intermediate%2||
       layers<1||layers>max_layers)return fail("Late/invalid model geometry");
    for(auto& layer:p.layers)if(layer.fd>=0)return fail("Configure model before binding");
    int groups=intermediate/2;
    int active=int(std::nearbyint(.2*groups));
    int keep=int(std::nearbyint(.85*active));
    if(groups>MAX_GROUPS||active>MAX_CAPACITY||keep>MAX_KEEP||active-keep>MAX_FILL)
        return fail("Model geometry exceeds storage capacity");
    p.hidden_size=h;p.intermediate_size=intermediate;p.layer_count=layers;
    p.group_size=2;p.groups=groups;p.keep_size=keep;p.fill_size=active-keep;
    p.group_bytes=3*h;p.stable=stable!=0;p.model_configured=true;
    for(auto& layer:p.layers)if(layer.cache.configure_model(2,groups,active,keep,p.stable))
        return fail("Model cache configuration");
    if(lm_configure_model(p.mapped,h,intermediate,groups,active,p.fill_size)||
       si_configure_model(p.shared,h,groups,keep,p.fill_size,1024))
        return fail("Mapped/predictor model configuration");
    return 0;
#endif
}
int lp_predictor_stride(void* ptr,int stride){
    auto& p=*static_cast<Pipeline*>(ptr);
    if(!p.model_configured||p.current>=0||p.io||p.async_control||p.split_ready||
       stride<1||stride>1664||stride%32)return fail("Invalid predictor stride configuration");
    for(auto& layer:p.layers)if(layer.fd>=0)return fail("Configure predictor before binding");
    if(si_configure_model(p.shared,p.hidden_size,p.groups,p.keep_size,p.fill_size,stride))
        return fail(si_error());
    return 0;
}
int lp_p1_policy(void* ptr,int legacy){if(legacy!=0&&legacy!=1)return fail("Invalid P1 policy");static_cast<Pipeline*>(ptr)->legacy_p1=legacy;return 0;}
int lp_speculative_config(void* ptr,int p1_size,int mid_size){
    auto& p=*static_cast<Pipeline*>(ptr);
    if(p.current>=0||p.io||p.async_control||p.split_ready||
       (p1_size!=0&&p1_size!=512&&p1_size!=1024&&p1_size!=2048)||
       (mid_size!=0&&mid_size!=128&&mid_size!=256&&mid_size!=512)||
       (mid_size&&!p1_size))return fail("Invalid speculative subset configuration");
    p.cheap_p1_size=p1_size;p.cheap_mid_size=mid_size;return 0;
}
int lp_early_speculation_config(void* ptr,int p0_budget,int lite_budget,int lite_features){
    auto& p=*static_cast<Pipeline*>(ptr);
    if(p.current>=0||p.io||p.async_control||p.split_ready||
       (p0_budget!=0&&p0_budget!=32&&p0_budget!=64)||
       (lite_budget!=0&&lite_budget!=128&&lite_budget!=256)||
       (lite_budget&&(!p.cheap_p1_size||(p.cheap_p1_size!=1024&&p.cheap_p1_size!=512)))||
       (lite_budget&&(lite_features!=256&&lite_features!=512)))
        return fail("Invalid progressive speculation configuration");
    p.p0_budget=p0_budget;p.lite_budget=lite_budget;p.lite_features=lite_features;return 0;
}
int lp_tiny_mid_config(void* ptr,int features){
    auto& p=*static_cast<Pipeline*>(ptr);
    if(p.current>=0||p.io||p.async_control||p.split_ready||
       (features!=0&&features!=64&&features!=128)||
       (features&&(p.lite_budget!=128||p.cheap_mid_size!=128)))
        return fail("Invalid tiny residual Mid configuration");
    p.tiny_mid_features=features;
    if(features){
        p.tiny_mid_shared=si_create(p.predictor_path.c_str());
        if(!p.tiny_mid_shared||si_configure_model(p.tiny_mid_shared,p.hidden_size,p.groups,
                p.keep_size,p.fill_size,features)||si_set_active_features(p.tiny_mid_shared,features))
            return fail("Tiny residual Mid predictor initialization failed");
    }
    return 0;
}
int lp_semantic_mid_config(void* ptr,int rescue){
    auto& p=*static_cast<Pipeline*>(ptr);
    if(p.current>=0||p.io||p.async_control||p.split_ready||
       (rescue!=0&&rescue!=16)||(rescue&&p.lite_budget!=128))
        return fail("Invalid semantic Mid rescue configuration");
    p.semantic_mid_rescue=rescue;return 0;
}
const uint64_t* lp_tiny_mid_metrics(void* ptr){return static_cast<Pipeline*>(ptr)->tiny_mid_metrics.data();}
int lp_lite_bind(void* ptr,int layer,const int32_t* features,const void* weights,const float* scales){
    auto& p=*static_cast<Pipeline*>(ptr);
    if(layer<0||layer>=p.layer_count||p.current>=0||!p.lite_budget||!features||!weights||!scales)
        return fail("Invalid compact P1 binding");
    if(!p.lite_shared){
        p.lite_shared=si_create(p.predictor_path.c_str());
        if(!p.lite_shared||si_configure_model(p.lite_shared,p.hidden_size,p.groups,p.keep_size,
                                              p.fill_size,p.lite_features)||
           si_set_active_features(p.lite_shared,p.lite_features))
            return fail("Compact P1 predictor initialization failed");
    }
    auto& l=p.layers[layer];l.lite_features=features;l.lite_weights=weights;l.lite_scales=scales;
    return 0;
}
const uint64_t* lp_early_trace(void* ptr){return static_cast<Pipeline*>(ptr)->early_trace.data();}
int lp_speculative_bind(void* ptr,int layer,const int32_t* hot,int count){
    auto& p=*static_cast<Pipeline*>(ptr);
    if(layer<0||layer>=p.layer_count||p.current>=0||!p.cheap_p1_size||!hot||count!=p.groups)
        return fail("Invalid speculative hot-list binding");
    std::array<uint8_t,MAX_GROUPS> seen{};
    for(int i=0;i<count;i++){int g=hot[i];if(g<0||g>=p.groups||seen[g])return fail("Invalid hot-list ID");seen[g]=1;}
    p.layers[layer].speculative_hot=hot;p.layers[layer].speculative_hot_count=count;return 0;
}
const uint64_t* lp_ready_times(void* p){return static_cast<Pipeline*>(p)->ready_times.data();}
const double* lp_times(void* p){return static_cast<Pipeline*>(p)->times.data();}
const int32_t* lp_fill(void* p,int which){auto& s=*static_cast<Pipeline*>(p);return which?s.fill2.data():s.fill1.data();}
int lp_bind(void* ptr,int i,int fd,const int64_t* offsets,const int32_t* features,const void* weights,
            const float* scales,const void* device_scales,void* rows){
    auto& p=*static_cast<Pipeline*>(ptr);if(i<0||i>=p.layer_count||fd<0)return fail("Invalid layer binding");
    auto& l=p.layers[i];if(l.fd>=0)return fail("Layer already bound");
    for(int g=0;g<p.groups;g++)if(offsets[g]<0||offsets[g]%p.group_bytes)return fail("Invalid group offset");
    l.fd=::dup(fd);if(l.fd<0)return fail("dup failed");std::copy(offsets,offsets+p.groups,l.offsets.begin());
    l.features=features;l.weights=weights;l.scales=scales;l.device_scales=device_scales;l.rows=rows;return 0;
}
int lp_bind_projection(void* ptr,int i,const float* projection,int stride){
    auto& p=*static_cast<Pipeline*>(ptr);
    if(i<0||i>=p.layer_count||p.current>=0||!p.model_configured||!projection||
       stride<1||stride%32)return fail("Invalid predictor projection binding");
    p.layers[i].projection=projection;p.layers[i].projection_stride=stride;return 0;
}
extern "C" int lp_mid_partial_bind(void* ptr,int i,const int32_t* features,
                                    const void* weights,const float* scales){
    auto& p=*static_cast<Pipeline*>(ptr);
    if(i<0||i>=p.layer_count||p.current>=0||!p.model_configured||
       !features||!weights||!scales)return fail("Invalid midpoint predictor binding");
    auto& l=p.layers[i];l.mid_features=features;l.mid_weights=weights;l.mid_scales=scales;
    return 0;
}
// Initialization-only opt-in. Reopen, rather than F_SETFL on dup(), so the
// buffered prefill reader retains its original open-file-description flags.
int lp_direct(void* ptr,int alignment){
    auto& p=*static_cast<Pipeline*>(ptr);
    if(p.current>=0||alignment<=0||(alignment&(alignment-1)))return fail("Late/invalid direct I/O configuration");
    if(reinterpret_cast<uintptr_t>(lm_pointer(p.mapped,3,0))%alignment||p.group_bytes%alignment)
        return fail("Mapped staging does not satisfy verified direct alignment");
    if(p.io){
        Stats stats{};if(pio_stats(p.io,&stats)||stats.inflight||stats.queued||stats.owned)return fail("Direct I/O requires drained ownership");
    }
    std::array<int,max_layers> descriptors;descriptors.fill(-1);
    for(int i=0;i<p.layer_count;i++)if(p.layers[i].fd>=0){
        for(int g=0;g<p.groups;g++)if(p.layers[i].offsets[g]%alignment)return fail("Unaligned direct offset");
    }
    for(int i=0;i<p.layer_count;i++)if(p.layers[i].fd>=0){
        char path[64];std::snprintf(path,sizeof(path),"/proc/self/fd/%d",p.layers[i].fd);
        descriptors[i]=::open(path,O_RDONLY|O_DIRECT|O_CLOEXEC);
        if(descriptors[i]<0){for(int fd:descriptors)if(fd>=0)::close(fd);return fail("O_DIRECT reopen failed");}
    }
    if(p.io){pio_close(p.io);p.io=nullptr;}
    for(int i=0;i<p.layer_count;i++)if(descriptors[i]>=0){::close(p.layers[i].fd);p.layers[i].fd=descriptors[i];}
    return 0;
}
int lp_reset(void* ptr){
    auto& p=*static_cast<Pipeline*>(ptr);if(p.finish_io())return -1;
    p.io_prebound=false;p.p1_prepared=false;p.p0_count=0;p.early_trace.fill(0);
    for(auto& l:p.layers)if(l.cache.reset())return fail("Cache not idle at reset");if(p.public_plan)p.public_plan->prepared=false;p.current=-1;return 0;
}
static int prepare_p1(Pipeline& p,int i,int hidden){
    if(i<0||i>=p.layer_count||hidden<0||hidden>1)return fail("Invalid P1 layer");
    auto& l=p.layers[i];if(l.fd<0)return fail("Unbound layer");
    if(p.model_configured && l.projection && si_set_projection(p.shared,l.projection,l.projection_stride))return fail(si_error());
    // Previous layer importance acquire guarantees GPU staging consumption; finish_io released owners.
    p.ready_armed.store(false,std::memory_order_release);
    if(p.execution)p.execution->disarm();
    if(p.ready_ids)p.ready_ids->begin(++p.ready_epoch);
    p.priority_trace.fill(0);p.mid_times.fill(0);p.mid_trace.fill(0);p.mid_generation=0;p.mid_count=0;
    p.p1_priority_count=0;p.mid_pending_count=0;p.tiny_mid_metrics.fill(0);p.semantic_mid_metrics.fill(0);
    p.no_waste_new_count=p.no_waste_optional_count=p.no_waste_exec_count=0;
    p.no_waste_maintenance_stream=p.no_waste_packed_cache=nullptr;p.no_waste_metrics.fill(0);
    p.early_trace.fill(0);
    if(p.ready_driven)for(auto& trace:p.ready_trace)trace.fill(0);
    p.times.fill(0);p.current=i;p.p2_published=0;p.completed_ns.fill(0);
    if(p.diagnostic_trace)p.p1_io_begin_ns=ns();
    if(!p.io){
#ifdef STREAM_IO_V2
        p.io=pio_stream_create(l.fd,l.offsets.data(),p.groups,p.group_bytes,lm_pointer(p.mapped,3,0),p.group_bytes,lm_pointer(p.mapped,5,0));
#else
        p.io=pio_create(l.fd,l.offsets.data(),p.groups,p.group_bytes,lm_pointer(p.mapped,3,0),256,p.group_bytes,1);
#endif
        if(p.io && p.async_control && pio_enable_workers(p.io,p.pool_qd))return fail("Final I/O worker initialization");
        if(p.io && p.diagnostic_trace && pio_log_enable(p.io))return fail("I/O diagnostic log initialization");
        if(p.io && p.async_control && pio_observe(p.io,final_completion,&p))return fail("I/O observer registration");
        if(p.io && pio_worker_cpus(p.io,p.io_cpus.data(),p.io_cpu_count))return fail("I/O worker affinity failed");
    }
    else if(!p.io_prebound && pio_rebind(p.io,l.fd,l.offsets.data()))return fail("Arena still owned at layer boundary");
    p.io_prebound=false;
    if(!p.io)return fail("I/O worker creation failed");
    if(p.pre_qd>1&&pio_parallel(p.io,p.pre_qd))return fail("Initial I/O window");
    if(p.diagnostic_trace)p.p1_io_end_ns=ns();
    p.p1_prepared=true;return 0;
}
static int build_p0_rank(Pipeline& p){
    if(!p.p0_budget)return 0;
    auto& l=p.layers[p.current];
    if(!l.speculative_hot||l.speculative_hot_count!=p.groups)return fail("P0 hot order unavailable");
    p.spec_marks.fill(0);
    int count=0;
    auto add=[&](int g){
        if(g<0||g>=p.groups||p.spec_marks[g]||l.cache.cache_resident(g))return;
        p.spec_marks[g]=1;
        if(count<p.p0_budget)p.p0_rank[count++]=g;
    };
    const int32_t* previous=l.cache.view(3);
    for(int j=0;j<p.keep_size+p.fill_size&&count<p.p0_budget;j++)add(previous[j]);
    for(int j=0;j<l.speculative_hot_count&&count<p.p0_budget;j++)add(l.speculative_hot[j]);
    p.p0_count=count;
    return 0;
}
int lp_p0_start(void* ptr,int layer){
    auto& p=*static_cast<Pipeline*>(ptr);
    if(!p.p0_budget||!p.ready_driven||p.maintenance_pending.load(std::memory_order_acquire))
        return fail("P0 ownership/READY scheduler violation");
    if(prepare_p1(p,layer,0))return -1;
    if(p.p0_count){
        if(p.publish(p.p0_rank.data(),p.p0_count))return -1;
        p.early_trace[0]=ns();p.early_trace[2]=p.p0_count;p.early_trace[4]=p.generation;
    }
    p.p0_count=0;return 0;
}
int lp_p1(void* ptr,int i,int hidden){
    auto& p=*static_cast<Pipeline*>(ptr);
    if(!p.p1_prepared && prepare_p1(p,i,hidden))return -1;
    p.p1_prepared=false;
    auto& l=p.layers[i];p.stamp(1,2);
    if(p.ready_driven){
        if(__atomic_load_n(lm_ready_flag(p.mapped,hidden),__ATOMIC_ACQUIRE)!=p.p1_epoch)
            return fail("P1 worker received non-READY payload");
    }else if(lm_wait(p.mapped,hidden))return fail(lm_error());
    p.stamp(1,3);
    if(p.async_control&&p.speculate){
        if(si_predict_ranked(p.shared,static_cast<const uint16_t*>(lm_pointer(p.mapped,hidden,0)),l.features,l.weights,l.scales,
                            l.cache.view(2),p.fill1.data(),p.times.data(),p.continuation.data()))return fail(si_error());
        double start=ms();int count=0;
        if(l.cache.prefetch_internal(p.fill1.data()))return fail("Invalid speculative P1 state");
        for(int g:p.continuation)if(!l.cache.cache_resident(g))p.speculative[count++]=g;
        p.times[20]=ms()-start;p.times[11]=count;
        double t=ms();if(p.publish(p.speculative.data(),count))return -1;p.times[22]=ms()-t;p.times[3]=ms()-start;
        Stats stats{};if(pio_stats(p.io,&stats))return fail("P1 idle snapshot");p.idle_at_p1=stats.idle_ns;
        if(p.ready_driven)p.continuation_token.store(p.submitted_token.load(),std::memory_order_release);
        return 0;
    }
    int ranked=0;
    if(p.cheap_p1_size){
        if(!l.speculative_hot||l.speculative_hot_count!=p.groups)return fail("Missing speculative hot list");
        p.spec_marks.fill(0);
        for(int j=0;j<p.keep_size;j++)p.spec_marks[l.cache.view(2)[j]]=1;
        int count=0;
        auto add=[&](int g){if(g>=0&&g<p.groups&&!p.spec_marks[g]&&count<p.cheap_p1_size){
            p.spec_marks[g]=1;p.spec_pool[count++]=g;}};
        const int32_t* previous=l.cache.view(3);
        for(int j=p.keep_size;j<p.keep_size+p.fill_size;j++)add(previous[j]);
        for(int j=0;j<l.speculative_hot_count&&count<p.cheap_p1_size;j++)add(l.speculative_hot[j]);
        if(count!=p.cheap_p1_size)return fail("Speculative candidate pool underfilled");
        ranked=std::min(count,int(p.continuation.size()));
        if(p.lite_budget&&(!p.lite_shared||!l.lite_features||!l.lite_weights||!l.lite_scales))
            return fail("Missing compact P1 predictor artifact");
        if(si_predict_subset(p.lite_budget?p.lite_shared:p.shared,
            static_cast<const uint16_t*>(lm_pointer(p.mapped,hidden,0)),
            p.lite_budget?l.lite_features:l.features,
            p.lite_budget?l.lite_weights:l.weights,
            p.lite_budget?l.lite_scales:l.scales,p.spec_pool.data(),count,ranked,
            p.continuation.data(),p.times.data()))return fail(si_error());
        std::copy_n(p.continuation.data(),p.fill_size,p.fill1.data());
    }else if(si_predict_ready(p.shared,static_cast<const uint16_t*>(lm_pointer(p.mapped,hidden,0)),l.features,l.weights,l.scales,
                        l.cache.view(2),p.fill1.data(),p.times.data()))return fail(si_error());
    p.stamp(1,4);if(p.diagnostic_trace){p.control_trace[13]=uint64_t(si_predict_start_ms(p.shared)*1.e6);p.control_trace[14]=uint64_t(si_predict_end_ms(p.shared)*1.e6);}
    double start=ms();
    int rc=p.legacy_p1?l.cache.prefetch(p.fill1.data()):l.cache.prefetch_internal(p.fill1.data());
    if(rc)return fail("P1 invalid fill");
    p.stamp(1,5);
    double filtered=ms();p.times[20]=filtered-start;p.times[11]=l.cache.prefetch_count();
    Stats stats{};
    if(p.legacy_p1){double t=ms();if(pio_stats(p.io,&stats))return fail("P1 stats failed");p.times[21]=ms()-t;}
    double t=ms();
    if(p.lite_budget){
        int count=0;
        for(int j=0;j<ranked&&count<p.lite_budget;j++){
            int g=p.continuation[j];
            if(!l.cache.cache_resident(g))p.speculative[count++]=g;
        }
        p.early_trace[3]=count;
        if(p.tiny_mid_features||p.semantic_mid_rescue||p.diagnostic_trace){
            if(count>int(p.p1_priority.size()))return fail("Tiny Mid P1 priority overflow");
            p.p1_priority_count=count;
            std::copy_n(p.speculative.data(),count,p.p1_priority.data());
        }
        if(p.publish(p.speculative.data(),count))return -1;
    }else if(p.publish(l.cache.view(7),l.cache.prefetch_count()))return -1;
    p.early_trace[1]=ns();p.early_trace[5]=p.generation;
    p.times[22]=ms()-t;
    // Publication installs the order atomically. Attention may start while the worker is waking.
    // P2 is allowed to supersede every unstarted P1 request; only an active read must finish.
    if(p.legacy_p1 && l.cache.prefetch_count()){
        t=ms();if(pio_wait_started(p.io,stats.calls,10000))return fail("P1 worker did not start");p.times[23]=ms()-t;
    }
    p.times[3]=ms()-start;
    if(p.async_control){Stats stats{};if(pio_stats(p.io,&stats))return fail("P1 idle snapshot");p.idle_at_p1=stats.idle_ns;}
    if(p.ready_driven)p.continuation_token.store(p.submitted_token.load(),std::memory_order_release);
    return 0;
}
int lp_p2(void* ptr,const void* input,const void* norm,float eps,void* stream){
    auto& p=*static_cast<Pipeline*>(ptr);if(p.current<0)return fail("P2 without P1");auto& l=p.layers[p.current];
    double t[5];
    if(si_run(p.shared,1,input,norm,eps,stream,l.features,l.weights,l.scales,l.cache.view(2),p.fill2.data(),t,nullptr))return fail(si_error());
    p.times[4]=t[1];p.times[5]=t[2];p.times[6]=t[3];p.times[18]=t[0];
    if(plan_p2(p))return -1;
    return await_final(p);
}
int lp_update(void* ptr,void* stream){
    auto& p=*static_cast<Pipeline*>(ptr);auto& l=p.layers[p.current];const auto& plan=l.cache.plan();
    if(lm_dequant(p.mapped,plan.missing_count,l.device_scales,l.rows,stream))return fail(lm_error());
    if(l.cache.mark_launched(plan.ticket)||l.cache.commit(plan.ticket))return fail("Cache commit failed");return 0;
}
int lp_keep(void* ptr){
    auto& p=*static_cast<Pipeline*>(ptr);auto& l=p.layers[p.current];
    if(lm_wait(p.mapped,2))return fail(lm_error());double start=ms();
    if(lm_keep(p.mapped,&l.cache))return fail("Native keep failed");p.times[10]=ms()-start;
    // No staging slot may be released until GPU importance publication after MLP has completed.
    return p.finish_io();
}
int lp_public_bind(void* ptr,void* desc,uint32_t* indices,void* control,int capacity,int progressive){
    auto& p=*static_cast<Pipeline*>(ptr);
    if(!desc||!indices||!control||capacity<no_waste_max_optional||p.public_plan||p.execution||p.ready_ids||p.no_waste_policy>3)return fail("Invalid public binding");
    p.public_plan=std::make_unique<PublicPlan>();auto& q=*p.public_plan;q.progressive=progressive;
    q.view={static_cast<leanstream::runtime::ExecDesc*>(desc),indices,static_cast<leanstream::runtime::QueueControl*>(control),uint32_t(capacity)};
    q.queue=std::make_unique<leanstream::runtime::ExecutionQueue>(q.view);
    q.compat_ready=p.split_ready;q.layout.resize(p.groups);q.cache_snapshot.resize(p.groups);q.destinations.resize(p.groups);
    q.final_ids.reserve(p.groups);q.reference_b.resize(no_waste_max_optional*5);
    leanstream::runtime::ManagerConfig config{uint32_t(p.groups),uint32_t(p.pool_qd),uint64_t(p.group_bytes),uint64_t(p.group_bytes),512,
        q.progressive?leanstream::runtime::ConsumerMode::Progressive:leanstream::runtime::ConsumerMode::BatchCompat};
    config.io_backend={&p,public_io_begin,public_io_update,public_io_consume,public_io_retire,public_io_slot,nullptr,nullptr};
    q.api=std::make_unique<leanstream::runtime::Runtime>(config,leanstream::runtime::ComputeBackend{&p,public_enqueue,public_complete,public_cache_commit,nullptr,public_cancel});
    return 0;
}
int lp_public_compute_bind(void* ptr,int (*callback)(void*),void* context,void* event){
    auto& q=*static_cast<Pipeline*>(ptr)->public_plan;q.compute=callback;q.compute_context=context;q.done_event=static_cast<cudaEvent_t>(event);return 0;
}
int lp_public_cuda_bind(void* ptr,int stage,void* function,uint32_t grid,uint32_t warps,uint32_t shared,
                        const uint64_t* values,int count,int epoch,int claim,uint32_t initial_claim,void* done){
    if(stage<0||stage>2||!function||!grid||!warps||count<0||count>32||epoch<0||epoch>=count||claim>=count)return EINVAL;
    auto& q=*static_cast<Pipeline*>(ptr)->public_plan;
    int rc=q.gpu.bind(leanstream::runtime::kTritonDefault.id,uint32_t(stage),static_cast<CUfunction>(function),
        grid,warps,shared,values,count,epoch,claim);if(rc)return rc;
    q.gpu.seed_claim(initial_claim);q.gpu.b_done=static_cast<cudaEvent_t>(done);
    q.bound_stages|=1u<<stage;q.cuda_bound=q.bound_stages==7&&q.gpu.b_done;return 0;
}
int lp_public_reduce_bind(void* ptr,uint32_t* done,int count){
    auto& q=*static_cast<Pipeline*>(ptr)->public_plan;if(!done||count<=0)return EINVAL;q.reduction_done=done;q.reduction_programs=count;return 0;
}
int lp_public_enqueue(void* ptr,const void* input,void* output,void* stream){auto& p=*static_cast<Pipeline*>(ptr);int rc=public_begin(p);if(rc)return rc;uint64_t start=ns();p.public_plan->compute_stream=static_cast<cudaStream_t>(stream);p.public_plan->compute_error=false;rc=p.public_plan->step.enqueue_compute(input,output,stream);p.public_plan->compute_error=rc!=0;p.public_plan->api_times[2]=double(ns()-start)/1e6;return rc;}
const double* lp_public_times(void* ptr){return static_cast<Pipeline*>(ptr)->public_plan->api_times;}
int lp_split_bind(void* ptr,int32_t* a,int32_t* b,uint32_t* ready){
    auto& p=*static_cast<Pipeline*>(ptr);if(!a||!b||!ready||p.final_worker.joinable())return fail("Invalid split binding");
    p.desc_a=a;p.desc_b=b;p.split_ready=ready;
    p.final_worker=std::thread([&p]{
        cpu_set_t allowed;CPU_ZERO(&allowed);
        for(int i=0;i<p.io_cpu_count;i++)CPU_SET(p.io_cpus[i],&allowed);
        sched_setaffinity(0,sizeof(allowed),&allowed);
        for(;;){
            {std::unique_lock<std::mutex> lock(p.final_mutex);
             p.final_cv.wait(lock,[&]{return p.final_job||p.final_stop;});
             if(p.final_stop)return;p.final_job=false;}
            int rc=0;
            if(p.no_waste_policy){
#ifdef LEANSTREAM_NVTX_PROFILE
                char range[96];std::snprintf(range,sizeof(range),"scheduling final layer=%d",int(p.current));
                LeanstreamNvtxScope trace(range);
#endif
                rc=publish_no_waste(p);
            }else if(p.async_control){
                std::unique_lock<std::mutex> lock(p.final_mutex);p.ready_times[1]=ns();
                if(!p.final_cv.wait_for(lock,std::chrono::seconds(10),[&]{return p.final_remaining.load(std::memory_order_acquire)==0||!p.final_error.empty();})){
                    Stats stats{};pio_stats(p.io,&stats);
                    std::fprintf(stderr,"Final timeout layer=%d epoch=%u remaining=%d armed=%d calls=%llu active=%llu owned=%llu queued=%llu\n",int(p.current),p.split_epoch,p.final_remaining.load(),int(p.ready_armed.load()),(unsigned long long)stats.calls,(unsigned long long)stats.inflight,(unsigned long long)stats.owned,(unsigned long long)stats.queued);
                    const auto& plan=p.layers[p.current].cache.plan();
                    for(int j=0;j<plan.missing_count;j++)if(!p.final_seen[j].load()){
                        int32_t slot;uint64_t done;pio_peek(p.io,plan.missing[j],&slot,&done);
                        std::fprintf(stderr,"Unseen j=%d group=%d slot=%d complete=%llu\n",j,plan.missing[j],slot,(unsigned long long)done);
                    }
                    rc=fail("Completion-driven final timeout");
                }
                if(!p.final_error.empty())rc=fail(p.final_error.c_str());
            }else rc=publish_final(p);
            // Accounting may follow readiness, overlapped with B compute.
            if(!rc){if(pio_wait_idle(p.io,10000))rc=fail("Final accounting quiescence");else rc=p.drain(true);}
            {std::lock_guard<std::mutex> lock(p.final_mutex);
             if(rc)p.final_error=error;

             p.final_busy=false;p.final_cv.notify_all();}
        }
    });return 0;
}
uint32_t lp_split_start(void* ptr,const void* input,const void* norm,float eps,void* stream){
    auto& p=*static_cast<Pipeline*>(ptr);
    if(p.current<0||!p.split_ready||p.split_epoch==UINT32_MAX){fail("Invalid split start");return 0;}
    {std::lock_guard<std::mutex> lock(p.final_mutex);if(p.final_busy||!p.final_error.empty()){fail("Previous split outstanding");return 0;}}
    auto& l=p.layers[p.current];
    if(!p.prebuilt_a){
    for(int k=0;k<p.keep_size;k++){
        int g=l.cache.view(2)[k],slot=l.cache.view(1)[g];
        if(slot<0){fail("Keep not resident");return 0;}
        int32_t* d=p.desc_a+5*k;d[0]=g;d[1]=int(l.offsets[g]/p.group_bytes);d[2]=slot;d[3]=slot;d[4]=0;
    }
    }
    if(si_start_norm(p.shared,input,norm,eps,stream)){fail(si_error());return 0;}
    ++p.split_epoch;if(p.public_plan)p.public_plan->reset(p.split_epoch);if(p.execution)p.execution->start(p.split_epoch);
    return p.split_epoch;
}
extern uint32_t si_reserve_norm(void*);
uint32_t lp_split_start_prepared(void* ptr){
 auto& p=*static_cast<Pipeline*>(ptr);if(p.current<0||!p.split_ready||p.split_epoch==UINT32_MAX){fail("Invalid prepared split start");return 0;}
 {std::lock_guard<std::mutex> lock(p.final_mutex);if(p.final_busy||!p.final_error.empty()){fail("Previous split outstanding");return 0;}}
 auto& l=p.layers[p.current];if(!p.prebuilt_a)for(int k=0;k<p.keep_size;k++){int g=l.cache.view(2)[k],slot=l.cache.view(1)[g];if(slot<0)return 0;int32_t* d=p.desc_a+5*k;d[0]=g;d[1]=l.offsets[g]/p.group_bytes;d[2]=d[3]=slot;d[4]=0;}
 if(!si_reserve_norm(p.shared)){fail(si_error());return 0;}++p.split_epoch;if(p.public_plan)p.public_plan->reset(p.split_epoch);if(p.execution)p.execution->start(p.split_epoch);return p.split_epoch;
}
int lp_split_p2(void* ptr){
    auto& p=*static_cast<Pipeline*>(ptr);auto& l=p.layers[p.current];double t[4];
    p.stamp(2,2);
    int rc=p.ready_driven?
        si_finish_norm_ready(p.shared,l.features,l.weights,l.scales,l.cache.view(2),p.fill2.data(),t):
        si_finish_norm(p.shared,l.features,l.weights,l.scales,l.cache.view(2),p.fill2.data(),t);
    if(rc)return fail(si_error());
    p.stamp(2,4);if(p.diagnostic_trace)p.control_trace[19]=uint64_t(si_ready_ms(p.shared)*1.e6);
    p.times[18]=t[0];p.times[4]=t[1];p.times[5]=t[2];p.times[6]=t[3];
    if(plan_p2(p))return -1;
    if(p.no_waste_policy){
        p.ready_times.fill(0);
        {std::lock_guard<std::mutex> lock(p.final_mutex);p.final_busy=true;p.final_job=true;}
        p.stamp(2,10);p.stamp(2,11);p.final_cv.notify_all();
        return 0;
    }
    if(p.diagnostic_trace&&p.tiny_mid_features&&p.mid_pending_count){
        const auto& plan=l.cache.plan();
        p.tiny_mid_metrics[0]=p.mid_pending_count;
        for(int j=0;j<3;j++){
            int count=std::min(p.mid_pending_count,16<<j);
            uint64_t before=0,after=0;
            for(int k=0;k<count;k++){
                before+=std::find(plan.missing.data(),plan.missing.data()+plan.missing_count,p.mid_before[k])!=plan.missing.data()+plan.missing_count;
                after+=std::find(plan.missing.data(),plan.missing.data()+plan.missing_count,p.mid_after[k])!=plan.missing.data()+plan.missing_count;
            }
            p.tiny_mid_metrics[1+j]=before;p.tiny_mid_metrics[4+j]=after;
        }
    }
    if(p.ready_ids){p.stamp(2,10);p.stamp(2,11);std::lock_guard<std::mutex> lock(p.final_mutex);p.final_cv.notify_all();return 0;}
    if(p.execution){
        p.final_remaining.store(p.fill_size);
        {std::lock_guard<std::mutex> lock(p.final_mutex);p.final_busy=true;p.final_job=true;}
        bool done=p.execution->confirm(l.cache,p.fill2.data(),l.offsets.data(),p.group_bytes,
            static_cast<int32_t*>(lm_pointer(p.mapped,4,0)),[&](int g,int& slot,uint64_t& ready){
                if(pio_peek(p.io,g,&slot,&ready))ready=0;
            });
        if(done)p.final_remaining.store(0,std::memory_order_release);
        p.stamp(2,10);p.stamp(2,11);
        {std::lock_guard<std::mutex> lock(p.final_mutex);p.final_cv.notify_all();}
        return 0;
    }
    p.ready_times.fill(0);if(prepare_b(p))return -1;
    p.stamp(2,10);
    if(p.async_control){
        // Set count before waking the waiter. Observer is armed only after all metadata is immutable.
        p.final_remaining.store(p.layers[p.current].cache.plan().missing_count);
        {std::lock_guard<std::mutex> lock(p.final_mutex);p.final_busy=true;p.final_job=true;}
        if(arm_final(p))return -1;
        p.stamp(2,11);
        {std::lock_guard<std::mutex> lock(p.final_mutex);p.final_cv.notify_all();}
    }else {std::lock_guard<std::mutex> lock(p.final_mutex);p.final_busy=true;p.final_job=true;p.final_cv.notify_one();}
    return 0;
}
int lp_split_update(void* ptr,void* packed_cache,void* stream){
    auto& p=*static_cast<Pipeline*>(ptr);
    if(p.no_waste_policy){
        p.no_waste_packed_cache=packed_cache;p.no_waste_maintenance_stream=stream;
        return 0;
    }
    // Enqueued AFTER B on the same stream; B's acquire guarantees metadata validity.
    if(lm_packed_update(p.mapped,p.async_control?p.fill_size:p.layers[p.current].cache.plan().missing_count,packed_cache,stream))return fail(lm_error());
    // Native commit must not race the worker reading the plan/cache for descriptors.
    // Defer commit until GPU completion in lp_split_keep.
    return 0;
}
static int no_waste_late_retain(Pipeline& p){
    auto& l=p.layers[p.current];
    if(!p.no_waste_maintenance_stream||!p.no_waste_packed_cache||
       p.no_waste_exec_count<l.cache.capacity||p.no_waste_exec_count>no_waste_max_exec||
       l.cache.phase()!=CacheState::IDLE)
        return fail("No-waste maintenance state invalid");
    double begin=ms();
    auto* importance=static_cast<float*>(lm_pointer(p.mapped,2,0));
    std::array<float,MAX_GROUPS> by_group{};
    std::array<int32_t,no_waste_max_exec> rank{};
    std::array<uint8_t,MAX_GROUPS> seen{};
    for(int i=0;i<p.no_waste_exec_count;i++){
        int g=p.no_waste_exec_ids[i];float value=importance[i];
        if(g<0||g>=p.groups||seen[g]||!std::isfinite(value)||value<0)
            return fail("No-waste actual-importance identity invalid");
        seen[g]=1;by_group[g]=value;rank[i]=i;
    }
    auto better=[&](int a,int b){
        float x=importance[a],y=importance[b];
        return x>y||(x==y&&p.no_waste_exec_ids[a]<p.no_waste_exec_ids[b]);
    };
    std::array<int32_t,MAX_CAPACITY> retained{};
    if(p.no_waste_policy==4||p.no_waste_policy==5){
        for(int i=0;i<p.keep_size;i++)retained[i]=l.cache.view(2)[i];
        for(int i=0;i<p.fill_size;i++)retained[p.keep_size+i]=p.fill2[i];
    }else{
        std::partial_sort(rank.begin(),rank.begin()+l.cache.capacity,
                          rank.begin()+p.no_waste_exec_count,better);
        for(int i=0;i<l.cache.capacity;i++)retained[i]=p.no_waste_exec_ids[rank[i]];
    }
    if(l.cache.plan_selected(retained.data()))return fail("No-waste late cache plan failed");
    const auto& plan=l.cache.plan();
    auto* meta=static_cast<int32_t*>(lm_pointer(p.mapped,4,0));
    for(int j=0;j<plan.missing_count;j++){
        int g=plan.missing[j];int slot=-1;uint64_t complete=0;
        if(pio_peek(p.io,g,&slot,&complete)||slot!=g||!complete)
            return fail("No-waste retained staging identity invalid");
        meta[3*j]=slot;meta[3*j+1]=int(l.offsets[g]/p.group_bytes);
        meta[3*j+2]=plan.destinations[j];
    }
    if(lm_packed_update(p.mapped,plan.missing_count,p.no_waste_packed_cache,
                        p.no_waste_maintenance_stream)||
       lm_stream_synchronize(p.no_waste_maintenance_stream))
        return fail(lm_error());
    if(l.cache.mark_launched(plan.ticket)||l.cache.commit(plan.ticket))
        return fail("No-waste late cache commit failed");
    for(int slot=0;slot<l.cache.capacity;slot++){
        int g=l.cache.view(0)[slot];importance[slot]=by_group[g];
    }
    if(lm_keep(p.mapped,&l.cache))return fail("No-waste KEEP update failed");
    p.no_waste_metrics[6]=plan.missing_count;
    p.no_waste_metrics[7]=ms()-begin;
    std::array<uint8_t,MAX_GROUPS> strict{};
    for(int j=0;j<p.fill_size;j++)strict[p.fill2[j]]=1;
    for(int g=0;g<p.groups;g++)if(p.completed_ns[g]){
        ++p.no_waste_metrics[8];
        if(seen[g])++p.no_waste_metrics[9];
        else ++p.no_waste_metrics[11];
        if(strict[g])++p.no_waste_metrics[10];
    }
    if(p.no_waste_policy<=3 && p.no_waste_metrics[8]!=p.no_waste_metrics[9])
        return fail("Completed no-waste payload was not executed");
    if(p.public_plan&&p.public_plan->api&&public_commit(p,retained.data(),l.cache.capacity))return fail("Public API commit failed");
    int rc=p.finish_io();if(rc)return rc;
    if(p.public_plan&&p.public_plan->api&&public_prepare(p))return fail("Public background metadata preparation failed");
    return 0;
}
int lp_split_keep(void* ptr){
    auto& p=*static_cast<Pipeline*>(ptr);auto& l=p.layers[p.current];
    if(lm_wait(p.mapped,2))return fail(lm_error());
    {std::unique_lock<std::mutex> lock(p.final_mutex);p.final_cv.wait(lock,[&]{return !p.final_busy;});
     if(!p.final_error.empty())return fail(p.final_error.c_str());}
    if(p.no_waste_policy)return no_waste_late_retain(p);
    auto ticket=l.cache.plan().ticket;
    if(l.cache.mark_launched(ticket)||l.cache.commit(ticket))return fail("Split commit failed");
    return lp_keep(ptr);
}

}

static int prepare_next_a(Pipeline& p,int layer){
    auto& l=p.layers[layer];
    if(p.execution){p.execution->build_prefix(l.cache,l.offsets.data(),p.group_bytes);p.prebuilt_a=true;return 0;}
    for(int k=0;k<p.keep_size;k++){
        int g=l.cache.view(2)[k],slot=l.cache.view(1)[g];
        if(slot<0)return fail("Keep not resident");
        int32_t* d=p.desc_a+5*k;d[0]=g;d[1]=int(l.offsets[g]/p.group_bytes);d[2]=slot;d[3]=slot;d[4]=0;
    }

    p.prebuilt_a=true;return 0;
}
struct MaintenanceQueue {
    std::mutex mutex;std::condition_variable cv;std::array<Pipeline*,64> queue{};size_t head=0,count=0;bool stop=false;std::thread worker;
    MaintenanceQueue():worker([this]{
        for(;;){Pipeline* p;
            {std::unique_lock<std::mutex> lock(mutex);cv.wait(lock,[&]{return stop||count;});
             if(stop&&!count)return;p=queue[head];head=(head+1)%queue.size();--count;}
            cpu_set_t mask;CPU_ZERO(&mask);for(int i=0;i<p->io_cpu_count;i++)CPU_SET(p->io_cpus[i],&mask);
            sched_setaffinity(0,sizeof(mask),&mask);
#ifdef LEANSTREAM_NVTX_PROFILE
            char range[96];std::snprintf(range,sizeof(range),"scheduling maintenance layer=%d",int(p->current));
            LeanstreamNvtxScope trace(range);
#endif
            int rc=lp_split_keep(p);if(!rc)rc=prepare_next_a(*p,p->current);
            if(!rc&&p->ready_driven&&p->io&&!p->diagnostic_trace){
                auto& l=p->layers[p->current];
                if(pio_rebind(p->io,l.fd,l.offsets.data()))rc=fail("Next-token I/O prebind failed");
                else p->io_prebound=true;
            }
            if(!rc&&p->p0_budget)rc=build_p0_rank(*p);
            {std::lock_guard<std::mutex> lock(p->maintenance_mutex);
             if(rc)p->maintenance_error=error;
             p->maintenance_pending.store(false,std::memory_order_release);p->maintenance_cv.notify_all();}
        }
    }){}
    ~MaintenanceQueue(){ {std::lock_guard<std::mutex> lock(mutex);stop=true;cv.notify_all();}worker.join();}
    int push(Pipeline* p){std::lock_guard<std::mutex> lock(mutex);if(count==queue.size())return fail("Maintenance queue full");queue[(head+count)%queue.size()]=p;++count;cv.notify_one();return 0;}
};
static MaintenanceQueue& maintenance_queue(){static MaintenanceQueue q;return q;}
extern "C" {
int lp_exec_bind(void* ptr,int32_t* queue,uint32_t* flags,int chunk,int profile){
    auto& p=*static_cast<Pipeline*>(ptr);
#ifndef STREAM_IO_V2
    return fail("Execution queue requires stable stream_io_v2 staging");
#else
    if(p.current>=0||p.execution||!queue||!flags||chunk<1||chunk>p.fill_size)return fail("Invalid execution-queue binding");
    if(lp_split_bind(ptr,queue,queue+p.keep_size*5,flags))return -1;
    p.execution.reset(new ExecutionQueue);
    p.execution->bind(queue,flags,p.missing_index.data(),p.groups,p.keep_size,p.fill_size,chunk,profile!=0);
    int mode=std::getenv("LEANSTREAM_READY_IDS")?std::atoi(std::getenv("LEANSTREAM_READY_IDS")):0;
    if(mode>0){
        if(chunk!=p.fill_size||mode==2||mode>3)return fail("ReadyID model requires one FILL launch, mode 1 or 3");
        unsigned profiling=std::getenv("LEANSTREAM_READY_PROFILE")?std::atoi(std::getenv("LEANSTREAM_READY_PROFILE")):0;
        auto& organizer=ready_organizer();
        if(const char* cpu=std::getenv("LEANSTREAM_ORGANIZER_CPU")){if(organizer.affinity(std::atoi(cpu)))return fail("Organizer affinity");}
        else if(organizer.affinity(p.io_cpus.data(),p.io_cpu_count))return fail("Organizer shared affinity");
        p.ready_ids.reset(new ReadyClient(organizer,p.groups,ReadyMode(mode),profiling,&p,organize_ready));
    }

    return 0;
#endif
}
const ExecProfile* lp_exec_profile(void* ptr){auto& p=*static_cast<Pipeline*>(ptr);return p.execution?p.execution->snapshot():nullptr;}
int lp_exec_count(void* ptr){auto& p=*static_cast<Pipeline*>(ptr);return p.execution?p.execution->size():-1;}
void* lp_create_bank(void* parent){
    auto& source=*static_cast<Pipeline*>(parent);auto* p=new Pipeline;
    p->predictor_path=source.predictor_path;p->io_cpus=source.io_cpus;p->io_cpu_count=source.io_cpu_count;
    p->host_slot.fill(-1);p->owners.fill(-1);p->mapped=lm_create();p->shared=si_create(p->predictor_path.c_str());
    if(!p->mapped||!p->shared){fail("Per-layer bank initialization");delete p;return nullptr;}return p;
}
int lp_prepare_a(void* ptr,int layer){auto& p=*static_cast<Pipeline*>(ptr);p.current=layer;return prepare_next_a(p,layer);}
int lp_maintenance_wait(void* ptr,int count){
    auto& p=*static_cast<Pipeline*>(ptr);
    if(p.maintenance_pending.load(std::memory_order_acquire)){
        double start=ms();if(count)p.main_waits++;
        std::unique_lock<std::mutex> lock(p.maintenance_mutex);
        p.maintenance_cv.wait(lock,[&]{return !p.maintenance_pending.load(std::memory_order_acquire);});
        if(count)p.main_wait_ms+=ms()-start;
    }
    return p.maintenance_error.empty()?0:fail(p.maintenance_error.c_str());
}
uint64_t lp_main_waits(void* ptr){return static_cast<Pipeline*>(ptr)->main_waits;}
double lp_main_wait_ms(void* ptr){return static_cast<Pipeline*>(ptr)->main_wait_ms;}
int lp_maintenance_submit(void* ptr){
    auto& p=*static_cast<Pipeline*>(ptr);
    if(p.maintenance_pending.exchange(true))return fail("Maintenance bank reused early");
    int rc=maintenance_queue().push(&p);if(rc)p.maintenance_pending.store(false,std::memory_order_release);return rc;
}
}

// Speculative only. FIFO is P1 -> optional midpoint -> P2. Token tickets
// and final-generation guard reject stale jobs without touching final/cache state.
static int midpoint(Pipeline& p,uint64_t token){
    if(p.diagnostic_trace)p.mid_trace[6]=ns();
    if(p.current<0||p.p2_queued_token.load(std::memory_order_acquire)>=token||
       !midpoint_current(token,p.submitted_token.load(std::memory_order_acquire),p.p2_published)){p.mid_times[6]++;return 0;}
    if(p.diagnostic_trace)p.mid_trace[0]=ns();
    if(p.ready_driven){
        if(__atomic_load_n(lm_ready_flag(p.mapped,1),__ATOMIC_ACQUIRE)!=p.mid_epoch)
            return fail("Mid worker received non-READY payload");
    }else if(lm_wait(p.mapped,1))return fail(lm_error());
    if(p.diagnostic_trace)p.mid_trace[1]=ns();
    if(p.p2_queued_token.load(std::memory_order_acquire)>=token){p.mid_times[6]++;return 0;}
    auto& l=p.layers[p.current];
    if(p.semantic_mid_rescue){
#ifndef STREAM_IO_V2
        return fail("Semantic Mid requires stable stream states");
#else
        p.mid_count=p.fill_size;
        if(si_predict_ready(p.shared,static_cast<const uint16_t*>(lm_pointer(p.mapped,1,0)),
            l.features,l.weights,l.scales,l.cache.view(2),p.fill_mid.data(),p.mid_times.data()))
            return fail(si_error());
        if(p.p2_queued_token.load(std::memory_order_acquire)>=token){p.mid_times[6]++;return 0;}
        const auto* states=static_cast<const uint32_t*>(lm_pointer(p.mapped,5,0));
        std::array<std::pair<int32_t,int32_t>,128> pending{};int pending_count=0;
        for(int j=0;j<p.p1_priority_count;j++){
            int g=p.p1_priority[j];
            if(l.cache.cache_resident(g)||__atomic_load_n(states+g,__ATOMIC_ACQUIRE)!=STREAM_EMPTY)continue;
            int rank=p.fill_size+j;
            for(int k=0;k<p.fill_size;k++)if(p.fill_mid[k]==g){rank=k;break;}
            pending[pending_count++]={rank,g};
        }
        std::stable_sort(pending.begin(),pending.begin()+pending_count,
            [](const auto& a,const auto& b){return a.first<b.first;});
        int n=0,rescued=0;
        for(int k=0;k<p.fill_size&&rescued<p.semantic_mid_rescue;k++){
            int g=p.fill_mid[k];bool published=false;
            for(int j=0;j<p.p1_priority_count;j++)if(p.p1_priority[j]==g){published=true;break;}
            if(published||l.cache.cache_resident(g)||
               __atomic_load_n(states+g,__ATOMIC_ACQUIRE)!=STREAM_EMPTY)continue;
            p.speculative[n++]=g;++rescued;
        }
        for(int j=0;j<pending_count;j++)p.speculative[n++]=pending[j].second;
        p.semantic_mid_metrics[0]=rescued;p.semantic_mid_metrics[1]=pending_count;
        p.mid_count=n;
        if(p.current<0||p.p2_queued_token.load(std::memory_order_acquire)>=token||
           !midpoint_current(token,p.submitted_token.load(std::memory_order_acquire),p.p2_published)){
            p.mid_times[6]++;p.semantic_mid_metrics[3]++;return 0;
        }
        std::copy_n(p.speculative.data(),n,p.fill_mid.data());
        if(p.diagnostic_trace)p.mid_trace[2]=ns();
        double start=ms();
        if(p.mid_publish_enabled){
            if(p.publish(p.speculative.data(),n))return -1;
            p.semantic_mid_metrics[2]=n;p.mid_generation=p.generation;
        }
        p.mid_times[3]=ms()-start;p.mid_times[4]=n;p.mid_times[5]++;
        if(p.diagnostic_trace)p.mid_trace[3]=ns();
        p.mid_times[11]=ms();return 0;
#endif
    }
    if(p.tiny_mid_features){
#ifndef STREAM_IO_V2
        return fail("Tiny Mid requires stable stream states");
#else
        const auto* states=static_cast<const uint32_t*>(lm_pointer(p.mapped,5,0));
        int pending=0;
        for(int j=0;j<p.p1_priority_count;j++){
            int g=p.p1_priority[j];
            if(!l.cache.cache_resident(g)&&__atomic_load_n(states+g,__ATOMIC_ACQUIRE)==STREAM_EMPTY)
                p.mid_before[pending++]=g;
        }
        p.mid_pending_count=pending;
        if(!pending){p.mid_times[5]++;return 0;}
        p.tiny_mid_metrics[7]=ns();
        if(si_predict_subset(p.tiny_mid_shared,static_cast<const uint16_t*>(lm_pointer(p.mapped,1,0)),
            l.mid_features,l.mid_weights,l.mid_scales,p.mid_before.data(),pending,pending,
            p.mid_after.data(),p.mid_times.data()))return fail(si_error());
        p.tiny_mid_metrics[8]=ns();
        p.mid_count=pending;
        std::copy_n(p.mid_after.data(),pending,p.fill_mid.data());
#endif
    }else if(p.cheap_mid_size){
        p.mid_count=std::min(p.cheap_mid_size,p.fill_size);
        const int32_t* features=l.mid_features?l.mid_features:l.features;
        const void* weights=l.mid_weights?l.mid_weights:l.weights;
        const float* scales=l.mid_scales?l.mid_scales:l.scales;
        if(si_predict_subset(p.shared,static_cast<const uint16_t*>(lm_pointer(p.mapped,1,0)),
            features,weights,scales,p.continuation.data(),p.cheap_mid_size,p.mid_count,
            p.fill_mid.data(),p.mid_times.data()))return fail(si_error());
    }else {
        p.mid_count=p.fill_size;
        if(si_predict_ready(p.shared,static_cast<const uint16_t*>(lm_pointer(p.mapped,1,0)),l.features,l.weights,l.scales,
            l.cache.view(2),p.fill_mid.data(),p.mid_times.data()))return fail(si_error());
    }
    if(p.current<0||!midpoint_current(token,p.submitted_token.load(std::memory_order_acquire),p.p2_published)){p.mid_times[6]++;return 0;}
    if(p.tiny_mid_features&&p.p2_queued_token.load(std::memory_order_acquire)>=token){p.mid_times[6]++;return 0;}
    if(p.diagnostic_trace)p.mid_trace[2]=ns();
    int n=0;
    for(int j=0;j<p.mid_count;j++)if(p.tiny_mid_features||!l.cache.cache_resident(p.fill_mid[j]))p.speculative[n++]=p.fill_mid[j];
    if(p.diagnostic_trace)p.mid_trace[4]=ns();
    double start=ms();
    if(p.mid_publish_enabled){
        if(p.publish(p.speculative.data(),n))return -1;
        p.mid_generation=p.generation;
    }
    if(p.tiny_mid_features)p.tiny_mid_metrics[9]=ns();
    if(p.diagnostic_trace)p.mid_trace[3]=ns();
    p.mid_times[3]=ms()-start;p.mid_times[4]=n;p.mid_times[5]++;
    p.mid_times[11]=ms();return 0;
}
extern "C" const uint64_t* lp_mid_trace(void* ptr){return static_cast<Pipeline*>(ptr)->mid_trace.data();}
extern "C" int lp_mid_publish(void* ptr,int enabled){
    auto& p=*static_cast<Pipeline*>(ptr);std::lock_guard<std::mutex> lock(p.control_mutex);
    if(p.control_jobs||(enabled!=0&&enabled!=1))return fail("Midpoint mode requires a drained control queue");
    p.mid_publish_enabled=enabled;return 0;
}
extern "C" const double* lp_mid_times(void* ptr){return static_cast<Pipeline*>(ptr)->mid_times.data();}
extern "C" const uint64_t* lp_semantic_mid_metrics(void* ptr){return static_cast<Pipeline*>(ptr)->semantic_mid_metrics.data();}
extern "C" const int32_t* lp_p1_priority(void* ptr){return static_cast<Pipeline*>(ptr)->p1_priority.data();}
extern "C" int lp_p1_priority_count(void* ptr){return static_cast<Pipeline*>(ptr)->p1_priority_count;}
extern "C" int lp_mid_stale_test(void* ptr,uint64_t token){return midpoint(*static_cast<Pipeline*>(ptr),token);}
// Optional orchestration: reuse the existing W4A8 context on the same worker.
static int checkpoint_predict(Pipeline& p,int phase,uint64_t token){
 int id=phase-4,point=id+1;
 if(__atomic_load_n(si_ready_flag(p.shared),__ATOMIC_ACQUIRE)==si_expected_epoch(p.shared)||p.p2_published){p.checkpoint_stats[5+point]++;return 0;}
 p.checkpoint_stats[point]++;auto& l=p.layers[p.current];double time[3];
 if(si_predict_ready(p.shared,p.checkpoint_data[id],l.features,l.weights,l.scales,l.cache.view(2),p.checkpoint_rank.data(),time))return fail(si_error());
 if(__atomic_load_n(si_ready_flag(p.shared),__ATOMIC_ACQUIRE)==si_expected_epoch(p.shared)||p.p2_queued_token.load(std::memory_order_acquire)>=token||p.p2_published||p.submitted_token.load()!=token){p.checkpoint_stats[5+point]++;return 0;}
 auto* states=static_cast<const uint32_t*>(lm_pointer(p.mapped,5,0));int count=0;
 for(int i=0;i<p.fill_size;i++){int g=p.checkpoint_rank[i];if(!l.cache.cache_resident(g)&&__atomic_load_n(states+g,__ATOMIC_ACQUIRE)==STREAM_EMPTY&&uint32_t(count)<p.checkpoint_budget[id])p.speculative[count++]=g;}
 uint64_t before=p.generation;p.checkpoint_publishing=true;int rc=public_update(p,p.speculative.data(),count,false);p.checkpoint_publishing=false;p.checkpoint_stats[10+point]+=p.generation-before;return rc;
}
// Bounded, persistent single predictor master; FIFO enforces P1 before P2.
struct ControlQueue {
    struct Job{Pipeline* p=nullptr;int phase=0,hidden=0;uint64_t token=0,submitted_ns=0;};
    std::array<Job,128> jobs{}; // Up to three queued refinements per each of 32 layers.
    size_t head=0,count=0;
    std::mutex mutex;std::condition_variable cv;bool stop=false,initialized=false;int init_status=0;std::thread worker;
    explicit ControlQueue(Pipeline* first):worker([this,first]{
        int rc=si_worker_init(first->shared);
        {std::lock_guard<std::mutex> lock(mutex);init_status=rc;initialized=true;cv.notify_all();}
        for(;;){Job job;
            {std::unique_lock<std::mutex> lock(mutex);cv.wait(lock,[&]{return stop||count;});
             if(stop&&!count)return;job=jobs[head];head=(head+1)%jobs.size();--count;}
            auto& p=*job.p;
            if(p.diagnostic_trace && job.phase<=2){auto begin=p.control_trace.begin()+(job.phase-1)*16;std::fill_n(begin,16,0);begin[0]=job.submitted_ns;p.stamp(job.phase,1);}
            {std::lock_guard<std::mutex> lock(p.control_mutex);rc=p.control_error.empty()?0:-1;}
            if(job.phase==3&&p.diagnostic_trace)p.mid_trace[5]=job.submitted_ns;
            if(!rc)rc=job.phase==1?lp_p1(&p,p.current,job.hidden):(job.phase==3?midpoint(p,job.token):lp_split_p2(&p));
            if(job.phase<=2)p.stamp(job.phase,12);
            {std::lock_guard<std::mutex> lock(p.control_mutex);
             if(rc&&p.control_error.empty())p.control_error=error;
             --p.control_jobs;p.control_cv.notify_all();}
        }
    }){std::unique_lock<std::mutex> lock(mutex);cv.wait(lock,[&]{return initialized;});}
    ~ControlQueue(){{std::lock_guard<std::mutex> lock(mutex);stop=true;cv.notify_all();}worker.join();}
    int push(Pipeline* p,int phase,int hidden){
        uint64_t submitted=p->diagnostic_trace?ns():0;
        std::lock_guard<std::mutex> lock(mutex);
        if(init_status||count==jobs.size())return fail("Control queue unavailable/full");
        {std::lock_guard<std::mutex> bank(p->control_mutex);if(!p->control_error.empty())return fail(p->control_error.c_str());++p->control_jobs;}
        uint64_t token=phase==1?p->submitted_token.fetch_add(1,std::memory_order_acq_rel)+1:p->submitted_token.load(std::memory_order_acquire);
        if(phase==2)p->p2_queued_token.store(token,std::memory_order_release);
        jobs[(head+count)%jobs.size()]={p,phase,hidden,token,submitted};++count;cv.notify_one();return 0;
    }
};
static ControlQueue& control_queue(Pipeline* first){static ControlQueue queue(first);return queue;}
// GPU epochs are observed by this separate scheduler thread. The predictor
// worker only receives already-acquired payloads and never waits for GPU data.
struct ReadyScheduler {
    std::array<Pipeline*,max_layers> clients{};int nclients=0;
    std::mutex mutex;std::condition_variable cv;
    bool stop=false,initialized=false;int init_status=0;
    int notification_mode=0,notification_error=0;
    uint32_t* bell_host=nullptr,*bell_device=nullptr;
    uint64_t wake_generation=0;
    std::atomic<uint64_t> observer_checks{0};
    std::thread observer,worker;std::array<int,6> stage_order{{2,3,1,0,0,0}};int stage_count=3;
    explicit ReadyScheduler(Pipeline* first){
        notification_mode=first->ready_notification_mode;
        if(notification_mode){
            auto rc=cudaHostAlloc(reinterpret_cast<void**>(&bell_host),64,cudaHostAllocMapped);
            if(rc==cudaSuccess)rc=cudaHostGetDevicePointer(reinterpret_cast<void**>(&bell_device),bell_host,0);
            if(rc==cudaSuccess)__atomic_store_n(bell_host,0u,__ATOMIC_RELEASE);
            else notification_error=-1;
        }
        worker=std::thread([this,first]{
            int init=si_worker_init(first->shared);
            {std::lock_guard<std::mutex> lock(mutex);init_status=init?init:notification_error;initialized=true;cv.notify_all();}
            for(;;){Pipeline* selected=nullptr;int phase=0;uint64_t token=0;
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    cv.wait(lock,[&]{
                        if(stop)return true;
                        for(int order=0;order<stage_count;order++)if(int stage=stage_order[order];true)for(int i=0;i<nclients;i++)
                            if(clients[i]->predictor_arms[stage].state==2)return true;
                        return false;
                    });
                    if(stop)return;
                    if(stage_count==6)for(int i=0;i<nclients;i++){auto* candidate=clients[i];auto& output=candidate->predictor_arms[2];if(candidate->checkpoint_mask&&output.state==1&&__atomic_load_n(output.flag,__ATOMIC_ACQUIRE)==output.epoch){output.state=2;candidate->p2_queued_token.store(output.token,std::memory_order_release);}}
                    for(int order=0;order<stage_count;order++)if(int stage=stage_order[order];true){
                        for(int i=0;i<nclients;i++){
                            auto* p=clients[i];auto& arm=p->predictor_arms[stage];
                            if(arm.state!=2)continue;
                            selected=p;phase=stage;token=arm.token;arm.state=3;if(stage>=4||stage==2){for(int older=4;older<stage || (stage==2&&older<=6);older++){auto& stale=p->predictor_arms[older];if(stale.state==1||stale.state==2){stale.state=4;p->checkpoint_stats[5+older-3]++;std::lock_guard<std::mutex> bank(p->control_mutex);--p->control_jobs;p->control_cv.notify_all();}}}
                            p->ready_trace[stage][2]=ns();break;
                        }
                        if(selected)break;
                    }
                }
                if(!selected)continue;
                auto& p=*selected;int rc=0;
#ifdef LEANSTREAM_NVTX_PROFILE
                char range[128];std::snprintf(range,sizeof(range),"predictor stage=%d token=%llu layer=%d",
                    phase,(unsigned long long)token,int(p.current));
                LeanstreamNvtxScope trace(range);
#endif
                uint64_t cp_before=p.generation;
                if(p.current.load(std::memory_order_acquire)!=p.predictor_arms[phase].layer||
                   p.submitted_token.load(std::memory_order_acquire)!=token||
                   (phase!=2&&p.p2_published)){
                    if(phase==1)p.p1_prepared=false;
                }else if(phase==1){if(p.checkpoint_mask)p.checkpoint_stats[0]++;rc=lp_p1(&p,p.current,p.predictor_arms[phase].hidden);}
                else if(phase==2){if(p.checkpoint_mask)p.checkpoint_stats[4]++;rc=lp_split_p2(&p);}
                else if(phase>=4)rc=checkpoint_predict(p,phase,token);else rc=midpoint(p,token);
                if(p.checkpoint_mask&&phase<=2)p.checkpoint_stats[10+(phase==1?0:4)]+=p.generation-cp_before;
                p.ready_trace[phase][3]=ns();p.ready_trace[phase][4]=p.ready_trace[phase][3];
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    p.predictor_arms[phase].state=4;++wake_generation;cv.notify_all();
                }
                {std::lock_guard<std::mutex> lock(p.control_mutex);
                    if(rc&&p.control_error.empty())p.control_error=error;
                    --p.control_jobs;p.control_cv.notify_all();}
            }
        });
        observer=std::thread([this]{
            uint32_t bell_seen=0;uint64_t generation_seen=0;
            for(;;){bool pending=false,changed=false;
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    cv.wait(lock,[&]{
                        if(stop)return true;
                        for(int i=0;i<nclients;i++)for(int stage=1;stage<=(stage_count==6?6:3);stage++)
                            if(clients[i]->predictor_arms[stage].state==1)return true;
                        return false;
                    });
                    if(stop)return;
                    if(notification_mode){
                        auto observed=__atomic_load_n(bell_host,__ATOMIC_ACQUIRE);
                        if(observed==bell_seen&&wake_generation==generation_seen){
                            lock.unlock();std::this_thread::sleep_for(std::chrono::microseconds(5));continue;
                        }
                        bell_seen=observed;generation_seen=wake_generation;
                    }
                    for(int order=0;order<stage_count;order++)if(int stage=stage_order[order];true)for(int i=0;i<nclients;i++){
                        auto* p=clients[i];auto& arm=p->predictor_arms[stage];
                        if(arm.state!=1)continue;
                        ++observer_checks;
                        pending=true;
                        const bool obsolete=(p->submitted_token.load(std::memory_order_acquire)!=arm.token)||
                            (stage!=2&&p->p2_published);
                        if(obsolete){
                            arm.state=4;changed=true;
                            if(stage==1)p->p1_prepared=false;
                            {std::lock_guard<std::mutex> bank(p->control_mutex);
                             --p->control_jobs;p->control_cv.notify_all();}
                            continue;
                        }
                        if(stage==3&&p->continuation_token.load(std::memory_order_acquire)!=arm.token)
                            continue;
                        if(__atomic_load_n(arm.flag,__ATOMIC_ACQUIRE)==arm.epoch){
                            arm.state=2;p->ready_trace[stage][1]=ns();changed=true;
                            if(stage==2)p->p2_queued_token.store(arm.token,std::memory_order_release);
                        }
                    }
                    if(changed)cv.notify_all();
                }
                if(pending&&!notification_mode){
#ifdef __aarch64__
                    asm volatile("yield" ::: "memory");
#endif
                    std::this_thread::yield();
                }
            }
        });
        std::unique_lock<std::mutex> lock(mutex);cv.wait(lock,[&]{return initialized;});
    }
    ~ReadyScheduler(){
        {std::lock_guard<std::mutex> lock(mutex);stop=true;cv.notify_all();}
        observer.join();worker.join();
        if(bell_host)cudaFreeHost(bell_host);
    }
    int add(Pipeline* p){
        std::lock_guard<std::mutex> lock(mutex);
        if(init_status||nclients==max_layers||p->ready_notification_mode!=notification_mode)
            return fail("Ready predictor scheduler unavailable/full/mode mismatch");
        if(notification_mode){lm_notify_doorbell(p->mapped,bell_device);si_notify_doorbell(p->shared,bell_device);}
        clients[nclients++]=p;++wake_generation;cv.notify_all();return 0;
    }
    int arm(Pipeline* p,int phase,int hidden){
        std::lock_guard<std::mutex> lock(mutex);
        auto& arm=p->predictor_arms[phase];
        if(arm.state==1||arm.state==2||arm.state==3)return fail("Predictor stage already armed");
        const uint32_t* flag=phase>=4?p->checkpoint_flags[phase-4]:(phase==2?si_ready_flag(p->shared):lm_ready_flag(p->mapped,phase==1?hidden:1));
        uint32_t epoch=phase>=4?uint32_t(p->submitted_token.load()):(phase==2?si_expected_epoch(p->shared):lm_expected_epoch(p->mapped,phase==1?hidden:1));
        if(!flag||!epoch)return fail("Missing predictor READY epoch");
        uint64_t token=phase==1?p->submitted_token.fetch_add(1,std::memory_order_acq_rel)+1:
                                p->submitted_token.load(std::memory_order_acquire);
        {std::lock_guard<std::mutex> bank(p->control_mutex);
         if(!p->control_error.empty())return fail(p->control_error.c_str());++p->control_jobs;}
        arm={1,hidden,p->current.load(),token,flag,epoch};
        if(phase==1)p->p1_epoch=epoch;
        else if(phase==2)p->p2_epoch=epoch;
        else p->mid_epoch=epoch;
        p->ready_trace[phase][0]=ns();++wake_generation;cv.notify_all();return 0;
    }
    void remove(Pipeline* p){
        {
            std::lock_guard<std::mutex> lock(mutex);
            for(int stage=1;stage<=(stage_count==6?6:3);stage++){
                auto& arm=p->predictor_arms[stage];
                if(arm.state==1||arm.state==2){
                    arm.state=4;
                    std::lock_guard<std::mutex> bank(p->control_mutex);
                    --p->control_jobs;p->control_cv.notify_all();
                }
            }
            cv.notify_all();
        }
        {std::unique_lock<std::mutex> bank(p->control_mutex);
         p->control_cv.wait(bank,[&]{return p->control_jobs==0;});}
        std::lock_guard<std::mutex> lock(mutex);
        if(notification_mode){lm_notify_doorbell(p->mapped,nullptr);si_notify_doorbell(p->shared,nullptr);}
        for(int i=0;i<nclients;i++)if(clients[i]==p){clients[i]=clients[--nclients];break;}
    }
};
static ReadyScheduler& ready_scheduler(Pipeline* first=nullptr){static ReadyScheduler scheduler(first);return scheduler;}
static void ready_unregister(void* ptr){ready_scheduler().remove(static_cast<Pipeline*>(ptr));}
extern "C" {
int lp_ready_scheduler_enable(void* ptr){
    auto& p=*static_cast<Pipeline*>(ptr);
    if(p.async_control||p.current>=0||p.ready_driven)return fail("Late/duplicate ready scheduler enable");
    if(ready_scheduler(&p).add(&p))return -1;p.ready_driven=true;return 0;
}
int lp_ready_notification_mode(void* ptr,int mode){
    auto& p=*static_cast<Pipeline*>(ptr);
    if(p.ready_driven||p.async_control||p.current>=0||(mode!=0&&mode!=1))return fail("Late/invalid READY notification mode");
    p.ready_notification_mode=mode;return 0;
}
void* lp_ready_doorbell_device(void* ptr){
    auto& p=*static_cast<Pipeline*>(ptr);
    return p.ready_driven&&p.ready_notification_mode?ready_scheduler().bell_device:nullptr;
}
uint64_t lp_ready_observer_checks(){return ready_scheduler().observer_checks.load();}
uint64_t lp_ready_observer_cpu_ns(){
    clockid_t clock_id;
    if(pthread_getcpuclockid(ready_scheduler().observer.native_handle(),&clock_id))return 0;
    timespec t{};if(clock_gettime(clock_id,&t))return 0;
    return uint64_t(t.tv_sec)*1000000000ull+uint64_t(t.tv_nsec);
}
int lp_async_enable(void* ptr){auto& p=*static_cast<Pipeline*>(ptr);
    if(!p.ready_driven){auto& q=control_queue(&p);if(q.init_status)return fail("Predictor control init");}
    if(p.speculate&&si_enable_ranking(p.shared))return fail(si_error());p.async_control=true;return 0;}
int lp_control_arm(void* ptr,int phase,int hidden){
    auto& p=*static_cast<Pipeline*>(ptr);
    if(!p.ready_driven||!p.async_control||(phase!=1&&phase!=2&&phase!=3)||
       hidden<0||(phase==1?hidden>=p.layer_count:hidden>1))
        return fail("Invalid READY predictor arm");
    if(phase==1&&!p.prediction_input_enabled){if(!p.p1_prepared&&prepare_p1(p,hidden,0))return -1;p.submitted_token.fetch_add(1,std::memory_order_acq_rel);return 0;}
    uint64_t arm_begin=p.diagnostic_trace?ns():0;
    if(phase==1 && !p.p1_prepared && prepare_p1(p,hidden,0))return -1;
    if(p.diagnostic_trace)p.ready_trace[phase][5]=arm_begin;
    return ready_scheduler().arm(&p,phase,phase==1?0:hidden);
}
int lp_control_submit(void* ptr,int phase,int hidden){
    auto& p=*static_cast<Pipeline*>(ptr);if(!p.async_control||(phase!=1&&phase!=2&&phase!=3)||hidden<0||hidden>1)return fail("Invalid control job");
    return control_queue(&p).push(&p,phase,hidden);
}
const uint64_t* lp_ready_predictor_trace(void* ptr,int phase){
    auto& p=*static_cast<Pipeline*>(ptr);return phase>=1&&phase<=3?p.ready_trace[phase].data():nullptr;
}
const uint64_t* lp_ready_p1_io_trace(void* ptr){return &static_cast<Pipeline*>(ptr)->p1_io_begin_ns;}
const uint64_t* lp_final_io_state(void* ptr){return static_cast<Pipeline*>(ptr)->final_io_state.data();}
int lp_control_wait(void* ptr){
    auto& p=*static_cast<Pipeline*>(ptr);std::unique_lock<std::mutex> lock(p.control_mutex);
    p.control_cv.wait(lock,[&]{return !p.control_jobs;});return p.control_error.empty()?0:fail(p.control_error.c_str());
}
}

extern "C" int lp_final_qd(void* ptr,int qd){auto& p=*static_cast<Pipeline*>(ptr);if(p.io||(qd<1||qd>32))return fail("Invalid initial final QD");p.final_qd=qd;p.pool_qd=qd;return 0;}
extern "C" double lp_idle_before_p2(void* ptr){return static_cast<Pipeline*>(ptr)->idle_before_p2_ms;}

extern "C" int lp_speculation(void* ptr,int enabled){auto& p=*static_cast<Pipeline*>(ptr);if(p.io||(enabled!=0&&enabled!=1))return fail("Invalid speculation option");p.speculate=enabled;return 0;}

extern "C" void* lp_io(void* ptr){return static_cast<Pipeline*>(ptr)->io;} // Borrowed diagnostic handle; no ownership transfer.

extern "C" uint64_t lp_mid_token(void* ptr){return static_cast<Pipeline*>(ptr)->submitted_token.load();}
extern "C" uint64_t lp_priority_generation(void* ptr){return static_cast<Pipeline*>(ptr)->generation;}

extern "C" const uint64_t* lp_generation_trace(void* ptr){return static_cast<Pipeline*>(ptr)->priority_trace.data();}

extern "C" void lp_diagnostic_trace(void* ptr,int enabled){
    auto& p=*static_cast<Pipeline*>(ptr);p.diagnostic_trace=enabled;
    si_capture_scores(p.shared,enabled||p.relaxed_p2_m||p.no_waste_policy);
}
extern "C" const uint64_t* lp_control_trace(void* ptr){return static_cast<Pipeline*>(ptr)->control_trace.data();}

// Initialization/drained comparison boundary only. CPU-only weight pointers.
extern "C" int lp_predictor_backing(void* ptr,int layer,const void* weights,const float* scales){
    auto& p=*static_cast<Pipeline*>(ptr);
    if(layer<0||layer>=p.layer_count||!weights||!scales||p.maintenance_pending.load())return fail("Predictor backing ownership");
    std::lock_guard<std::mutex> lock(p.control_mutex);
    if(p.control_jobs)return fail("Predictor backing changed while active");
    p.layers[layer].weights=weights;p.layers[layer].scales=scales;return 0;
}

extern "C" int lp_io_config(void* ptr,int pre,int final){
 auto& p=*static_cast<Pipeline*>(ptr);std::lock_guard<std::mutex> lock(p.control_mutex);
 if(p.control_jobs||p.maintenance_pending.load()||pre<1||final<1||pre>p.pool_qd||final>p.pool_qd)return fail("I/O config ownership");
 p.pre_qd=pre;p.final_qd=final;return 0;
}
extern "C" int lp_io_log(void* ptr,Completion* out,int capacity){auto& p=*static_cast<Pipeline*>(ptr);return p.io?pio_log(p.io,out,capacity):0;}

extern "C" int lp_ready_id_stats(void* ptr,ReadyStats* out){auto& p=*static_cast<Pipeline*>(ptr);if(!p.ready_ids||!out)return -1;*out=p.ready_ids->snapshot();return 0;}
extern "C" const ReadyTrace* lp_ready_id_trace(void* ptr){auto& p=*static_cast<Pipeline*>(ptr);return p.ready_ids?p.ready_ids->trace():nullptr;}
extern "C" uint64_t lp_io_cpu_mask(void* ptr){auto& p=*static_cast<Pipeline*>(ptr);uint64_t mask=0;for(int i=0;i<p.io_cpu_count;i++)if(p.io_cpus[i]<64)mask|=uint64_t(1)<<p.io_cpus[i];return mask;}

extern "C" int lp_ready_id_tail(void* ptr,uint64_t* tail){auto& p=*static_cast<Pipeline*>(ptr);if(!p.ready_ids||!p.execution||!tail||uintptr_t(tail)%8)return -1;p.execution->bind_tail(tail);return 0;}

extern "C" int lp_checkpoint_bind(void* ptr,uint32_t mask,const uint16_t* const* data,const uint32_t* const* flags,const uint32_t* budgets){
 auto& p=*static_cast<Pipeline*>(ptr);if(!p.public_plan||!p.ready_driven||p.control_jobs||mask>7||!data||!flags||!budgets)return fail("Invalid checkpoint binding");
 auto& scheduler=ready_scheduler();std::lock_guard<std::mutex> lock(scheduler.mutex);p.checkpoint_mask=mask;
 for(int i=0;i<3;i++)if(mask&(1u<<i)){if(!data[i]||!flags[i])return fail("Null checkpoint mailbox");p.checkpoint_data[i]=data[i];p.checkpoint_flags[i]=flags[i];p.checkpoint_budget[i]=budgets[i]==UINT32_MAX?128:budgets[i];}
 scheduler.stage_order={2,6,5,4,3,1};scheduler.stage_count=6;return 0;
}
extern "C" uint32_t lp_checkpoint_arm(void* ptr){auto& p=*static_cast<Pipeline*>(ptr);uint64_t token=p.submitted_token.load();if(!token||token>UINT32_MAX)return 0;for(int i=0;i<3;i++)if(p.checkpoint_mask&(1u<<i))if(ready_scheduler().arm(&p,4+i,0))return 0;return token;}
extern "C" const uint64_t* lp_checkpoint_stats(void* ptr){return static_cast<Pipeline*>(ptr)->checkpoint_stats.data();}

extern "C" uint32_t lp_checkpoint_token(void* ptr){return uint32_t(static_cast<Pipeline*>(ptr)->submitted_token.load());}

extern "C" int lp_prediction_input_config(void* ptr,int enabled,uint32_t budget){auto& p=*static_cast<Pipeline*>(ptr);if(p.control_jobs)return fail("Prediction config while busy");p.prediction_input_enabled=enabled!=0;p.input_io_budget=budget;return 0;}
