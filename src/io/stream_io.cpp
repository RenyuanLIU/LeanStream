#include "stream_io.h"
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <time.h>
#include <errno.h>
#include <algorithm>
#include <condition_variable>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include <chrono>
#ifdef LEANSTREAM_NVTX_PROFILE
#include <nvToolsExt.h>
#include <cstdio>
struct LeanstreamStreamNvtxScope {
 explicit LeanstreamStreamNvtxScope(const char* name){nvtxRangePushA(name);}
 ~LeanstreamStreamNvtxScope(){nvtxRangePop();}
};
#endif
#ifdef STREAM_TESTING
extern "C" ssize_t stream_test_pread(int,void*,size_t,off_t);
#define READ_BLOCK stream_test_pread
#else
#define READ_BLOCK pread
#endif
static uint64_t wall(){timespec t;clock_gettime(CLOCK_MONOTONIC_RAW,&t);return uint64_t(t.tv_sec)*1000000000+t.tv_nsec;}
static uint64_t cpu(){timespec t;clock_gettime(CLOCK_THREAD_CPUTIME_ID,&t);return uint64_t(t.tv_sec)*1000000000+t.tv_nsec;}
static uint32_t state(const uint32_t* p){return __atomic_load_n(p,__ATOMIC_ACQUIRE);}
static void publish(uint32_t* p,uint32_t v){__atomic_store_n(p,v,__ATOMIC_RELEASE);}
struct StreamSession {
 StreamEngine* engine;int fd=-1,count=0;const StreamExtent* layout;StreamArena arena;
 std::vector<int32_t> ranks[2],committed;std::vector<uint8_t> committed_bits,seen;
 // CPU-only membership, accessed under engine->mutex once the session is live.
 std::vector<uint64_t> claimed_bits;
 bool claimed(int id) const {return (claimed_bits[id >> 6] >> (id & 63)) & 1u;}
 void claim(int id) {claimed_bits[id >> 6] |= uint64_t{1} << (id & 63);}
 std::vector<uint8_t> consume_states; // profiling-only, never GPU handoff storage
 StreamRefinementProfile refinement{};
 StreamMaintenanceProfile maintenance{};
 StreamFlowProfile flow{};std::vector<StreamFlowBlock> flow_blocks;uint64_t depth_changed=0;
 std::vector<StreamBlockInfo> info;std::vector<StreamUpdateProfile> updates;
 int active_rank=0,rank_size=0,cursor=0,fallback=0,debug_waiters=0;bool consume=false,closing=false;
 uint64_t speculation_ceiling=UINT64_MAX;
 StreamStatus status{};StreamProfile profile{};StreamObserver observer=nullptr;void* observer_arg=nullptr;
 ~StreamSession(){if(fd>=0)close(fd);}
};
struct StreamEngine {
 StreamConfig config;std::mutex mutex;std::condition_variable work,changed;bool stop=false;
 std::unique_ptr<StreamSession> session;std::vector<std::thread> workers;uint64_t generation=0;
 explicit StreamEngine(StreamConfig c):config(c){
  try{for(unsigned i=0;i<c.total_qd;i++)workers.emplace_back([this]{run();});}
  catch(...){ {std::lock_guard<std::mutex> l(mutex);stop=true;work.notify_all();}for(auto& t:workers)t.join();throw;}
 }
 int select(StreamSession& s){
  if(!s.consume&&s.status.admissions>=s.speculation_ceiling)return -1;
  auto eligible=[&](int id){
   bool claimed=s.claimed(id);
   if(config.profile>=2){s.flow.candidates++;if(claimed){auto v=state(s.arena.states+id);s.flow.skipped_ready+=v==STREAM_READY;s.flow.skipped_inflight+=v==STREAM_INFLIGHT;s.flow.skipped_error+=v==STREAM_ERROR;}}
   if(config.profile==1){
    s.profile.candidates++;
    if(claimed){auto t=cpu();auto v=state(s.arena.states+id);s.profile.skipped_ready+=v==STREAM_READY;s.profile.skipped_inflight+=v==STREAM_INFLIGHT;s.profile.skipped_error+=v==STREAM_ERROR;
     auto dt=cpu()-t;if(v==STREAM_READY){s.maintenance.skip_ready_count++;s.maintenance.skip_ready_cpu_ns+=dt;}
     if(v==STREAM_INFLIGHT){s.maintenance.skip_inflight_count++;s.maintenance.skip_inflight_cpu_ns+=dt;}}
   }
   return !claimed;
  };
  while(s.cursor<s.rank_size){int id=s.ranks[s.active_rank][s.cursor++];if(eligible(id))return id;}
  if(s.consume)while(s.fallback<int(s.committed.size())){int id=s.committed[s.fallback++];if(eligible(id))return id;}
  return -1;
 }
 void run(){
  const bool profiling=config.profile==1,flow=config.profile>=2;StreamSession* previous=nullptr;int previous_id=-1;uint64_t previous_generation=0;
  std::unique_lock<std::mutex> lock(mutex);
  while(!stop){
   StreamSession* s=session.get();int id=-1;uint64_t t=(profiling||flow)?cpu():0;
   if(s&&!s->closing&&!state(s->arena.states+s->count)&&s->status.outstanding<config.total_qd){
    if(profiling)s->profile.admission_calls++;
    auto checks=profiling?s->profile.candidates:0;
#ifdef LEANSTREAM_NVTX_PROFILE
    {LeanstreamStreamNvtxScope trace("scheduling I/O admission");id=select(*s);}
#else
    id=select(*s);
#endif
    if(profiling){s->maintenance.check_cpu_ns+=cpu()-t;s->maintenance.check_count+=s->profile.candidates-checks;}
   }
   if(id<0){if(profiling&&s&&!s->closing){auto dt=cpu()-t;s->profile.scheduler_cpu_ns+=dt;s->profile.worker_cpu_ns+=dt;}if(s&&s->debug_waiters)changed.notify_all();work.wait(lock);continue;}
   if(flow){auto now=wall();if(!s->flow.begin){s->flow.begin=now;s->depth_changed=now;}
    s->flow.depth_ns[s->status.outstanding]+=now-s->depth_changed;s->depth_changed=now;
    s->flow_blocks[id].admitted=now;s->flow.admissions++;
    if(previous==s&&previous_generation==s->status.generation&&previous_id>=0){s->flow_blocks[previous_id].next_admission=now;s->flow.refills++;s->flow.refill_ns+=now-s->flow_blocks[previous_id].pread_return;}
   }
   auto transition=profiling?cpu():0;auto extent=s->layout[id];auto& info=s->info[id];info.epoch=s->status.epoch;
   if(profiling||flow)info.admitted_ns=wall();
   // Admission linearizes here, while holding the same lock as rank selection.
   s->claim(id);
   publish(s->arena.states+id,STREAM_INFLIGHT);s->status.outstanding++;s->status.admissions++;if(flow)s->flow.max_depth=std::max(s->flow.max_depth,s->status.outstanding);
   if(profiling){s->maintenance.empty_inflight_count++;s->maintenance.empty_inflight_cpu_ns+=cpu()-transition;}
   if(s->status.outstanding<config.total_qd)work.notify_one(); // chained wake, not a herd
   if(profiling){s->profile.admissions++;auto dt=cpu()-t;s->profile.scheduler_cpu_ns+=dt;s->profile.worker_cpu_ns+=dt;}
   if(flow)s->flow.scheduler_cpu_ns+=cpu()-t;
   lock.unlock();
   uint64_t c0=(profiling||flow)?cpu():0,w0=(profiling||flow)?wall():0;
   if(flow)s->flow_blocks[id].pread_start=w0;ssize_t got;uint64_t calls=0;
#ifdef LEANSTREAM_NVTX_PROFILE
   char io_range[96];std::snprintf(io_range,sizeof(io_range),"I/O pread group=%d",id);
   {LeanstreamStreamNvtxScope trace(io_range);
#endif
   do{got=::READ_BLOCK(s->fd,(char*)s->arena.base+extent.arena_offset,extent.bytes,extent.file_offset);calls++;}while(got<0&&errno==EINTR);
#ifdef LEANSTREAM_NVTX_PROFILE
   }
#endif
   int saved_errno=errno;uint64_t w1=(profiling||flow)?wall():0,c1=(profiling||flow)?cpu():0;
   if(flow)s->flow_blocks[id].pread_return=w1;
   int error=got<0?saved_errno:(got!=extent.bytes?EIO:0);
   info.bytes=got>0?got:0;info.error=error;if(profiling||flow){info.completed_ns=w1;info.ready_ns=wall();}
   if(error)publish(s->arena.states+s->count,error);
   // Existing LeanStream CPU release -> mapped system-acquire convention.
   publish(s->arena.states+id,error?STREAM_ERROR:STREAM_READY);
   if(flow)s->flow_blocks[id].ready=wall();
   auto ready_cost=profiling?cpu()-c1:0;
   uint64_t before_observer=(profiling||flow)?cpu():0;
   if(s->observer)s->observer(s->observer_arg,id,&info);
   uint64_t after_observer=(profiling||flow)?cpu():0;
   uint64_t lock_start=flow?wall():0;if(flow)s->flow_blocks[id].observer_end=lock_start;
   lock.lock();
   uint64_t lock_acquired=flow?wall():0;
   if(flow){auto& f=s->flow;f.depth_ns[s->status.outstanding]+=lock_acquired-s->depth_changed;s->depth_changed=lock_acquired;
    f.end=lock_acquired;f.lock_wait_ns+=lock_acquired-lock_start;f.syscalls+=calls;f.bytes+=info.bytes;
    f.worker_cpu_ns+=cpu()-c0;f.observer_cpu_ns+=after_observer-before_observer;
    f.completion_cpu_ns+=cpu()-c1-(after_observer-before_observer);
    s->flow_blocks[id].credit_return=lock_acquired;s->flow_blocks[id].lock_wait=lock_acquired-lock_start;
    previous=s;previous_id=id;previous_generation=s->status.generation;
   }
   s->status.outstanding--;s->status.completions++;s->status.bytes+=info.bytes;
   if(profiling&&!error){s->maintenance.ready_count++;s->maintenance.ready_cpu_ns+=ready_cost;}
   if(error)s->status.error=error;
   if(profiling){auto end=cpu();s->profile.syscalls+=calls;s->profile.requested_bytes+=extent.bytes*calls;s->profile.completed_bytes+=info.bytes;
    s->profile.pread_wall_ns+=w1-w0;s->profile.worker_cpu_ns+=end-c0;
    s->profile.completion_cpu_ns+=end-c1-(after_observer-before_observer);s->profile.observer_cpu_ns+=after_observer-before_observer;}
   if(s->debug_waiters||s->closing)changed.notify_all();
   work.notify_one(); // the completing worker also immediately refills
   if(flow)s->flow.lock_hold_ns+=wall()-lock_acquired;
  }
 }
};
extern "C" {
StreamEngine* stream_create_engine(const StreamConfig* c){
 if(!c||!c->total_qd||c->total_qd>32||!c->rank_capacity||!c->memory_alignment||!c->file_alignment){errno=EINVAL;return nullptr;}
 try{return new StreamEngine(*c);}catch(...){errno=ENOMEM;return nullptr;}
}
int stream_worker_cpus(StreamEngine* e,const int32_t* ids,int n){
 if(!e||!ids||n<=0)return EINVAL;std::lock_guard<std::mutex> l(e->mutex);
 if(e->session&&!e->session->status.ended)return EBUSY;cpu_set_t mask;CPU_ZERO(&mask);
 for(int i=0;i<n;i++){if(ids[i]<0||ids[i]>=CPU_SETSIZE)return EINVAL;CPU_SET(ids[i],&mask);}
 for(auto& t:e->workers){int rc=pthread_setaffinity_np(t.native_handle(),sizeof mask,&mask);if(rc)return rc;}return 0;
}
StreamSession* stream_begin_layer(StreamEngine* e,int fd,const StreamExtent* layout,int count,StreamArena arena,const int32_t* resident,int nr){
 if(!e||!layout||count<=0||!arena.base||!arena.states||uintptr_t(arena.states)%alignof(uint32_t)||nr<0||(nr&&!resident)){errno=EINVAL;return nullptr;}
 const bool p=e->config.profile;auto begin=p?wall():0;
 {std::lock_guard<std::mutex> l(e->mutex);if(e->session&&!e->session->status.ended){errno=EBUSY;return nullptr;}}
 int flags=fcntl(fd,F_GETFL);if(flags<0||!(flags&O_DIRECT)){errno=EINVAL;return nullptr;}
 uint64_t validation=p?wall():0;
 try{
 std::vector<std::pair<uint64_t,uint64_t>> extents;extents.reserve(count);
 for(int i=0;i<count;i++){auto x=layout[i];
  if(!x.bytes||x.arena_offset>arena.bytes||x.bytes>arena.bytes-x.arena_offset||x.file_offset>uint64_t(INT64_MAX)-x.bytes||
    (uintptr_t(arena.base)+x.arena_offset)%e->config.memory_alignment||x.file_offset%e->config.file_alignment||x.bytes%e->config.file_alignment){errno=EINVAL;return nullptr;}
  extents.emplace_back(x.arena_offset,x.arena_offset+x.bytes);
 }
 std::sort(extents.begin(),extents.end());for(int i=1;i<count;i++)if(extents[i].first<extents[i-1].second){errno=EINVAL;return nullptr;}
 for(int i=0;i<nr;i++)if(resident[i]<0||resident[i]>=count){errno=EINVAL;return nullptr;}
 uint64_t validated=p?wall():0;
  auto s=std::make_unique<StreamSession>();s->engine=e;s->fd=dup(fd);if(s->fd<0)return nullptr;s->layout=layout;s->count=count;s->arena=arena;
  for(auto& r:s->ranks)r.resize(e->config.rank_capacity);
  s->claimed_bits.resize((size_t(count)+63)/64,0);s->committed.reserve(count);s->committed_bits.resize(count);s->seen.resize(count);s->info.resize(count);if(e->config.profile>=2)s->flow_blocks.resize(count);
  if(p){s->updates.reserve(256);s->consume_states.resize(count);}
  std::fill_n(arena.states,count+1,0u);auto initialized=p?wall():0;
  for(int i=0;i<nr;i++){s->claim(resident[i]);s->info[resident[i]].bytes=layout[resident[i]].bytes;publish(arena.states+resident[i],STREAM_READY);}
  auto installed=p?wall():0;std::lock_guard<std::mutex> l(e->mutex);s->status.generation=++e->generation;
  s->profile.layout_ns=validated-validation;s->profile.state_init_ns=initialized-validated;s->profile.resident_ns=installed-initialized;s->profile.begin_ns=p?wall()-begin:0;
  e->session=std::move(s);return e->session.get();
 }catch(...){errno=ENOMEM;return nullptr;}
}
int stream_observe(StreamSession* s,StreamObserver f,void* arg){if(!s)return EINVAL;std::lock_guard<std::mutex> l(s->engine->mutex);if(s->status.epoch)return EBUSY;s->observer=f;s->observer_arg=arg;return 0;}
static int install(StreamSession* s,const int32_t* ids,int n,bool consume,StreamView* view,int32_t* output,int capacity){
#ifdef LEANSTREAM_NVTX_PROFILE
 LeanstreamStreamNvtxScope trace(consume?"I/O streaming final consume":"I/O streaming priority update");
#endif
 if(!s||n<0||(n&&!ids)||(consume&&(!view||(n&&!output)||capacity<n)))return EINVAL;
 auto& e=*s->engine;bool p=e.config.profile;if(unsigned(n)>e.config.rank_capacity)return E2BIG;
 auto c0=p?cpu():0;auto t0=p?wall():0;int inactive=1-s->active_rank;
 // Caller serializes updates/consume/end. Workers never read inactive storage.
 for(int i=0;i<n;i++){if(ids[i]<0||ids[i]>=s->count)return EINVAL;s->ranks[inactive][i]=ids[i];}
 auto t1=p?wall():0;std::unique_lock<std::mutex> l(e.mutex);auto t2=p?wall():0;
 if(s->closing||state(s->arena.states+s->count))return s->closing?ESHUTDOWN:EIO;
 const int previous_rank=s->active_rank,previous_count=s->rank_size;const uint64_t previous_epoch=s->status.epoch;
 s->active_rank=inactive;s->rank_size=n;s->cursor=0;s->fallback=0;s->status.epoch++;
 auto t3=p?wall():0;
 if(consume){
  s->consume=true;uint64_t ready=0,inflight=0,empty=0;int unique=0;
  for(int i=0;i<n;i++){int id=ids[i];if(s->seen[id])continue;s->seen[id]=1;output[unique++]=id;
   if(!s->committed_bits[id]){s->committed_bits[id]=1;s->committed.push_back(id);}
   auto v=s->claimed(id)?state(s->arena.states+id):STREAM_EMPTY;
   ready+=v==STREAM_READY;inflight+=v==STREAM_INFLIGHT;empty+=v==STREAM_EMPTY;
   if(p)s->consume_states[id]=v;
  }
  if(p){
   auto& r=s->refinement;r={};r.previous_epoch=previous_epoch;r.consume_epoch=s->status.epoch;
   // Inspect only the previous ordered rank, never the whole logical layer.
   // Required IDs reuse their classification above, avoiding contradictory samples.
   for(int i=0;i<previous_count;i++){
    int id=s->ranks[previous_rank][i];if(s->seen[id]&2)continue;
    bool useful=s->seen[id]&1;s->seen[id]|=2;r.rank_count++;r.overlap_count+=useful;
    if(s->info[id].epoch!=previous_epoch||!s->claimed(id))continue;
    auto v=useful?s->consume_states[id]:state(s->arena.states+id);
    if(v==STREAM_READY)(useful?r.completed_useful:r.completed_obsolete)++;
    if(v==STREAM_INFLIGHT)(useful?r.inflight_useful:r.inflight_obsolete)++;
   }
   for(int i=0;i<previous_count;i++)s->seen[s->ranks[previous_rank][i]]&=1;
  }
  for(int i=0;i<unique;i++)s->seen[output[i]]=0;
  auto committed=p?wall():0;*view={s->status.generation,uint32_t(unique),0,output,s->layout,s->arena};
  s->status.committed=s->committed.size();
  if(p){s->profile.commit_ns+=committed-t3;s->profile.view_ns+=wall()-committed;s->profile.required_count=unique;
   s->profile.already_ready=ready;s->profile.inflight_required=inflight;s->profile.empty_required=empty;s->profile.consume_start_ns=t0;}
 }
 e.work.notify_one();auto t4=p?wall():0;
 if(p){s->profile.update_count++;if(s->updates.size()<s->updates.capacity())s->updates.push_back({0,t1-t0,t2-t1,t4-t2,t3-t2,uint64_t(n),s->status.epoch,0});else s->profile.updates_dropped++;}
 l.unlock();
 if(p){std::lock_guard<std::mutex> finish(e.mutex);auto total=wall()-t0;if(s->profile.update_count<=s->updates.capacity()){s->updates.back().total_ns=total;s->updates.back().cpu_ns=cpu()-c0;}if(consume)s->profile.consume_ns+=total;}
 return 0;
}
int stream_update_priority(StreamSession* s,const int32_t* ids,int n){return install(s,ids,n,false,nullptr,nullptr,0);}
int stream_consume(StreamSession* s,const int32_t* ids,int n,StreamView* v,int32_t* output,int cap){return install(s,ids,n,true,v,output,cap);}
int stream_get_status(StreamSession* s,StreamStatus* out){if(!s||!out)return EINVAL;std::lock_guard<std::mutex> l(s->engine->mutex);*out=s->status;out->error=state(s->arena.states+s->count);return 0;}
int stream_get_profile(StreamSession* s,StreamProfile* out,StreamUpdateProfile* rows,int capacity){
 if(!s||!out||capacity<0)return -EINVAL;std::lock_guard<std::mutex> l(s->engine->mutex);
 if(capacity<int(s->updates.size())||(capacity&&!rows))return -ENOSPC;*out=s->profile;std::copy(s->updates.begin(),s->updates.end(),rows);return int(s->updates.size());
}
int stream_refinement_profile(StreamSession* s,StreamRefinementProfile* out){
 if(!s||!out)return EINVAL;std::lock_guard<std::mutex> l(s->engine->mutex);
 if(!s->engine->config.profile)return ENODATA;*out=s->refinement;return 0;
}
int stream_maintenance_profile(StreamSession* s,StreamMaintenanceProfile* out){
 if(!s||!out)return EINVAL;std::lock_guard<std::mutex> l(s->engine->mutex);*out=s->maintenance;return 0;
}
const StreamBlockInfo* stream_block_info(StreamSession* s,int id){if(!s||id<0||id>=s->count)return nullptr;auto v=state(s->arena.states+id);return v>=STREAM_READY?&s->info[id]:nullptr;}
int stream_wait(StreamSession* s,int timeout){
 if(!s)return EINVAL;auto& e=*s->engine;std::unique_lock<std::mutex> l(e.mutex);s->debug_waiters++;
 bool ok=e.changed.wait_for(l,std::chrono::milliseconds(timeout),[&]{if(state(s->arena.states+s->count))return true;if(!s->consume)return false;for(int id:s->committed)if(state(s->arena.states+id)!=STREAM_READY)return false;return true;});s->debug_waiters--;
 return ok?int(state(s->arena.states+s->count)):ETIMEDOUT;
}
int stream_wait_idle(StreamSession* s,int timeout){
 if(!s)return EINVAL;auto& e=*s->engine;std::unique_lock<std::mutex> l(e.mutex);s->debug_waiters++;
 bool ok=e.changed.wait_for(l,std::chrono::milliseconds(timeout),[&]{return state(s->arena.states+s->count)||(!s->status.outstanding&&s->cursor>=s->rank_size&&(!s->consume||s->fallback>=int(s->committed.size())));});s->debug_waiters--;
 return ok?int(state(s->arena.states+s->count)):ETIMEDOUT;
}
int stream_end_layer(StreamSession* s){
 if(!s)return EINVAL;auto& e=*s->engine;bool p=e.config.profile;auto c=p?cpu():0;std::unique_lock<std::mutex> l(e.mutex);
 if(s->status.ended)return s->status.error;s->closing=true;auto w=p?wall():0;
 e.changed.wait(l,[&]{return s->status.outstanding==0;});if(p){s->profile.end_wait_ns=wall()-w;s->profile.end_cpu_ns=cpu()-c;}
 s->status.ended=1;return state(s->arena.states+s->count);
}
void stream_destroy_engine(StreamEngine* e){if(!e)return;if(e->session)stream_end_layer(e->session.get());{std::lock_guard<std::mutex> l(e->mutex);e->stop=true;e->work.notify_all();}for(auto& t:e->workers)t.join();delete e;}
}

#ifdef STREAM_TESTING
extern "C" int stream_test_claimed(StreamSession* s,int id){
 if(!s||id<0||id>=s->count)return -EINVAL;
 std::lock_guard<std::mutex> lock(s->engine->mutex);return s->claimed(id);
}
#endif

extern "C" int stream_flow_profile(StreamSession* s,StreamFlowProfile* out,StreamFlowBlock* blocks,int n){
 if(!s||!out||n<s->count||!blocks)return EINVAL;
 std::lock_guard<std::mutex> l(s->engine->mutex);if(s->engine->config.profile<2)return ENODATA;
 if(s->status.outstanding)return EBUSY;
 *out=s->flow;std::copy(s->flow_blocks.begin(),s->flow_blocks.end(),blocks);return 0;
}

extern "C" int stream_speculation_ceiling(StreamSession* s,uint64_t count){if(!s)return EINVAL;auto& e=*s->engine;std::lock_guard<std::mutex> lock(e.mutex);s->speculation_ceiling=count;e.work.notify_all();return 0;}
