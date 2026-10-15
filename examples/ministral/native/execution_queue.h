#pragma once
#include <array>
#include <algorithm>
#include <cstdint>
#include <mutex>
#include <time.h>
#include "leanstream_control.h"
namespace leanstream {
// CPU-only final-index/dedup state. Descriptors and release flags are supplied
// by the caller in cudaHostAllocMapped memory. No cache or staging ownership.
struct ExecMetric {uint64_t count=0,cpu_ns=0,samples=0;};
inline thread_local unsigned exec_sample_weight=1;
inline thread_local bool exec_sample_on=true;
inline thread_local uint64_t exec_sample_calls=0;
struct ExecSample {
 bool old_on=true,enabled=false;unsigned old_weight=1;
 ExecSample(unsigned position,bool measure):enabled(measure){if(enabled){old_on=exec_sample_on;old_weight=exec_sample_weight;if(old_weight==1){exec_sample_on=old_on&&position%16==0;exec_sample_weight=16;}}}
 ~ExecSample(){if(enabled){exec_sample_on=old_on;exec_sample_weight=old_weight;}}
};
enum ExecOp {CONFIRM,ELIGIBLE,DEDUP,PREFIX,IMMEDIATE,COMPLETION,LOCK_WAIT,LOCK_HOLD,DESCRIPTOR,BOUNDARY,RELEASE,APPEND,EXEC_OPS};
struct ExecChunk {uint64_t first_payload=0,last_payload=0,last_append=0,release=0;};
struct ExecInterval {uint64_t begin=0,end=0,cpu_ns=0,kind=0;};
struct ExecProfile {
 std::array<ExecMetric,EXEC_OPS> ops{};
 std::array<ExecChunk,FILL> chunks{};
 std::array<ExecInterval,2048> intervals{};
 uint64_t interval_count=0,interval_dropped=0,prefix_begin=0,prefix_end=0,group_cpu_ns=0,queue_cpu_ns=0;
};
inline uint64_t exec_clock(clockid_t which=CLOCK_MONOTONIC){timespec t;clock_gettime(which,&t);return uint64_t(t.tv_sec)*1000000000+t.tv_nsec;}
class ExecutionQueue {
 struct Final {int g=-1,physical=0,dest=-1,kind=0,missing=-1;bool enqueued=false;};
 std::array<Final,FILL> final{};
 // Final membership only. Persistent residency and I/O readiness stay authoritative elsewhere.
 int32_t* index=nullptr;int32_t* queue=nullptr;uint32_t* flags=nullptr;uint64_t* visible_tail=nullptr;
 int core=0,fill=0,groups=0,chunk=0,tail=0;uint32_t epoch=0;bool armed=false,profile=false;
 uint64_t epoch_start=0,timer_cost=0;std::mutex mutex;ExecProfile stats{};
 ExecMetric next_prefix{};uint64_t prefix_begin=0,prefix_end=0;
 uint64_t tick() const {return profile&&exec_sample_on?exec_clock(CLOCK_THREAD_CPUTIME_ID):0;}
 uint64_t elapsed(uint64_t start)const {if(!profile||!exec_sample_on)return 0;auto dt=tick()-start;return dt>timer_cost?dt-timer_cost:0;}
 void metric(ExecOp op,uint64_t start,uint64_t count=1){if(profile){stats.ops[op].count+=count;if(exec_sample_on){stats.ops[op].samples+=count;stats.ops[op].cpu_ns+=elapsed(start)*exec_sample_weight;}}}
 void interval(uint64_t start,uint64_t c,ExecOp kind){if(profile){auto end=exec_clock(),cost=elapsed(c)*exec_sample_weight;if(stats.interval_count<stats.intervals.size())stats.intervals[stats.interval_count++]={start,end,cost,uint64_t(kind)};else stats.interval_dropped++;}}
 bool append(int j,int source,uint64_t payload,int32_t* meta){
  auto t=tick();auto& f=final[j];bool duplicate=f.enqueued;metric(DEDUP,t);
  if(duplicate)return false;
  t=tick();int position=core+tail;int32_t* d=queue+5*position;
  d[0]=f.g;d[1]=f.physical;d[2]=source;d[3]=f.dest;d[4]=f.kind;
  if(f.missing>=0){meta[3*f.missing]=source;meta[3*f.missing+1]=f.physical;meta[3*f.missing+2]=f.dest;}
  f.enqueued=true;metric(DESCRIPTOR,t);
  t=tick();int k=tail/chunk;++tail;
  if(profile){auto& c=stats.chunks[k];if(!c.first_payload)c.first_payload=payload;c.first_payload=std::min(c.first_payload,payload);c.last_payload=std::max(c.last_payload,payload);c.last_append=exec_clock();stats.ops[APPEND].count++;}
  if(visible_tail)__atomic_store_n(visible_tail,(uint64_t(epoch)<<32)|uint32_t(tail),__ATOMIC_RELEASE);
  bool publish=tail%chunk==0||tail==fill;metric(BOUNDARY,t);
  if(publish){t=tick();if(profile)stats.chunks[k].release=exec_clock();__atomic_store_n(flags+k,epoch,__ATOMIC_RELEASE);metric(RELEASE,t);}
  return tail==fill;
 }
public:
 void bind_tail(uint64_t* ptr){visible_tail=ptr;__atomic_store_n(ptr,uint64_t(0),__ATOMIC_RELEASE);} // initialization only; cudaHostAllocMapped
 void bind(int32_t* q,uint32_t* f,int32_t* final_index,int g,int keep,int count,int chunk_size,bool prof){
  queue=q;flags=f;index=final_index;groups=g;core=keep;fill=count;chunk=chunk_size;profile=prof;
  if(profile){timer_cost=UINT64_MAX;for(int i=0;i<64;i++){auto t=exec_clock(CLOCK_THREAD_CPUTIME_ID);timer_cost=std::min(timer_cost,exec_clock(CLOCK_THREAD_CPUTIME_ID)-t);}}
 }
 void disarm(){std::lock_guard<std::mutex> l(mutex);armed=false;if(profile){stats={};stats.ops[PREFIX]=next_prefix;stats.prefix_begin=prefix_begin;stats.prefix_end=prefix_end;stats.queue_cpu_ns=next_prefix.cpu_ns;}}
 void start(uint32_t e){std::lock_guard<std::mutex> l(mutex);armed=false;epoch=e;tail=0;if(!profile)stats={};epoch_start=profile?exec_clock():0;
  if(profile&&!stats.ops[PREFIX].count){stats.ops[PREFIX]=next_prefix;stats.prefix_begin=prefix_begin;stats.prefix_end=prefix_end;stats.queue_cpu_ns+=next_prefix.cpu_ns;}
 }
 void build_prefix(const CacheState& cache,const int64_t* offsets,int bytes){
  if(profile){exec_sample_on=true;exec_sample_weight=1;}
  auto c=tick();prefix_begin=profile?exec_clock():0;
  for(int i=0;i<core;i++){int g=cache.view(2)[i],slot=cache.view(1)[g];int32_t* d=queue+5*i;d[0]=g;d[1]=int(offsets[g]/bytes);d[2]=slot;d[3]=slot;d[4]=0;}
  if(profile){prefix_end=exec_clock();next_prefix={uint64_t(core),elapsed(c),uint64_t(core)};}
 }
 // Caller has finalized CachePlan before this call. peek() performs an acquire
 // of existing I/O completion truth. Callback and reconciliation serialize here.
 template<class Peek> bool confirm(const CacheState& cache,const int32_t* fill_ids,const int64_t* offsets,int bytes,int32_t* meta,Peek peek,bool reconcile=true){
  if(profile){exec_sample_on=true;exec_sample_weight=1;}
  auto root=tick(),w=profile?exec_clock():0,wait=tick();std::unique_lock<std::mutex> lock(mutex);metric(LOCK_WAIT,wait);auto hold=tick();
  auto before=stats.ops[ELIGIBLE].cpu_ns+stats.ops[DEDUP].cpu_ns+stats.ops[CONFIRM].cpu_ns;
  auto c=tick();std::fill_n(index,groups,-1);const auto& plan=cache.plan();
  for(int j=0;j<fill;j++){int g=fill_ids[j],slot=cache.view(1)[g];index[g]=j;final[j]={g,int(offsets[g]/bytes),slot,slot<0?1:0,-1,false};meta[3*j+2]=-1;}
  for(int j=0;j<plan.missing_count;j++){auto& f=final[index[plan.missing[j]]];f.dest=plan.destinations[j];f.missing=j;}
  armed=true;metric(CONFIRM,c,fill);bool finished=false;
  for(int j=0;reconcile&&j<fill;j++){ExecSample sample(j,profile);c=tick();auto& f=final[j];int source=f.dest;uint64_t ready=epoch_start;bool available=f.kind==0;
   if(!available){peek(f.g,source,ready);available=ready!=0;}metric(ELIGIBLE,c);
   if(available)finished=append(j,source,ready,meta)||finished;
  }
  auto group=stats.ops[ELIGIBLE].cpu_ns+stats.ops[DEDUP].cpu_ns+stats.ops[CONFIRM].cpu_ns-before;
  if(profile){stats.group_cpu_ns+=group;auto total=elapsed(root);stats.queue_cpu_ns+=total>group?total-group:0;}
  metric(LOCK_HOLD,hold);metric(IMMEDIATE,root);interval(w,root,IMMEDIATE);return finished;
 }
 bool complete(int group,int slot,uint64_t ready,int32_t* meta){
  if(profile){exec_sample_on=(exec_sample_calls++%16)==0;exec_sample_weight=16;}
  auto root=tick(),w=profile?exec_clock():0,wait=tick();std::unique_lock<std::mutex> lock(mutex);metric(LOCK_WAIT,wait);auto hold=tick();
  auto before=stats.ops[ELIGIBLE].cpu_ns+stats.ops[DEDUP].cpu_ns;
  auto c=tick();int j=armed?index[group]:-1;metric(ELIGIBLE,c);
  bool finished=j>=0&&append(j,slot,ready,meta);
  auto used=stats.ops[ELIGIBLE].cpu_ns+stats.ops[DEDUP].cpu_ns-before;
  if(profile){stats.group_cpu_ns+=used;auto total=elapsed(root)*exec_sample_weight;stats.queue_cpu_ns+=total>used?total-used:0;}
  metric(LOCK_HOLD,hold);metric(COMPLETION,root);interval(w,root,COMPLETION);return finished;
 }
 int size(){std::lock_guard<std::mutex> l(mutex);return core+tail;}
 const ExecProfile* snapshot()const{return &stats;} // quiescent diagnostic use only
};
}
