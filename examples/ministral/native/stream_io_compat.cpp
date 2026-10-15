// Opt-in legacy pipeline bridge, linked INSTEAD of priority_io.cpp only in v2 builds.
// GPU-resident cache entries remain resolved by the existing cache/descriptors.
// Only nonresident I/O requests enter this stable host arena; no resident payload copy.
#include "priority_io.h"
#include "leanstream_stream_io.h"
#include <unistd.h>
#include <errno.h>
#include <chrono>
#include <atomic>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <cstdlib>
#include <cstdio>
#include <fcntl.h>
static uint64_t ns(){return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}
static int native_direct_fd(int input){
 int flags=fcntl(input,F_GETFL);if(flags<0)return -1;
 if(flags&O_DIRECT)return dup(input);
 // Independent file description: do not change the Python reader's flags.
 char path[64];std::snprintf(path,sizeof(path),"/proc/self/fd/%d",input);
 return open(path,O_RDONLY|O_DIRECT);
}
struct Bridge {
 int fd=-1,n=0,bytes=0,qd=24;uint64_t stride=0,generation=0,base_generation=0,published=0;
 void* arena=nullptr;void* shared=nullptr;StreamEngine* engine=nullptr;StreamSession* session=nullptr;
 std::vector<StreamExtent> layout;std::vector<Completion> records;std::vector<int> queue;std::vector<int32_t> cpus;
 std::vector<std::atomic<uint64_t>> complete;int head=0,tail=0;bool ended=false;
 std::mutex mutex;std::condition_variable changed;PioObserver observer=nullptr;void* arg=nullptr;
 StreamView view{};
 Bridge(int input,const int64_t* offsets,int count,int b,void* a,uint64_t st,void* flags):fd(native_direct_fd(input)),n(count),bytes(b),stride(st),arena(a),shared(flags),layout(n),records(n),queue(n),complete(n){
  if(const char* q=getenv("LEANSTREAM_STREAM_QD"))qd=atoi(q);
  for(int i=0;i<n;i++){layout[i]={uint64_t(offsets[i]),uint64_t(i)*stride,uint32_t(bytes),0};complete[i]=0;}
 }
 int begin(){
  if(!engine){StreamConfig c{uint32_t(qd),16384,uint32_t(getenv("LEANSTREAM_STREAM_PROFILE")?atoi(getenv("LEANSTREAM_STREAM_PROFILE")):0),512,512};engine=stream_create_engine(&c);if(!engine)return errno;
   if(!cpus.empty()){int rc=stream_worker_cpus(engine,cpus.data(),cpus.size());if(rc)return rc;}}
  session=stream_begin_layer(engine,fd,layout.data(),n,{arena,stride*n,(uint32_t*)shared},nullptr,0);if(!session)return errno;
  head=tail=0;ended=false;base_generation=generation;for(int i=0;i<n;i++)complete[i]=0;
  return stream_observe(session,callback,this);
 }
 static void callback(void* ptr,int32_t id,const StreamBlockInfo* info){
  auto& b=*(Bridge*)ptr;Completion c{};c.group=c.slot=id;c.bytes=info->bytes;c.error=info->error;c.sequence=id+1;c.generation=b.base_generation+info->epoch;
  // Bridge clocks are the legacy CLOCK_MONOTONIC domain. Core diagnostic stamps
  // remain CLOCK_MONOTONIC_RAW and are exported separately, never mixed.
  c.complete_ns=ns();c.submit_ns=0;c.selected_ns=0;c.published_ns=0;
  if(!c.error)b.complete[id].store(c.complete_ns,std::memory_order_release);
  if(b.observer)b.observer(b.arg,&c);
  {std::lock_guard<std::mutex> l(b.mutex);b.records[id]=c;b.queue[b.tail++]=id;}b.changed.notify_one();
 }
 ~Bridge(){if(engine)stream_destroy_engine(engine);if(fd>=0)close(fd);}
};
extern "C" {
void* pio_stream_create(int fd,const int64_t* off,int n,uint64_t bytes,void* arena,uint64_t stride,void* flags){
 try{auto b=new Bridge(fd,off,n,bytes,arena,stride,flags);if(!b->shared||b->fd<0){delete b;return nullptr;}return b;}catch(...){return nullptr;}
}
int pio_enable_workers(void*,int){return 0;}
int pio_parallel(void*,int){return 0;} // fixed aggregate QD selected once by the opt-in bridge
int pio_worker_cpus(void* p,const int32_t* ids,int n){auto& b=*(Bridge*)p;if(b.engine)return EBUSY;if(n>0)b.cpus.assign(ids,ids+n);return 0;}
int pio_observe(void* p,PioObserver f,void* a){auto& b=*(Bridge*)p;b.observer=f;b.arg=a;return 0;}
static int update(void* p,const int32_t* ids,int n,uint64_t epoch,uint64_t* pub,int* active,bool final){
 auto& b=*(Bridge*)p;if(epoch<=b.generation){fprintf(stderr,"stream epoch %lu <= %lu\n",epoch,b.generation);return EINVAL;}if(!b.session){int rc=b.begin();if(rc){fprintf(stderr,"stream begin rc=%d flags=%x n=%d bytes=%d stride=%lu arena=%p\n",rc,fcntl(b.fd,F_GETFL),b.n,b.bytes,b.stride,b.arena);return rc;}}
 StreamStatus status{};stream_get_status(b.session,&status);*active=status.outstanding;
 int rc=final?stream_consume(b.session,ids,n,&b.view,(int32_t*)b.shared+b.n+1,b.n):stream_update_priority(b.session,ids,n);
 if(rc)fprintf(stderr,"stream install rc=%d n=%d groups=%d first=%d final=%d\n",rc,n,b.n,n?ids[0]:-1,final);
 if(!rc){b.generation=epoch;b.published=ns();*pub=b.published;}return rc;
}
int pio_update(void* p,const int32_t* ids,int n,uint64_t epoch,int,uint64_t* pub,int* active){return update(p,ids,n,epoch,pub,active,false);}
int pio_stream_consume(void* p,const int32_t* ids,int n,uint64_t epoch,uint64_t* pub,int* active){return update(p,ids,n,epoch,pub,active,true);}
int pio_peek(void* p,int id,int32_t* slot,uint64_t* complete){auto& b=*(Bridge*)p;if(id<0||id>=b.n)return EINVAL;*slot=id;*complete=b.complete[id].load(std::memory_order_acquire);return 0;}
int pio_poll(void* p,Completion* out,int capacity){auto& b=*(Bridge*)p;std::lock_guard<std::mutex> l(b.mutex);int n=0;while(n<capacity&&b.head<b.tail)out[n++]=b.records[b.queue[b.head++]];return n;}
int pio_release(void*,int,uint64_t){return 0;} // never evict within a session
int pio_resident(void*,const int32_t*,int){return ENOTSUP;} // GPU residency is resolved before bridge admission
int pio_stats(void* p,Stats* out){auto& b=*(Bridge*)p;StreamStatus s{};if(b.session)stream_get_status(b.session,&s);std::lock_guard<std::mutex> l(b.mutex);*out={};out->calls=s.completions;out->bytes=s.bytes;out->inflight=s.outstanding;out->queued=b.tail-b.head;out->owned=b.ended?0:s.completions;return 0;}
int pio_wait_idle(void* p,int timeout){auto& b=*(Bridge*)p;return b.session?stream_wait_idle(b.session,timeout):0;}
int pio_stream_begin_current(void* p){auto& b=*(Bridge*)p;return b.session&&!b.ended?0:b.begin();}
int pio_stream_end(void* p){auto& b=*(Bridge*)p;if(!b.session)return 0;int rc=stream_end_layer(b.session);b.ended=true;return rc;}
int pio_rebind(void* p,int fd,const int64_t* off){auto& b=*(Bridge*)p;if(!b.ended)return EBUSY;int next=native_direct_fd(fd);if(next<0)return errno;close(b.fd);b.fd=next;for(int i=0;i<b.n;i++)b.layout[i].file_offset=off[i];return b.begin();}
int pio_reset(void* p){return pio_stream_end(p);}
int pio_wait_completion_us(void* p,int us){auto& b=*(Bridge*)p;std::unique_lock<std::mutex> l(b.mutex);return b.changed.wait_for(l,std::chrono::microseconds(us),[&]{return b.tail>b.head;})?0:ETIMEDOUT;}
int pio_wait_completion(void* p,int ms){return pio_wait_completion_us(p,ms*1000);}
int pio_wait_started(void*,uint64_t,int){return ENOTSUP;}
int pio_log_enable(void*){return 0;}
int pio_log(void* p,Completion* out,int capacity){auto& b=*(Bridge*)p;std::lock_guard<std::mutex> l(b.mutex);if(capacity<b.tail)return -ENOSPC;for(int i=0;i<b.tail;i++)out[i]=b.records[b.queue[i]];return b.tail;}
int pio_window(void*,uint64_t*,uint64_t*){return ENOTSUP;}
int pio_stream_flow_profile(void* p,StreamFlowProfile* out){
 auto& b=*(Bridge*)p;if(!b.session||!out)return EINVAL;
 std::vector<StreamFlowBlock> blocks(b.n);
 return stream_flow_profile(b.session,out,blocks.data(),b.n);
}
void pio_stop(void* p){pio_stream_end(p);}
void pio_close(void* p){delete (Bridge*)p;}
int pio_stream_profile(void* p,StreamProfile* out,StreamUpdateProfile* updates,int capacity){auto& b=*(Bridge*)p;return b.session?stream_get_profile(b.session,out,updates,capacity):-ENODATA;}
int pio_stream_refinement(void* p,StreamRefinementProfile* out){auto& b=*(Bridge*)p;return b.session?stream_refinement_profile(b.session,out):ENODATA;}
int pio_stream_maintenance(void* p,StreamMaintenanceProfile* out){auto& b=*(Bridge*)p;return b.session?stream_maintenance_profile(b.session,out):ENODATA;}
const StreamView* pio_stream_view(void* p){return &((Bridge*)p)->view;}
const StreamBlockInfo* pio_stream_info(void* p,int id){auto& b=*(Bridge*)p;return b.session?stream_block_info(b.session,id):nullptr;}
}

extern "C" int pio_stream_flow(void* p,StreamFlowProfile* out,StreamFlowBlock* blocks,int count){auto& b=*(Bridge*)p;return b.session?stream_flow_profile(b.session,out,blocks,count):ENODATA;}

extern "C" int stream_speculation_ceiling(StreamSession*,uint64_t);
extern "C" int pio_stream_speculation_ceiling(void* p,uint64_t count){auto& b=*(Bridge*)p;return b.session?stream_speculation_ceiling(b.session,count):EINVAL;}
