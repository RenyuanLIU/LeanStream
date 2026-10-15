#include "ready_id_queue.h"
#include <algorithm>
#include <stdexcept>
#include <pthread.h>
#include <sched.h>
#include <time.h>
#include <cerrno>
namespace leanstream {
uint64_t rid_wall(){timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return uint64_t(t.tv_sec)*1000000000+t.tv_nsec;}
uint64_t rid_cpu(){timespec t;clock_gettime(CLOCK_THREAD_CPUTIME_ID,&t);return uint64_t(t.tv_sec)*1000000000+t.tv_nsec;}
ReadyOrganizer::ReadyOrganizer(size_t cap):ring(cap){if(!cap)throw std::invalid_argument("ReadyID capacity");worker=std::thread([this]{run();});}
ReadyOrganizer::~ReadyOrganizer(){{std::lock_guard<std::mutex> l(mutex);stopping=true;paused=false;}available.notify_one();worker.join();}
int ReadyOrganizer::affinity(int cpu){if(cpu<0)return 0;if(cpu>=CPU_SETSIZE)return EINVAL;cpu_set_t mask;CPU_ZERO(&mask);CPU_SET(cpu,&mask);return pthread_setaffinity_np(worker.native_handle(),sizeof mask,&mask);}
int ReadyOrganizer::affinity(const int32_t* cpus,int count){cpu_set_t mask;CPU_ZERO(&mask);for(int i=0;i<count;i++){if(cpus[i]<0||cpus[i]>=CPU_SETSIZE)return EINVAL;CPU_SET(cpus[i],&mask);}return pthread_setaffinity_np(worker.native_handle(),sizeof mask,&mask);}
void ReadyOrganizer::test_pause(bool p){{std::lock_guard<std::mutex> l(mutex);paused=p;}available.notify_one();}
void ReadyOrganizer::push(ReadyClient* c,int id,uint64_t epoch){
 auto t=c->profile?rid_cpu():0,w=c->profile?rid_wall():0;
 std::unique_lock<std::mutex> l(mutex);
 if(count==ring.size()){c->count(9);space.wait(l,[&]{return count<ring.size();});}
 c->count(12,c->profile?rid_wall()-w:0);
 // The queue mutex orders publication and the consumer. No descriptor work here.
 ring[(head+count)%ring.size()]={c,epoch,id};++count;c->count(5);
 if(c->profile){auto old=c->counters[8].load();while(old<count&&!c->counters[8].compare_exchange_weak(old,count)){} }
 if(c->profile==2)c->traces[id].published=rid_wall();
 c->count(10,c->profile?rid_cpu()-t:0);
 l.unlock();available.notify_one();
}
void ReadyOrganizer::run(){
 for(;;){Item item;uint64_t pop=0;
  {std::unique_lock<std::mutex> l(mutex);available.wait(l,[&]{return stopping||(!paused&&count);});
   if(stopping&&!count)return;if(paused)continue;item=pop_locked();
   if(item.owner->profile)pop=rid_wall();}
  space.notify_one();item.owner->finish(item.id,item.epoch,pop);
 }
}
ReadyClient::ReadyClient(ReadyOrganizer& o,int n,ReadyMode m,unsigned p,void* c,Callback f):tags(new Tag[n]),groups(n),mode(m),profile(p),organizer(o),context(c),callback(f),traces(p==2?n:0){if(n<=0||p>2)throw std::invalid_argument("ReadyID config");}
ReadyClient::~ReadyClient(){wait_idle();}
void ReadyClient::wait_idle(){std::unique_lock<std::mutex> l(idle_mutex);idle.wait(l,[&]{return pending.load(std::memory_order_acquire)==0;});}
void ReadyClient::begin(uint64_t e){
 if(!e||e<=active.load())throw std::invalid_argument("ReadyID epoch must increase");
 wait_idle();visible_ns=0;for(auto& x:counters)x=0;if(profile==2)std::fill(traces.begin(),traces.end(),ReadyTrace{});
 active.store(e,std::memory_order_release);
}
void ReadyClient::require(const int32_t* ids,int n,uint64_t e){
 if(e!=epoch())throw std::invalid_argument("ReadyID wrong final epoch");
 for(int i=0;i<n;i++)if(ids[i]<0||ids[i]>=groups)throw std::out_of_range("ReadyID required ID");
 for(int i=0;i<n;i++)tags[ids[i]].required.store(e,std::memory_order_seq_cst);
 // Together with completion's fence, prevents both sides missing each other's
 // READY / final-required publication (store-buffering race on ARM).
 std::atomic_thread_fence(std::memory_order_seq_cst);
}
bool ReadyClient::publish(int id,uint64_t e,uint64_t ready,unsigned origin){
 count(0);if(id<0||id>=groups||e!=epoch()){count(4);return false;}
 std::atomic_thread_fence(std::memory_order_seq_cst);
 if(tags[id].required.load(std::memory_order_seq_cst)!=e){count(3);return false;}
 auto old=tags[id].published.load(std::memory_order_relaxed);
 for(;;){if(old==e){count(2);return false;}if(tags[id].published.compare_exchange_weak(old,e,std::memory_order_acq_rel,std::memory_order_relaxed))break;}
 count(1);pending.fetch_add(1,std::memory_order_relaxed);
 if(profile==2){auto& t=traces[id];t.epoch=e;t.ready=ready;t.origin=origin;}
 if(mode==RID_CAS_INLINE){if(profile==2)traces[id].published=rid_wall();finish(id,e,profile?rid_wall():0);}
 else organizer.push(this,id,e);
 return true;
}
void ReadyClient::finish(int id,uint64_t e,uint64_t pop){
 auto cpu=profile?rid_cpu():0;count(6);
 if(e==epoch()){
  if(profile==2){auto& t=traces[id];t.popped=pop;t.queue_wait=pop-t.published;}
  if(mode!=RID_ID_ONLY&&callback){callback(context,id,e);count(14);}
  if(profile==2){auto& t=traces[id];t.descriptor=rid_wall();t.visible=t.descriptor;t.organizer_cpu=rid_cpu()-cpu;}
 }else count(4);
 count(11,profile?rid_cpu()-cpu:0);
 // Lifetime waiters are background/test code, never required for online GPU readiness.
 {std::lock_guard<std::mutex> l(idle_mutex);if(pending.fetch_sub(1,std::memory_order_release)==1)idle.notify_all();}
}
ReadyStats ReadyClient::snapshot()const{
 ReadyStats s{};s.attempts=counters[0];s.cas_success=counters[1];s.cas_failure=counters[2];s.not_required=counters[3];s.stale=counters[4];s.pushed=counters[5];s.popped=counters[6];s.high_water=counters[8];s.producer_waits=counters[9];s.push_cpu_ns=counters[10];s.organizer_cpu_ns=counters[11];s.queue_wait_ns=counters[12];s.callback_count=counters[14];s.epoch=epoch();s.pending=pending;return s;
}
}

namespace leanstream {
void ReadyOrganizer::measure_queue(unsigned n,unsigned producers,double* ns){
 if(n>ring.size()||!producers)throw std::invalid_argument("queue microbatch");
 ReadyClient client(*this,n,RID_ID_ONLY,0,nullptr,nullptr);client.begin(1);client.pending=n;test_pause(true);
 std::vector<uint64_t> times(producers);std::vector<std::thread> threads;std::atomic<unsigned> arrived{0};
 for(unsigned w=0;w<producers;w++)threads.emplace_back([&,w]{
  auto start=rid_cpu();for(unsigned j=w;j<n;j+=producers)asm volatile(""::"r"(j):"memory");auto empty=rid_cpu()-start;
  arrived++;while(arrived<producers)std::this_thread::yield();start=rid_cpu();
  for(unsigned j=w;j<n;j+=producers)push(&client,j,1);
  times[w]=rid_cpu()-start;times[w]=times[w]>empty?times[w]-empty:0;
 });
 for(auto& t:threads)t.join();ns[0]=0;for(auto t:times)ns[0]+=double(t)/n;
 auto start=rid_cpu();for(unsigned j=0;j<n;j++)asm volatile(""::"r"(j):"memory");auto empty=rid_cpu()-start;
 start=rid_cpu();for(unsigned j=0;j<n;j++){{std::lock_guard<std::mutex> l(mutex);auto x=pop_locked();asm volatile(""::"r"(x.id):"memory");}space.notify_one();}
 ns[1]=(double(rid_cpu()-start)-empty)/n;client.pending=0;test_pause(false);
}
}
