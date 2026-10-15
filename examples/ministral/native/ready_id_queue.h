#pragma once
// CPU-only handoff. GPU descriptors/flags remain caller-owned mapped memory.
#include <atomic>
#include <array>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
namespace leanstream {
uint64_t rid_wall();
uint64_t rid_cpu();
enum ReadyMode { RID_CAS_INLINE=1, RID_ID_ONLY=2, RID_ORGANIZER=3 };
struct ReadyTrace {
 uint64_t epoch=0,ready=0,published=0,popped=0,descriptor=0,visible=0;
 uint64_t queue_wait=0,push_cpu=0,organizer_cpu=0;uint32_t origin=0,reserved=0;
};
struct ReadyStats {
 uint64_t attempts=0,cas_success=0,cas_failure=0,not_required=0,stale=0;
 uint64_t pushed=0,popped=0,high_water=0,producer_waits=0,push_cpu_ns=0;
 uint64_t organizer_cpu_ns=0,queue_wait_ns=0,consumer_waits=0,wake_ns=0;
 uint64_t callback_count=0,epoch=0,pending=0;
};
class ReadyClient;
class ReadyOrganizer {
 struct Item {ReadyClient* owner;uint64_t epoch;int id;};
 std::vector<Item> ring;size_t head=0,count=0;bool stopping=false,paused=false;
 std::mutex mutex;std::condition_variable available,space;
 std::thread worker;void run();
 Item pop_locked(){auto item=ring[head];head=(head+1)%ring.size();--count;return item;}
public:
 explicit ReadyOrganizer(size_t capacity=32768);
 ~ReadyOrganizer();
 ReadyOrganizer(const ReadyOrganizer&)=delete;
 void push(ReadyClient*,int,uint64_t);
 int affinity(const int32_t* cpus,int count);
 int affinity(int cpu); // -1: inherit/caller-selected CPU set
 void measure_queue(unsigned n,unsigned producers,double* ns); // bounded batched diagnostic
 void test_pause(bool); // tests only; no hot-path branch beyond wait predicate
};
class ReadyClient {
 friend class ReadyOrganizer;
 struct Tag {std::atomic<uint64_t> required{0},published{0};};
 std::unique_ptr<Tag[]> tags;int groups;ReadyMode mode;unsigned profile;
 ReadyOrganizer& organizer;void* context;
 using Callback=void(*)(void*,int,uint64_t);
 Callback callback;
 std::atomic<uint64_t> active{0},pending{0};
 std::mutex idle_mutex;std::condition_variable idle;
 std::array<std::atomic<uint64_t>,17> counters{};
 std::vector<ReadyTrace> traces;uint64_t visible_ns=0;
 void finish(int,uint64_t,uint64_t);
 void count(int i,uint64_t n=1){if(profile)counters[i].fetch_add(n,std::memory_order_relaxed);}
public:
 ReadyClient(ReadyOrganizer&,int,ReadyMode,unsigned,void*,Callback);
 ~ReadyClient();
 // Caller must first quiesce old I/O. Queue consumers are drained here off the GPU thread.
 void begin(uint64_t epoch);
 void wait_idle();
 void require(const int32_t* ids,int n,uint64_t epoch);
 // Caller already acquired authoritative availability. Completion needs membership;
 // scanner has already selected final IDs, but both use the same CAS linearization.
 bool publish(int id,uint64_t epoch,uint64_t ready_ns=0,unsigned origin=0);
 uint64_t epoch()const{return active.load(std::memory_order_acquire);}
 ReadyStats snapshot()const;
 void mark_visible(){if(profile==2)visible_ns=rid_wall();}
 uint64_t final_visible()const{return visible_ns;}
 const ReadyTrace* trace(){return traces.data();} // quiescent only
};
}
