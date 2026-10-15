#pragma once
#include <cstdint>
struct Completion {
    int32_t group,slot,error,reserved;
    uint64_t bytes,generation,sequence,published_ns,selected_ns,submit_ns,complete_ns;
};
struct Stats {uint64_t calls,bytes,updates,updates_active,idle_ns,intentional_idle_ns,inflight,queued,owned;};
struct StreamFlowProfile;
extern "C" {
using PioObserver=void(*)(void*,const Completion*);
int pio_enable_workers(void*,int); // Initialization only; default remains one worker.
int pio_parallel(void*,int); // Target sliding window; rebind resets QD1 until runtime selects its phase window.
int pio_observe(void*,PioObserver,void*); // Quiescent initialization; callback before queue accounting, outside scheduler lock.
void* pio_create(int,const int64_t*,int,uint64_t,void*,int,uint64_t,int);
int pio_reset(void*);
int pio_worker_cpus(void*,const int32_t*,int); // Initialization-only, caller supplied permitted CPU set.
int pio_resident(void*,const int32_t*,int);
int pio_update(void*,const int32_t*,int,uint64_t,int,uint64_t*,int*);
// Acquire-visible snapshots; caller retains slot ownership until its GPU consumer completes.
// complete_ns is published immediately after successful pread, before queue accounting.
int pio_peek(void*,int group,int32_t* slot,uint64_t* complete_ns);
int pio_poll(void*,Completion*,int);
int pio_release(void*,int,uint64_t);
int pio_stats(void*,Stats*);
int pio_log_enable(void*);
int pio_log(void*,Completion*,int);
int pio_window(void*,uint64_t* elapsed_ns,uint64_t* depth_histogram_ns_33);
int pio_stream_flow_profile(void*,StreamFlowProfile*); // quiescent diagnostic snapshot
// Only changes source when idle, with no owned slots/completions/pending requests.
int pio_rebind(void*,int,const int64_t*);
// Blocking native control waits: the worker's pread never holds this mutex.
int pio_wait_completion(void*,int timeout_ms);
int pio_wait_completion_us(void*,int timeout_us);
int pio_wait_started(void*,uint64_t previous_calls,int timeout_ms);
int pio_wait_idle(void*,int timeout_ms);
void pio_stop(void*);
void pio_close(void*);
}
