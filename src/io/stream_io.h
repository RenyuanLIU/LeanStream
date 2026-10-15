#pragma once
#include <stdint.h>
// Generic CPU-only core. For GPU consumers, caller supplies cudaHostAllocMapped
// arena, states[count+1] and view IDs/layout, using the existing system-acquire protocol.
// states[count] is terminal errno. No payload copies or CUDA lifetime tracking.
struct StreamEngine;
struct StreamSession;
enum StreamState : uint32_t { STREAM_EMPTY, STREAM_INFLIGHT, STREAM_READY, STREAM_ERROR };
struct StreamConfig { uint32_t total_qd, rank_capacity, profile; uint64_t memory_alignment, file_alignment; };
struct StreamExtent { uint64_t file_offset, arena_offset; uint32_t bytes, reserved; };
struct StreamArena { void* base; uint64_t bytes; uint32_t* states; };
struct StreamBlockInfo { uint64_t epoch, admitted_ns, completed_ns, ready_ns, bytes; int32_t error, reserved; };
struct StreamView { uint64_t generation; uint32_t count, reserved; const int32_t* ids; const StreamExtent* layout; StreamArena arena; };
struct StreamStatus { uint64_t generation, epoch, outstanding, admissions, completions, bytes, committed; int32_t error, ended; };
struct StreamUpdateProfile { uint64_t total_ns, copy_ns, lock_wait_ns, lock_hold_ns, install_ns, length, epoch, cpu_ns; };
struct StreamProfile {
 uint64_t begin_ns,state_init_ns,resident_ns,layout_ns,update_count,updates_dropped;
 uint64_t admission_calls,candidates,skipped_ready,skipped_inflight,skipped_error,admissions,scheduler_cpu_ns;
 uint64_t syscalls,requested_bytes,completed_bytes,worker_cpu_ns,pread_wall_ns,completion_cpu_ns,observer_cpu_ns;
 uint64_t consume_ns,commit_ns,view_ns,required_count,already_ready,inflight_required,empty_required,consume_start_ns;
 uint64_t end_cpu_ns,end_wait_ns;
};
// Diagnostic snapshot of the immediately preceding rank at CONSUME.
struct StreamRefinementProfile {
 uint64_t previous_epoch,consume_epoch,rank_count,overlap_count;
 uint64_t completed_useful,completed_obsolete,inflight_useful,inflight_obsolete;
};
struct StreamMaintenanceProfile {
 uint64_t check_count,check_cpu_ns,skip_ready_count,skip_ready_cpu_ns,skip_inflight_count,skip_inflight_cpu_ns;
 uint64_t empty_inflight_count,empty_inflight_cpu_ns,ready_count,ready_cpu_ns;
};
using StreamObserver=void(*)(void*,int32_t,const StreamBlockInfo*);
extern "C" {
StreamEngine* stream_create_engine(const StreamConfig*);
int stream_worker_cpus(StreamEngine*,const int32_t*,int); // quiescent only
// Caller serializes begin/update/consume/end operations; status/debug waits may be concurrent.
// Layout must be immutable until end_layer. Resident IDs mean valid bytes ALREADY
// at layout[id].arena_offset; externally resident GPU caches are resolved by integration.
StreamSession* stream_begin_layer(StreamEngine*,int fd,const StreamExtent*,int count,StreamArena,const int32_t* resident,int resident_count);
int stream_observe(StreamSession*,StreamObserver,void*); // before first update
int stream_update_priority(StreamSession*,const int32_t*,int);
// view_ids is caller-owned output (capacity >= n), immutable until its consumers finish.
// Repeated consume calls add commitments; each returned view describes that call's set.
int stream_consume(StreamSession*,const int32_t*,int,StreamView*,int32_t* view_ids,int capacity);
int stream_get_status(StreamSession*,StreamStatus*);
int stream_get_profile(StreamSession*,StreamProfile*,StreamUpdateProfile*,int capacity); // row count, or negative errno
int stream_refinement_profile(StreamSession*,StreamRefinementProfile*); // profiling only
int stream_maintenance_profile(StreamSession*,StreamMaintenanceProfile*);
const StreamBlockInfo* stream_block_info(StreamSession*,int); // acquire READY/ERROR before reading
int stream_wait(StreamSession*,int timeout_ms); // tests/debug only; waits all committed IDs
int stream_wait_idle(StreamSession*,int timeout_ms); // debug/background quiescence, not retirement
int stream_end_layer(StreamSession*); // stops admissions, drains I/O; DOES NOT wait for GPU
// Core metadata survives end until next begin/destroy. Caller-owned view storage
// (layout, IDs, arena) must remain alive for any post-end inspection.
void stream_destroy_engine(StreamEngine*);
}
// Optional LIGHT (config.profile=2) / TRACE (3) data, separate from legacy ABI.
struct StreamFlowBlock {uint64_t admitted,pread_start,pread_return,ready,observer_end,credit_return,next_admission,lock_wait;};
struct StreamFlowProfile {
 uint64_t begin,end,depth_ns[33],max_depth,scheduler_cpu_ns,worker_cpu_ns,observer_cpu_ns;
 uint64_t candidates,skipped_ready,skipped_inflight,skipped_error,admissions,syscalls,bytes;
 uint64_t completion_cpu_ns,lock_wait_ns,lock_hold_ns,refills,refill_ns,notify_ns;
};
extern "C" int stream_flow_profile(StreamSession*,StreamFlowProfile*,StreamFlowBlock*,int);

// Optional aggregate speculative admission ceiling; required consume is never capped.
extern "C" int stream_speculation_ceiling(StreamSession*,uint64_t);
