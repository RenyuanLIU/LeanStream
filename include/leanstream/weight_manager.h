#pragma once
#include "execution_queue.h"
#include "group_layout.h"
#include <memory>
namespace leanstream::runtime {
// Optional borrowed production transport. Calls are once per update/phase,
// never virtual dispatch or heap callbacks per group. Backend retains slot/ABA
// ownership and drains every callback before retire returns.
struct IOCompletion {
    uint32_t epoch; GroupId logical_group; SlotId staging_slot;
    int error; uint64_t slot_sequence;
};
using IOCompletionCallback=void (*)(void*,const IOCompletion&);
struct IOBackend {
    void* context=nullptr;
    int (*begin)(void*,IOCompletionCallback,void*,uint32_t epoch)=nullptr;
    int (*update)(void*,Span<GroupId>)=nullptr;
    int (*consume)(void*,Span<GroupId>)=nullptr;
    int (*retire)(void*)=nullptr;
    SlotId (*ready_slot)(void*,GroupId)=nullptr;
    uint64_t (*requests)(void*)=nullptr;
    int (*wait)(void*,int timeout_ms)=nullptr;
};
struct ManagerConfig {
    uint32_t group_count, qd=24;
    uint64_t group_bytes, staging_stride, alignment=512;
    ConsumerMode mode=ConsumerMode::BatchCompat;
    IOBackend io_backend{}; // empty selects the owned production stream_io engine
};
struct LayerStorage {
    int fd=-1;
    Span<GroupLayout> layout;
    const SlotId* cache_slots=nullptr; // immutable through this step
    void* staging=nullptr;
    uint32_t* io_states=nullptr; // group_count + terminal error
    Span<GroupId> core;
    DeviceExecQueue queue;
    ExecDesc* mapped_core_descriptors=nullptr; // optional zero-copy GPU-visible core metadata
    const SlotId* destination_slots=nullptr; // optional application layout, indexed by logical ID
    uint64_t layout_generation=0; // nonzero: immutable layout registration; bump on any change
    uint64_t plan_generation=0; // optional prepared cache/core plan; bump when bindings change
    uint64_t staging_bytes=0; // zero: caller guarantees group_count*staging_stride; nonzero validates capacity
};
struct PredictionUpdateOptions {
    uint32_t io_budget=UINT32_MAX; // maximum new EMPTY candidates; cache/READY/INFLIGHT do not spend it
    uint32_t execution_epoch=0; // zero: current epoch
    uint32_t diagnostic_point=0; // opaque orchestration label
};
class WeightManager {
public:
    explicit WeightManager(ManagerConfig);
    ~WeightManager();
    WeightManager(const WeightManager&)=delete;
    WeightManager& operator=(const WeightManager&)=delete;
    int prepare_layer(const LayerStorage&); // optional idle/background metadata preparation; no I/O or epoch
    int begin_layer(uint32_t layer, const LayerStorage&);
    int update_prediction(Span<GroupId>);
    int update_prediction(Span<GroupId>,PredictionUpdateOptions);
    int finalize(Span<GroupId> final_groups);
    // Caller has proved GPU completion; stops admissions and drains callbacks.
    int retire();
    void cancel(int error=0); // terminal publication; caller still must prove GPU completion
    GroupState group_state(GroupId) const;
    uint64_t prediction_generation() const;
    uint32_t epoch() const;
    DeviceExecQueue queue() const;
    Span<ExecDesc> core_descriptors() const;
    int wait_io(int timeout_ms); // diagnostics/tests, not enqueue/finalize
    uint64_t read_requests() const;
private:
    struct Impl; std::unique_ptr<Impl> impl_;
};
}
