#pragma once
#include "types.h"
namespace leanstream::runtime {
// Caller supplies coherent mapped storage. No allocation in publication.
struct alignas(64) QueueControl {
    uint32_t ready_count=0, reserved=0, total_count=0, epoch=0;
    uint32_t batch_epoch=0, error=0, consumer_path=0, init_arrived=0, init_epoch=0;
    alignas(64) uint32_t claim_count=0; // GPU writers never share CPU ready-count cache line
};
static_assert(offsetof(QueueControl,epoch)==12);
static_assert(offsetof(QueueControl,claim_count)==64);
static_assert(sizeof(QueueControl)==128);
struct DeviceExecQueue {
    ExecDesc* descriptors=nullptr;
    uint32_t* ready_indices=nullptr;
    QueueControl* control=nullptr;
    uint32_t capacity=0;
};
class ExecutionQueue {
public:
    explicit ExecutionQueue(DeviceExecQueue view={}): view_(view) {}
    void bind(DeviceExecQueue view) { view_=view; } // quiescent only
    DeviceExecQueue device_view() const { return view_; }
    void reset(uint32_t epoch); // only after previous GPU and callbacks retire
    void prepare(uint32_t total); // unpublished plan, caller serializes producers
    void append_unpublished(uint32_t index);
    void activate(); // publish initial prefix and stable plan epoch once
    void seal(uint32_t total);
    void publish_all(); // sealed, all-ready canonical descriptor order
    void publish(uint32_t index); // serialized producers; descriptor/data already visible
    void fail(uint32_t error);
private:
    DeviceExecQueue view_; uint32_t tail_=0, epoch_=0, total_=0;
};
}
