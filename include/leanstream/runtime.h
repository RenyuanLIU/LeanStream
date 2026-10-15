#pragma once
#include "weight_manager.h"
#include <memory>
#include <vector>
#include <mutex>
#include <condition_variable>
namespace leanstream::runtime {
// Enqueues A, B, reduction once. Completion/commit callbacks never launch B.
// Even a failed enqueue may leave partial work live: complete/cancel must remain valid.
struct ComputeBackend {
    void* context=nullptr;
    int (*enqueue)(void*,uint32_t layer,const void* input,void* output,void* stream,
                   Span<ExecDesc> core,DeviceExecQueue,uint32_t epoch,ConsumerMode)=nullptr;
    bool (*complete)(void*,uint32_t layer)=nullptr;
    int (*commit)(void*,uint32_t layer,Span<GroupId>)=nullptr;
    // Optional banked production maintenance: schedule cache commit off the
    // foreground, then invoke retired ONLY after GPU/data users have finished.
    // IDs are copied into preallocated storage and survive until retired.
    int (*defer_commit)(void*,uint32_t layer,Span<GroupId>,uint32_t epoch,
                        void (*retired)(void*,uint32_t epoch),void* retirement_context)=nullptr;
    void (*cancel)(void*,uint32_t layer,uint32_t epoch)=nullptr;
};
class Runtime;
class LayerStep {
public:
    LayerStep()=default;
    int update_prediction(Span<GroupId>);
    int update_prediction(Span<GroupId>,PredictionUpdateOptions);
    int enqueue_compute(const void*,void*,void* stream);
    int finalize(Span<GroupId>);
    int commit(Span<GroupId> next_resident);
    int abort();
private:
    friend class Runtime;
    LayerStep(Runtime* r,uint32_t l,uint32_t e): runtime_(r),layer_(l),epoch_(e) {}
    Runtime* runtime_=nullptr; uint32_t layer_=0,epoch_=0;
};
class Runtime {
public:
    Runtime(ManagerConfig,ComputeBackend,uint32_t layer_count=1);
    ~Runtime(); // all steps must be committed/aborted; drains deferred retirement
    int begin_layer(uint32_t,const LayerStorage&,LayerStep*);
    WeightManager& weights(uint32_t layer=0);
    int drain(); // prompt-end/background retirement wait, never a layer timing sync
private:
    friend class LayerStep;
    struct LayerContext;
    std::vector<std::unique_ptr<LayerContext>> layers_;
    ComputeBackend backend_; ConsumerMode mode_;
    std::mutex mutex_;std::condition_variable retired_;
    static void retire_callback(void*,uint32_t epoch);
};
}
