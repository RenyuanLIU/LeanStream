#pragma once
#include "triton_backend.h"
#include "runtime.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <array>
namespace leanstream::runtime {
// Bound at initialization from an already compiled Triton specialization.
struct TritonKernel {
    CUfunction function=nullptr;
    uint32_t grid=0, warps=4, shared_bytes=0;
    int count=0, epoch_index=-1, claim_index=-1;
    std::array<uint64_t,32> values{};
    std::array<void*,32> arguments{};
    int bind(CUfunction,uint32_t grid,uint32_t warps,uint32_t shared,
             const uint64_t*,int count,int epoch_index,int claim_index);
    int launch(CUstream,uint32_t epoch,uint32_t claim);
};
// Caller owns buffers, compiled kernel handles and completion lifetime.
// No allocations, GPU queries, profiling or specialization occur in enqueue.
class TritonBackend {
public:
    explicit TritonBackend(KernelDecisionTable table={}): selector_(triton_selector(table)) {}
    TritonBackend(const TritonBackend&)=delete;
    TritonBackend& operator=(const TritonBackend&)=delete;
    void seed_claim(uint32_t initial) noexcept { if(initial>claim_)claim_=initial; }
    int bind(uint32_t candidate,uint32_t stage,CUfunction,uint32_t grid,uint32_t warps,
             uint32_t shared,const uint64_t*,int count,int epoch_index,int claim_index);
    int enqueue(KernelRuntimeState,uint32_t epoch,cudaStream_t);
    // Optional orchestration at the original A -> B -> arm -> event -> reduction boundary.
    void* arm_context=nullptr;
    int (*arm_final_prediction)(void*)=nullptr;
    cudaEvent_t b_done=nullptr; // borrowed, timing disabled, used only for lifetime
    // Completion and cache installation belong to the application, after reduction.
    void* lifecycle_context=nullptr;
    bool (*completed)(void*,uint32_t)=nullptr;
    int (*install_cache)(void*,uint32_t,Span<GroupId>)=nullptr;
    KernelRuntimeState state{}; // latest available caller state, set before enqueue
    ComputeBackend view();
private:
    KernelSelector selector_;
    TritonKernel core_,reduction_, consumers_[4];
    uint32_t claim_=0;
    static int submit(void*,uint32_t,const void*,void*,void*,Span<ExecDesc>,DeviceExecQueue,uint32_t,ConsumerMode);
    static bool complete(void*,uint32_t);
    static int commit(void*,uint32_t,Span<GroupId>);
};
}
extern "C" {
void* ls_triton_create();
void* ls_triton_create_table(const uint8_t*,uint32_t layers,uint32_t buckets);
void ls_triton_destroy(void*);
void ls_triton_seed_claim(void*,uint32_t);
int ls_triton_bind(void*,uint32_t candidate,uint32_t stage,void* function,uint32_t grid,
    uint32_t warps,uint32_t shared,const uint64_t*,int count,int epoch_index,int claim_index);
int ls_triton_enqueue(void*,uint32_t layer,uint32_t bucket,uint32_t epoch,void* stream);
}
