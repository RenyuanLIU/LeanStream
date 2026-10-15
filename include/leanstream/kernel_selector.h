#pragma once
#include "types.h"
namespace leanstream::runtime {
struct KernelRuntimeState { uint32_t layer; uint32_t io_bucket; };
// IDs are dense registry indices. Configuration fields are interpreted by the backend.
struct KernelCandidate {
    uint32_t id;
    uint32_t workers;
    const char* name;
    uint32_t cta_count=0, tile_size=0, unroll=0, group_batch=0, entry_point=0;
    bool available=true;
};
inline constexpr KernelCandidate kKernelCandidates[] = {
    {0,1,"workers_1"}, {1,2,"workers_2"}, {2,4,"workers_4"},
    {3,8,"workers_8"}, {4,16,"workers_16"}
};
inline constexpr KernelCandidate kDefaultKernelCandidate=kKernelCandidates[2];
// Borrowed immutable row-major storage: layers * io_buckets bytes.
struct KernelDecisionTable {
    const uint8_t* entries=nullptr;
    uint32_t layers=0, io_buckets=0;
};
struct KernelPlan { KernelCandidate candidate; };
// floor(min(remaining,total) * buckets / total), clamped to buckets-1.
// Bucket zero is least remaining load; total==0 or buckets==0 yields zero.
uint32_t quantize_io_load(uint32_t remaining,uint32_t total,uint32_t buckets=8) noexcept;
class KernelSelector {
public:
    explicit KernelSelector(KernelDecisionTable table={},
        Span<KernelCandidate> candidates={kKernelCandidates,5},
        KernelCandidate fallback=kDefaultKernelCandidate) noexcept
        : table_(table), candidates_(candidates), fallback_(fallback) {}
    uint32_t select(const KernelRuntimeState&) const noexcept;
    uint32_t select(uint32_t layer,uint32_t io_bucket) const noexcept { return select({layer,io_bucket}); }
    KernelCandidate candidate(uint32_t id) const noexcept;
    KernelPlan plan(const KernelRuntimeState& s) const noexcept { return {candidate(select(s))}; }
private:
    KernelDecisionTable table_;
    Span<KernelCandidate> candidates_;
    KernelCandidate fallback_;
};
}
