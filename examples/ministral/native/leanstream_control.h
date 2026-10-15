#pragma once
#include <array>
#include <cstdint>

namespace leanstream {
constexpr int GROUPS=7168, CAPACITY=1434, KEEP=1219, FILL=215;
constexpr int MAX_GROUPS=11520, MAX_CAPACITY=2304, MAX_KEEP=1958, MAX_FILL=346;
// CPU-only Stage-A state. Future mapped metadata belongs to the runtime, not the GPU cache.
struct CachePlan {
    uint64_t ticket=0;
    int32_t missing_count=0,hits=0;
    std::array<int32_t,MAX_CAPACITY> selected{},missing{},destinations{};
};
class CacheState {
public:
    enum Phase {IDLE,PLANNED,LAUNCHED};
    CacheState() noexcept;
    int configure(int group_size, bool stable) noexcept;
    int configure_model(int group_size, int group_count, int active_count,
                        int retained_count, bool stable) noexcept;
    int stage_core(const float* scores) noexcept;
    int groups=GROUPS, capacity=CAPACITY, keep_size=KEEP, fill_size=FILL, eligible=GROUPS;
    int reset() noexcept; // Caller must also ensure external GPU/I/O users are quiescent.
    int plan_selected(const int32_t* selected) noexcept; // Prefill/offline initialization.
    int plan_final(const int32_t* fill) noexcept;
    int mark_launched(uint64_t ticket) noexcept;
    int commit(uint64_t ticket) noexcept;
    int abort(uint64_t ticket) noexcept; // Only before GPU launch.
    int update_keep(const float* importance_by_slot) noexcept;
    int prefetch_internal(const int32_t* fill) noexcept; // Predictor-owned IDs; range-safe, no full keep validation.
    int prefetch(const int32_t* fill) noexcept; // P1: resident filtering only, no cache mutation.
    // Membership only; view(1) remains the authoritative resident slot mapping.
    bool cache_resident(int g) const noexcept {return (resident_bits[g >> 6] >> (g & 63)) & 1u;}
    const CachePlan& plan() const noexcept {return pending;}
    const int32_t* view(int kind) const noexcept;
    int phase() const noexcept {return current_phase;}
    int prefetch_count() const noexcept {return n_prefetch;}
private:
    std::array<int32_t,MAX_CAPACITY> slot_to_group{},selected_order{},ranking{};
    std::array<int32_t,MAX_GROUPS> group_to_slot{};
    std::array<uint64_t,(MAX_GROUPS+63)/64> resident_bits{};
    void cache_set_resident(int g) noexcept {resident_bits[g >> 6] |= uint64_t{1} << (g & 63);}
    void cache_clear_resident(int g) noexcept {resident_bits[g >> 6] &= ~(uint64_t{1} << (g & 63));}
    std::array<int32_t,MAX_KEEP> keep{};
    std::array<int32_t,MAX_FILL> prefetch_ids{};
    std::array<uint8_t,MAX_GROUPS> marks{};
    CachePlan pending{},draft{};
    std::array<int32_t,MAX_KEEP> next_core{};
    bool stable_core=false, core_staged=false;
    uint64_t next_ticket=0;
    int resident_count=0,keep_count=0,n_prefetch=0;
    Phase current_phase=IDLE;
    int validate_fill(const int32_t* fill) noexcept;
    int finish_plan() noexcept;
};
}

extern "C" {
void* lc_cache_create();
void lc_cache_destroy(void*);
int lc_cache_reset(void*);
int lc_cache_plan_selected(void*,const int32_t*);
int lc_cache_plan_final(void*,const int32_t*);
int lc_cache_mark_launched(void*,uint64_t);
int lc_cache_commit(void*,uint64_t);
int lc_cache_abort(void*,uint64_t);
int lc_cache_update_keep(void*,const float*);
int lc_cache_prefetch(void*,const int32_t*);
int lc_cache_prefetch_internal(void*,const int32_t*);
int lc_cache_configure_model(void*,int,int,int,int,int);
// Borrowed read-only views for native integration and OFFLINE tests; no Python hot path.
const int32_t* lc_cache_view(void*,int);
uint64_t lc_cache_ticket(void*);
int lc_cache_missing_count(void*);
int lc_cache_hits(void*);
int lc_cache_phase(void*);
int lc_cache_prefetch_count(void*);
}
