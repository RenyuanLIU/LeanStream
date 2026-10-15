#include "leanstream/kernel_selector.h"
#include <limits>
namespace leanstream::runtime {
uint32_t quantize_io_load(uint32_t remaining,uint32_t total,uint32_t buckets) noexcept {
    if(!total||!buckets)return 0;
    if(remaining>=total)return buckets-1;
    return uint32_t(uint64_t(remaining)*buckets/total);
}
KernelCandidate KernelSelector::candidate(uint32_t id) const noexcept {
    // The backend owns its fallback; CUDA and Triton have different geometry.
    if(id==fallback_.id)return fallback_;
    if(!candidates_.data||id>=candidates_.size)return fallback_;
    const auto& c=candidates_[id];
    if(c.id!=id||!c.available||!c.workers)return fallback_;
    return c;
}
uint32_t KernelSelector::select(const KernelRuntimeState& s) const noexcept {
    if(!table_.entries||s.layer>=table_.layers||s.io_bucket>=table_.io_buckets ||
       (table_.io_buckets && size_t(table_.layers)>std::numeric_limits<size_t>::max()/table_.io_buckets))
        return fallback_.id;
    return candidate(table_.entries[size_t(s.layer)*table_.io_buckets+s.io_bucket]).id;
}
}
