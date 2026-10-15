#pragma once
#include <cstdint>
namespace leanstream {
// A speculative result may publish only in its own token, before final P2.
inline bool midpoint_current(uint64_t ticket,uint64_t current,uint64_t final_publication) noexcept {
    return ticket!=0 && ticket==current && final_publication==0;
}
}
