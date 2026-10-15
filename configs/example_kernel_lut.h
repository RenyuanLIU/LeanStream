#pragma once
#include "leanstream/kernel_selector.h"
namespace leanstream::example {
// SYNTHETIC API DEMONSTRATION ONLY. Not measured, tuned, or production-optimal.
inline constexpr uint8_t kKernelDecisionTable[3*8] = {
    3,3,2,2,1,1,0,0, // layer 0
    4,3,3,2,2,1,1,0, // layer 1
    2,2,2,2,1,1,0,0  // layer 2
};
inline constexpr runtime::KernelDecisionTable table() noexcept {
    return {kKernelDecisionTable,3,8};
}
}
