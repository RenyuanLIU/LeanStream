#pragma once
#include "kernel_selector.h"
namespace leanstream::runtime {
// workers and cta_count both denote persistent B CTAs here, never warps.
inline constexpr KernelCandidate kTritonCandidates[] = {
    {0,4,"triton_b_ctas_4",4}, {1,8,"triton_b_ctas_8",8},
    {2,16,"triton_b_ctas_16",16}, {3,32,"triton_b_ctas_32",32}
};
inline constexpr KernelCandidate kTritonDefault=kTritonCandidates[3];
inline KernelSelector triton_selector(KernelDecisionTable offline_table={}) noexcept {
    return KernelSelector(offline_table,{kTritonCandidates,4},kTritonDefault);
}
struct TritonLaunchGeometry {
    uint32_t core_ctas=0, persistent_ctas=32, reduction_ctas=0;
    uint32_t warps_per_cta=4; // 128 threads; independent of grid dimensions
};
}
