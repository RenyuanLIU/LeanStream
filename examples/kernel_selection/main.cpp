#include <initializer_list>
#include "leanstream/kernel_selector.h"
#include "configs/example_kernel_lut.h"
#include <cstdio>
using namespace leanstream::runtime;
int main(){
    puts("Synthetic LUT demonstration; no timing measurements.");
    for(auto c:kKernelCandidates)std::printf("candidate=%u name=%s workers=%u\n",c.id,c.name,c.workers);
    KernelSelector selector(leanstream::example::table());
    for(uint32_t layer:{0u,1u,2u,99u})for(uint32_t remaining:{0u,4u,8u}){
        auto bucket=quantize_io_load(remaining,8);auto plan=selector.plan({layer,bucket});
        std::printf("layer=%u remaining=%u total=8 io_bucket=%u -> candidate=%s\n",layer,remaining,bucket,plan.candidate.name);
    }
}
