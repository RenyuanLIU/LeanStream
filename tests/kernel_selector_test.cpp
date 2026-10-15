#include "leanstream/kernel_selector.h"
#include "leanstream/triton_backend.h"
#include "configs/example_kernel_lut.h"
#include <cassert>
#include <cstdlib>
#include <new>
#include <limits>
#include <cstdio>
static size_t allocations=0;
void* operator new(size_t n){++allocations;if(auto p=std::malloc(n?n:1))return p;throw std::bad_alloc();}
void* operator new[](size_t n){return ::operator new(n);}
void operator delete(void* p) noexcept {std::free(p);}
void operator delete[](void* p) noexcept {std::free(p);}
void operator delete(void* p,size_t) noexcept {std::free(p);}
void operator delete[](void* p,size_t) noexcept {std::free(p);}
using namespace leanstream::runtime;
int main(){
    KernelSelector selector(leanstream::example::table());
    for(uint32_t l=0;l<3;l++)for(uint32_t b=0;b<8;b++)assert(selector.select(l,b)==leanstream::example::kKernelDecisionTable[l*8+b]);
    assert(quantize_io_load(1,1)==7&&quantize_io_load(10,8)==7);
    assert(quantize_io_load(0,0)==0&&quantize_io_load(1,0)==0);
    assert(quantize_io_load(1,8,0)==0&&quantize_io_load(0,8)==0);
    for(uint32_t i=0;i<8;i++)assert(quantize_io_load(i,8)==i);
    assert(quantize_io_load(UINT32_MAX-1,UINT32_MAX)==7);
    assert(quantize_io_load(UINT32_MAX-1,UINT32_MAX,UINT32_MAX)==UINT32_MAX-1);
    assert(selector.select(3,0)==2&&selector.select(0,8)==2);
    assert(KernelSelector{}.select(0,0)==2);
    const uint8_t bad[]={255,0,1,2,3,4};
    KernelCandidate candidates[5];for(int i=0;i<5;i++)candidates[i]=kKernelCandidates[i];
    candidates[0].available=false;candidates[1].workers=0;candidates[3].id=99;
    KernelSelector invalid({bad,1,6},{candidates,5});
    for(int i=0;i<5;i++)assert(invalid.select(0,i)==2);
    assert(invalid.select(0,5)==4);
    assert(KernelSelector({bad+5,1,1},{nullptr,0}).select(0,0)==2);
    assert(invalid.candidate(255).workers==4);
    auto production=triton_selector();
    assert(production.plan({0,0}).candidate.workers==32);
    assert(production.plan({999,999}).candidate.cta_count==32);
    const uint8_t production_lut[]={0,1,2,3,255};
    auto configured=triton_selector({production_lut,1,5});
    for(uint32_t i=0;i<4;i++)assert(configured.candidate(configured.select(0,i)).workers==kTritonCandidates[i].cta_count);
    assert(configured.plan({0,4}).candidate.workers==32);
    assert(KernelSelector{}.plan({0,0}).candidate.workers==4);
    size_t before=allocations;
    for(int repeat=0;repeat<10000;repeat++)for(uint32_t l=0;l<3;l++)for(uint32_t b=0;b<8;b++){
        assert(production.plan({l,b}).candidate.cta_count==32);
        assert(selector.plan({l,b}).candidate.id==selector.select(l,b));
        assert(selector.select(l,b)==leanstream::example::kKernelDecisionTable[l*8+b]);
    }
    assert(allocations==before);
    puts("PASS: direct indexing, quantization, deterministic plans, invalid/unavailable fallback, allocation-free lookup");
}
