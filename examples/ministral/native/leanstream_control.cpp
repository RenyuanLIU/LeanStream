#include "leanstream_control.h"
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <limits>
#include <new>

namespace leanstream {
CacheState::CacheState() noexcept {reset();}
int CacheState::configure(int g,bool stable) noexcept {
    if(g<2||g>4)return EINVAL;
    int count=g==2?7168:g==3?4779:3584;
    int active=2868/g, retained=g==2?1219:g==3?812:609;
    if(g!=2){
        if(current_phase!=IDLE || resident_count)return EBUSY;
        groups=count;eligible=count-(g==3);capacity=active;
        keep_size=retained;fill_size=active-retained;stable_core=stable;return reset();
    }
    return configure_model(g,count,active,retained,stable);
}
int CacheState::configure_model(int g,int count,int active,int retained,bool stable) noexcept {
    if(current_phase!=IDLE || resident_count)return EBUSY;
    if(g!=2 || count<active || count>MAX_GROUPS || active<1 || active>MAX_CAPACITY ||
       retained<1 || retained>MAX_KEEP || retained>=active || active-retained>MAX_FILL)return EINVAL;
    groups=eligible=count;capacity=active;keep_size=retained;fill_size=active-retained;
    stable_core=stable;return reset();
}
int CacheState::stage_core(const float* scores) noexcept {
    if(!stable_core)return 0;
    if(current_phase!=PLANNED || !scores || keep_count!=keep_size)return EINVAL;
    for(int g=0;g<groups;g++)if(!std::isfinite(scores[g])||scores[g]<0)return EINVAL;
    next_core=keep;
    std::array<int32_t,MAX_KEEP> flexible{};
    std::copy_n(keep.begin(),keep_size,flexible.begin());
    auto better=[&](int a,int b){return scores[a]>scores[b]||(scores[a]==scores[b]&&a<b);};
    std::sort(flexible.begin(),flexible.begin()+keep_size,better);
    std::array<int32_t,MAX_FILL> candidates{},existing{};
    std::copy_n(pending.selected.begin()+keep_size,fill_size,candidates.begin());
    std::sort(candidates.begin(),candidates.begin()+fill_size,better);
    int ne=0;for(int i=0;i<fill_size;i++)if(cache_resident(candidates[i]))existing[ne++]=candidates[i];
    std::array<uint8_t,MAX_GROUPS> in_core{};for(int i=0;i<keep_size;i++)in_core[keep[i]]=1;
    int a=0,b=0,limit=int(std::nearbyint(keep_size*.05));
    for(int k=0;k<limit;k++){
        int incumbent=flexible[keep_size-1-k];
        while(a<fill_size&&in_core[candidates[a]])++a;
        while(b<ne&&in_core[existing[b]])++b;
        if(a==fill_size)break;
        int incoming=candidates[a];
        // Python policy compares FP32 scores with a double relative threshold.
        if(b<ne && double(scores[existing[b]])>=.995*double(scores[incoming]))incoming=existing[b];
        if(double(scores[incumbent])>=.995*double(scores[incoming]))break;
        for(int i=0;i<keep_size;i++)if(next_core[i]==incumbent){next_core[i]=incoming;break;}
        in_core[incumbent]=0;in_core[incoming]=1;
    }
    core_staged=true;return 0;
}
int CacheState::reset() noexcept {
    if(current_phase!=IDLE)return EBUSY;
    slot_to_group.fill(-1);group_to_slot.fill(-1);resident_bits.fill(0);selected_order.fill(-1);keep.fill(-1);
    resident_count=keep_count=n_prefetch=0;core_staged=false;
    // Tickets stay monotonic across reset, preventing acceptance of stale launch/commit tokens.
    pending=CachePlan{};return 0;
}
int CacheState::validate_fill(const int32_t* fill) noexcept {
    if(!fill)return EINVAL;
    if(resident_count!=capacity || keep_count!=keep_size)return ENOENT;
    marks.fill(0);
    for(int k=0;k<keep_size;k++){int g=keep[k];
        if(g<0 || g>=eligible || !cache_resident(g) || marks[g])return EINVAL;
        marks[g]=1;
    }
    for(int i=0;i<fill_size;i++){
        int g=fill[i];if(g<0 || g>=eligible || marks[g])return EINVAL;
        marks[g]=1;
    }
    return 0;
}
int CacheState::finish_plan() noexcept {
    int n=0;
    for(int slot=0;slot<capacity;slot++){
        int g=slot_to_group[slot];
        if(g<0 || !marks[g])draft.destinations[n++]=slot;
    }
    if(n!=draft.missing_count)return EINVAL;
    if(next_ticket==std::numeric_limits<uint64_t>::max())return EOVERFLOW;
    draft.hits=capacity-n;draft.ticket=++next_ticket;
    pending=draft;current_phase=PLANNED;return 0;
}
int CacheState::plan_selected(const int32_t* selected) noexcept {
    if(current_phase!=IDLE)return EBUSY;
    if(!selected)return EINVAL;
    marks.fill(0);draft.missing_count=0;
    for(int i=0;i<capacity;i++){
        int g=selected[i];if(g<0 || g>=eligible || marks[g])return EINVAL;
        marks[g]=1;draft.selected[i]=g;
        if(!cache_resident(g))draft.missing[draft.missing_count++]=g;
    }
    return finish_plan();
}
int CacheState::plan_final(const int32_t* fill) noexcept {
    if(current_phase!=IDLE)return EBUSY;
    int status=validate_fill(fill);if(status)return status;
    draft.missing_count=0;
    std::copy_n(keep.begin(),keep_size,draft.selected.begin());
    for(int i=0;i<fill_size;i++){
        int g=fill[i];draft.selected[keep_size+i]=g;
        if(!cache_resident(g))draft.missing[draft.missing_count++]=g;
    }
    return finish_plan();
}
int CacheState::mark_launched(uint64_t ticket) noexcept {
    if(current_phase!=PLANNED || ticket!=pending.ticket)return EINVAL;
    // Only the integration caller can attest that its CUDA launch succeeded.
    current_phase=LAUNCHED;return 0;
}
int CacheState::commit(uint64_t ticket) noexcept {
    if(current_phase!=LAUNCHED || ticket!=pending.ticket)return EPERM;
    for(int i=0;i<pending.missing_count;i++){
        int slot=pending.destinations[i],old=slot_to_group[slot];
        if(old>=0){group_to_slot[old]=-1;cache_clear_resident(old);}
    }
    for(int i=0;i<pending.missing_count;i++){
        int slot=pending.destinations[i],g=pending.missing[i];
        slot_to_group[slot]=g;group_to_slot[g]=slot;cache_set_resident(g);
    }
    selected_order=pending.selected;resident_count=capacity;
    // Old keep membership is invalid until actual current-token importance is consumed.
    keep_count=0;current_phase=IDLE;return 0;
}
int CacheState::abort(uint64_t ticket) noexcept {
    if(current_phase!=PLANNED || ticket!=pending.ticket)return EPERM;
    current_phase=IDLE;return 0;
}
int CacheState::update_keep(const float* importance) noexcept {
    if(current_phase!=IDLE)return EBUSY;
    if(!importance || resident_count!=capacity)return EINVAL;
    for(int i=0;i<capacity;i++)if(!std::isfinite(importance[i]) || importance[i]<0)return EINVAL;
    if(stable_core && core_staged){keep=next_core;keep_count=keep_size;core_staged=false;return 0;}
    for(int i=0;i<capacity;i++)ranking[i]=i;
    // Exactly match NumPy stable descending sort in previous selected order.
    // Keep ordering affects later tie-breaking, even though W4A8 uses only its membership.
    auto better=[&](int a,int b){
        float x=importance[group_to_slot[selected_order[a]]];
        float y=importance[group_to_slot[selected_order[b]]];
        return x>y || (x==y && a<b);
    };
    std::partial_sort(ranking.begin(),ranking.begin()+keep_size,ranking.begin()+capacity,better);
    for(int i=0;i<keep_size;i++)keep[i]=selected_order[ranking[i]];
    keep_count=keep_size;return 0;
}
int CacheState::prefetch(const int32_t* fill) noexcept {
    if(current_phase!=IDLE)return EBUSY;
    int status=validate_fill(fill);if(status)return status;
    n_prefetch=0;
    for(int i=0;i<fill_size;i++)if(!cache_resident(fill[i]))prefetch_ids[n_prefetch++]=fill[i];
    return 0;
}
int CacheState::prefetch_internal(const int32_t* fill) noexcept {
    if(current_phase!=IDLE)return EBUSY;
    if(!fill)return EINVAL;
    if(resident_count!=capacity || keep_count!=keep_size)return ENOENT;
    // Bounds check even this internal entry point: malformed input must never index out of range.
    // Uniqueness/keep exclusion is guaranteed by native W4A8; external callers use prefetch().
    for(int i=0;i<fill_size;i++)if(fill[i]<0 || fill[i]>=eligible)return EINVAL;
    n_prefetch=0;
    for(int i=0;i<fill_size;i++)if(!cache_resident(fill[i]))prefetch_ids[n_prefetch++]=fill[i];
    return 0;
}
const int32_t* CacheState::view(int kind) const noexcept {
    switch(kind){
        case 0:return slot_to_group.data();case 1:return group_to_slot.data();
        case 2:return keep.data();case 3:return selected_order.data();
        case 4:return pending.missing.data();case 5:return pending.destinations.data();
        case 6:return pending.selected.data();case 7:return prefetch_ids.data();
        default:return nullptr;
    }
}
}
extern "C" {
void* lc_cache_create(){return new(std::nothrow) leanstream::CacheState;}
int lc_cache_configure(void* p,int g,int stable){return static_cast<leanstream::CacheState*>(p)->configure(g,stable);}
int lc_cache_stage_core(void* p,const float* scores){return static_cast<leanstream::CacheState*>(p)->stage_core(scores);}
void lc_cache_destroy(void* p){delete static_cast<leanstream::CacheState*>(p);}
int lc_cache_reset(void* p){return p?static_cast<leanstream::CacheState*>(p)->reset():EINVAL;}
int lc_cache_plan_selected(void* p,const int32_t* a){return p?static_cast<leanstream::CacheState*>(p)->plan_selected(a):EINVAL;}
int lc_cache_plan_final(void* p,const int32_t* a){return p?static_cast<leanstream::CacheState*>(p)->plan_final(a):EINVAL;}
int lc_cache_mark_launched(void* p,uint64_t t){return p?static_cast<leanstream::CacheState*>(p)->mark_launched(t):EINVAL;}
int lc_cache_commit(void* p,uint64_t t){return p?static_cast<leanstream::CacheState*>(p)->commit(t):EINVAL;}
int lc_cache_abort(void* p,uint64_t t){return p?static_cast<leanstream::CacheState*>(p)->abort(t):EINVAL;}
int lc_cache_update_keep(void* p,const float* a){return p?static_cast<leanstream::CacheState*>(p)->update_keep(a):EINVAL;}
int lc_cache_prefetch(void* p,const int32_t* a){return p?static_cast<leanstream::CacheState*>(p)->prefetch(a):EINVAL;}
int lc_cache_prefetch_internal(void* p,const int32_t* a){return p?static_cast<leanstream::CacheState*>(p)->prefetch_internal(a):EINVAL;}
int lc_cache_configure_model(void* p,int g,int groups,int active,int retained,int stable){
    return p?static_cast<leanstream::CacheState*>(p)->configure_model(g,groups,active,retained,stable!=0):EINVAL;
}
const int32_t* lc_cache_view(void* p,int k){return p?static_cast<leanstream::CacheState*>(p)->view(k):nullptr;}
uint64_t lc_cache_ticket(void* p){return p?static_cast<leanstream::CacheState*>(p)->plan().ticket:0;}
int lc_cache_missing_count(void* p){return p?static_cast<leanstream::CacheState*>(p)->plan().missing_count:-1;}
int lc_cache_hits(void* p){return p?static_cast<leanstream::CacheState*>(p)->plan().hits:-1;}
int lc_cache_phase(void* p){return p?static_cast<leanstream::CacheState*>(p)->phase():-1;}
int lc_cache_prefetch_count(void* p){return p?static_cast<leanstream::CacheState*>(p)->prefetch_count():-1;}
}
