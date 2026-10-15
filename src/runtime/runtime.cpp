#include "../../include/leanstream/runtime.h"
#include <atomic>
#include <cerrno>
#include <cassert>
#include <algorithm>
namespace leanstream::runtime {
struct Runtime::LayerContext {
    Runtime* parent;WeightManager manager;
    std::vector<GroupId> next_resident;
    std::vector<uint8_t> next_marks;
    std::atomic<bool> active{false},pending{false};
    std::atomic<uint32_t> retire_epoch{0};
    bool enqueued=false,finalized=false;
    int error=0; uint32_t layer;
    LayerContext(Runtime* r,uint32_t l,ManagerConfig c):parent(r),manager(c),next_resident(c.group_count),next_marks(c.group_count),layer(l) {}
};
Runtime::Runtime(ManagerConfig c,ComputeBackend b,uint32_t n):backend_(b),mode_(c.mode) {
    layers_.reserve(n);
    for(uint32_t l=0;l<n;l++)layers_.emplace_back(new LayerContext(this,l,c));
}
Runtime::~Runtime(){int rc=drain();assert(rc!=EBUSY);(void)rc;}
WeightManager& Runtime::weights(uint32_t l){return layers_.at(l)->manager;}
int Runtime::begin_layer(uint32_t l,const LayerStorage& s,LayerStep* step) {
    if(l>=layers_.size()||!step)return EINVAL;
    auto& c=*layers_[l];if(c.active.load(std::memory_order_acquire))return EBUSY;
    int rc=c.manager.begin_layer(l,s);if(rc)return rc;
    c.enqueued=c.finalized=false;c.error=0;c.pending=false;c.active=true;
    *step=LayerStep(this,l,c.manager.epoch());return 0;
}
int LayerStep::update_prediction(Span<GroupId> p){return update_prediction(p,{});}
int LayerStep::update_prediction(Span<GroupId> p,PredictionUpdateOptions options) {
    if(!runtime_)return EINVAL;auto& c=*runtime_->layers_[layer_];
    if(!c.active||epoch_!=c.manager.epoch()||c.finalized)return EINVAL;
    return c.manager.update_prediction(p,options);
}
int LayerStep::enqueue_compute(const void* input,void* output,void* stream) {
    if(!runtime_)return EINVAL;auto& r=*runtime_;auto& c=*r.layers_[layer_];
    if(!c.active||epoch_!=c.manager.epoch()||c.enqueued||!r.backend_.enqueue||!r.backend_.complete)return EINVAL;
    c.enqueued=true; // A backend error may follow a partial GPU enqueue; abort must still wait.
    int rc=r.backend_.enqueue(r.backend_.context,layer_,input,output,stream,c.manager.core_descriptors(),c.manager.queue(),epoch_,r.mode_);
    if(rc){c.error=rc;c.manager.cancel(rc);if(r.backend_.cancel)r.backend_.cancel(r.backend_.context,layer_,epoch_);}
    return rc;
}
int LayerStep::finalize(Span<GroupId> ids) {
    if(!runtime_)return EINVAL;auto& c=*runtime_->layers_[layer_];
    if(!c.active||epoch_!=c.manager.epoch()||c.finalized)return EINVAL;
    uint32_t error=__atomic_load_n(&c.manager.queue().control->error,__ATOMIC_ACQUIRE);if(error)return int(error);
    int rc=c.manager.finalize(ids);if(!rc)c.finalized=true;
    else if(c.enqueued){c.error=rc;c.manager.cancel(rc);if(runtime_->backend_.cancel)runtime_->backend_.cancel(runtime_->backend_.context,layer_,epoch_);}
    return rc;
}
void Runtime::retire_callback(void* ptr,uint32_t epoch) {
    auto& c=*static_cast<LayerContext*>(ptr);auto& r=*c.parent;
    uint32_t expected=epoch;
    if(!epoch||!c.retire_epoch.compare_exchange_strong(expected,0,std::memory_order_acq_rel))return;
    // Contract: backend has completed actual cache byte installation, after GPU
    // consumption. Callback retirement precedes active=false / bank reuse.
    int error=c.manager.retire();
    {std::lock_guard<std::mutex> lock(r.mutex_);c.error=error;c.pending=false;c.active.store(false,std::memory_order_release);}
    r.retired_.notify_all();
}
int LayerStep::commit(Span<GroupId> ids) {
    if(!runtime_)return EINVAL;auto& r=*runtime_;auto& c=*r.layers_[layer_];
    if(!c.active||epoch_!=c.manager.epoch()||!c.finalized||!c.enqueued||c.pending)return EINVAL;
    uint32_t queue_error=__atomic_load_n(&c.manager.queue().control->error,__ATOMIC_ACQUIRE);
    if(queue_error)return int(queue_error);
    if(ids.size>c.next_resident.size()||(ids.size&&!ids.data))return EINVAL;
    if(!r.backend_.defer_commit && !r.backend_.complete(r.backend_.context,layer_))return EBUSY;
    std::fill(c.next_marks.begin(),c.next_marks.end(),0);
    for(size_t i=0;i<ids.size;i++) {
        if(ids[i]<0||size_t(ids[i])>=c.next_resident.size())return EINVAL;
        if(c.next_marks[ids[i]])return EINVAL;
        c.next_marks[ids[i]]=1;
        if(!r.backend_.defer_commit) {
            auto state=c.manager.group_state(ids[i]);
            if(state.cache_slot<0&&state.io_state!=IOState::Ready)return EAGAIN;
        }
    }
    if(r.backend_.defer_commit) {
        std::copy_n(ids.data,ids.size,c.next_resident.data());c.pending=true;c.retire_epoch.store(epoch_,std::memory_order_release);
        int rc=r.backend_.defer_commit(r.backend_.context,layer_,{c.next_resident.data(),ids.size},epoch_,Runtime::retire_callback,&c);
        if(rc){c.pending=false;c.retire_epoch=0;}
        return rc;
    }
    if(!r.backend_.commit)return ENOTSUP;
    int rc=r.backend_.commit(r.backend_.context,layer_,ids);if(rc)return rc;
    rc=c.manager.retire();c.active=false;return rc;
}
int LayerStep::abort() {
    if(!runtime_)return EINVAL;auto& r=*runtime_;auto& c=*r.layers_[layer_];
    if(!c.active||epoch_!=c.manager.epoch()||c.pending)return EINVAL;
    c.manager.cancel();
    if(c.enqueued&&r.backend_.cancel)r.backend_.cancel(r.backend_.context,layer_,epoch_);
    if(c.enqueued&&!r.backend_.complete(r.backend_.context,layer_))return EBUSY;
    int rc=c.manager.retire();c.active=false;return rc;
}
int Runtime::drain() {
    std::unique_lock<std::mutex> lock(mutex_);
    for(auto& c:layers_)if(c->active&&!c->pending)return EBUSY;
    retired_.wait(lock,[&]{for(auto& c:layers_)if(c->active)return false;return true;});
    for(auto& c:layers_)if(c->error)return c->error;
    return 0;
}
}
