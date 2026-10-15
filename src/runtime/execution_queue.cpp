#include "../../include/leanstream/execution_queue.h"
#include <cassert>
namespace leanstream::runtime {
void ExecutionQueue::reset(uint32_t epoch) {
    assert(view_.control && view_.descriptors && view_.ready_indices && epoch);
    tail_=total_=0; epoch_=epoch;
    __atomic_store_n(&view_.control->epoch,0,__ATOMIC_RELEASE);
    view_.control->ready_count=0; view_.control->reserved=0; view_.control->claim_count=0;
    view_.control->total_count=0; view_.control->batch_epoch=0; view_.control->error=0; view_.control->consumer_path=0; view_.control->init_arrived=0; view_.control->init_epoch=0;
}
void ExecutionQueue::prepare(uint32_t total) {
    assert(total<=view_.capacity); total_=total;view_.control->total_count=total;
}
void ExecutionQueue::append_unpublished(uint32_t index) {
    assert(tail_<total_&&index<total_);
#ifdef LEANSTREAM_RUNTIME_TESTING
    for(uint32_t i=0;i<tail_;i++)assert(view_.ready_indices[i]!=index);
#endif
    view_.ready_indices[tail_++]=index;
}
void ExecutionQueue::activate() {
    __atomic_store_n(&view_.control->ready_count,tail_,__ATOMIC_RELEASE);
    __atomic_store_n(&view_.control->epoch,epoch_,__ATOMIC_RELEASE);
    if(tail_==total_)__atomic_store_n(&view_.control->batch_epoch,epoch_,__ATOMIC_RELEASE);
}
void ExecutionQueue::seal(uint32_t total) {prepare(total);activate();}
void ExecutionQueue::publish_all() {
    assert(tail_==0);
    for(uint32_t i=0;i<total_;i++)view_.ready_indices[i]=i;
    tail_=total_;
    __atomic_store_n(&view_.control->ready_count,tail_,__ATOMIC_RELEASE);
    __atomic_store_n(&view_.control->batch_epoch,epoch_,__ATOMIC_RELEASE);
}
void ExecutionQueue::publish(uint32_t index) {
    assert(tail_<total_ && index<total_);
    #ifdef LEANSTREAM_RUNTIME_TESTING
    for(uint32_t i=0;i<tail_;i++)assert(view_.ready_indices[i]!=index);
    #endif
    view_.ready_indices[tail_++]=index;
    __atomic_store_n(&view_.control->ready_count,tail_,__ATOMIC_RELEASE);
    if(tail_==total_)__atomic_store_n(&view_.control->batch_epoch,epoch_,__ATOMIC_RELEASE);
}
void ExecutionQueue::fail(uint32_t error) {
    uint32_t clear=0;
    __atomic_compare_exchange_n(&view_.control->error,&clear,error,false,__ATOMIC_RELEASE,__ATOMIC_RELAXED);
}
}
