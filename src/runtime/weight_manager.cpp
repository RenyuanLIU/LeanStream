#include "../../include/leanstream/weight_manager.h"
#include "../io/stream_io.h"
#include <vector>
#include <mutex>
#include <algorithm>
#include <cerrno>
#include <stdexcept>
#include <limits>
#include <atomic>
#ifdef LEANSTREAM_RUNTIME_TESTING
extern "C" void leanstream_test_finalize_hook(int);
#define FINALIZE_HOOK(point) leanstream_test_finalize_hook(point)
#else
#define FINALIZE_HOOK(point) ((void)0)
#endif
namespace leanstream::runtime {
struct WeightManager::Impl {
    ManagerConfig config;
    StreamEngine* engine=nullptr; StreamSession* session=nullptr;
    LayerStorage storage{},prepared{};
    const GroupLayout* registered_layout=nullptr;uint64_t layout_generation=0;
    std::vector<StreamExtent> extents;
    std::vector<GroupLayout> layout;
    std::vector<GroupId> priority,final_reads,view_ids;
    std::vector<int32_t> final_index;
    std::vector<uint8_t> marks,published,core_marks;
    std::vector<ExecDesc> core;
    ExecDesc* core_storage=nullptr;
    ExecutionQueue execution;
    std::mutex mutex;
    std::atomic<uint32_t> epoch{0};uint32_t core_count=0; uint64_t generation=0;
    std::atomic<bool> active{false},finalized{false};
    explicit Impl(ManagerConfig c):config(c),extents(c.group_count),layout(c.group_count),
        final_index(c.group_count,-1),marks(c.group_count),published(c.group_count),core_marks(c.group_count),core(c.group_count) {
        if(!c.group_count||!c.group_bytes||c.staging_stride<c.group_bytes)throw std::invalid_argument("layout geometry");
        priority.reserve(c.group_count);final_reads.reserve(c.group_count);view_ids.resize(c.group_count);
        if(c.io_backend.begin) {
            if(!c.io_backend.update||!c.io_backend.consume||!c.io_backend.retire||!c.io_backend.ready_slot)
                throw std::invalid_argument("incomplete borrowed IO backend");
            return;
        }
        StreamConfig sc{c.qd,c.group_count,0,c.alignment,c.alignment};
        engine=stream_create_engine(&sc);if(!engine)throw std::runtime_error("stream_create_engine");
    }
    ~Impl(){if(engine)stream_destroy_engine(engine);}
    void publish_group(GroupId id,SlotId slot=-1) { // mutex held; completed bytes already acquire-visible
        int index=final_index[id]; if(index<0||published[id])return;
        auto& d=storage.queue.descriptors[index];
        if(slot<0)slot=config.io_backend.begin?config.io_backend.ready_slot(config.io_backend.context,id):id;
        if(slot<0||uint32_t(slot)>=config.group_count){execution.fail(EIO);return;}
        d.source_slot=uint32_t(slot); d.source_kind=SourceKind::Staging;
        published[id]=1; execution.publish(uint32_t(index));
    }
    static void external_completion(void* arg,const IOCompletion& notice) {
        auto& p=*static_cast<Impl*>(arg);if(!p.active.load(std::memory_order_acquire)||notice.epoch!=p.epoch.load(std::memory_order_relaxed))return;
        if(notice.error){std::lock_guard<std::mutex> lock(p.mutex);if(p.active&&notice.epoch==p.epoch)p.execution.fail(notice.error);return;}
        if(notice.logical_group<0||uint32_t(notice.logical_group)>=p.config.group_count||notice.staging_slot<0||uint32_t(notice.staging_slot)>=p.config.group_count) {
            p.execution.fail(EINVAL);return;
        }
        // The transport has verified slot_sequence ownership and data visibility.
        if(!p.finalized.load(std::memory_order_acquire))return;
        std::lock_guard<std::mutex> lock(p.mutex);
        if(p.active&&notice.epoch==p.epoch)p.publish_group(notice.logical_group,notice.staging_slot);
    }
    static void completion(void* arg,int32_t id,const StreamBlockInfo* info) {
        auto& p=*static_cast<Impl*>(arg);
        // begin cannot replace this context until end drains every callback.
        if(!p.active)return;
        if(info->error){p.execution.fail(info->error);return;}
        if(!p.finalized.load(std::memory_order_acquire))return;
        std::lock_guard<std::mutex> lock(p.mutex);if(p.active)p.publish_group(id);
    }
};
static ManagerConfig checked_config(ManagerConfig c) {
    if(!c.group_count||c.group_count>uint32_t(std::numeric_limits<int32_t>::max())||!c.group_bytes||c.staging_stride<c.group_bytes||
       c.staging_stride>std::numeric_limits<size_t>::max()/c.group_count)throw std::invalid_argument("layout geometry/capacity overflow");
    return c;
}
WeightManager::WeightManager(ManagerConfig c):impl_(new Impl(checked_config(c))) {}
WeightManager::~WeightManager()=default;
int WeightManager::prepare_layer(const LayerStorage& s) {
    auto& p=*impl_; if(p.active)return EBUSY;
    if(s.layout.size!=p.config.group_count||!s.layout.data||!s.cache_slots||!s.staging||!s.io_states||
       !s.queue.control||!s.queue.descriptors||!s.queue.ready_indices||s.queue.capacity>p.config.group_count||
       (s.core.size&&!s.core.data)||s.core.size>p.config.group_count)return EINVAL;
    if(s.staging_bytes && s.staging_bytes < p.config.group_count*p.config.staging_stride)return ENOSPC;
    p.prepared={}; // failed preparation cannot reuse partially updated metadata
    if(!s.layout_generation||s.layout_generation!=p.layout_generation||s.layout.data!=p.registered_layout) {
    p.registered_layout=nullptr;p.layout_generation=0; // failed registration cannot reuse a partial copy
    std::fill(p.marks.begin(),p.marks.end(),0);
    for(size_t i=0;i<s.layout.size;i++) {
        const auto& g=s.layout[i];
        if(g.logical_group>=p.config.group_count||p.marks[g.logical_group]++||g.bytes!=p.config.group_bytes)return EINVAL;
        p.layout[g.logical_group]=g;
        p.extents[g.logical_group]={g.file_offset,uint64_t(g.logical_group)*p.config.staging_stride,g.bytes,0};
    }
    p.registered_layout=s.layout.data;p.layout_generation=s.layout_generation;
    }
    std::fill(p.core_marks.begin(),p.core_marks.end(),0);std::fill(p.marks.begin(),p.marks.end(),0);p.core_count=s.core.size;
    p.core_storage=s.mapped_core_descriptors?s.mapped_core_descriptors:p.core.data();
    for(size_t i=0;i<s.core.size;i++) {
        int g=s.core[i];
        if(g<0||uint32_t(g)>=p.config.group_count||p.core_marks[g]++||s.cache_slots[g]<0)return EINVAL;
        if(s.destination_slots&&(s.destination_slots[g]<0||uint32_t(s.destination_slots[g])>=p.config.group_count||p.marks[s.destination_slots[g]]++))return EINVAL;
        p.core_storage[i]={uint32_t(g),p.layout[g].physical_group,uint32_t(s.cache_slots[g]),uint32_t(s.destination_slots?s.destination_slots[g]:i),SourceKind::Cache};
    }
    p.prepared=s;return 0;
}
int WeightManager::begin_layer(uint32_t,const LayerStorage& s) {
    auto& p=*impl_;if(p.active)return EBUSY;
    if(p.epoch==std::numeric_limits<uint32_t>::max())return EOVERFLOW;
    const auto& prepared=p.prepared;
    bool reuse=s.plan_generation&&s.plan_generation==prepared.plan_generation&&
        s.layout.data==prepared.layout.data&&s.layout.size==prepared.layout.size&&s.layout_generation==prepared.layout_generation&&
        s.cache_slots==prepared.cache_slots&&s.destination_slots==prepared.destination_slots&&
        s.core.data==prepared.core.data&&s.core.size==prepared.core.size&&s.mapped_core_descriptors==prepared.mapped_core_descriptors&&
        s.queue.control==prepared.queue.control&&s.queue.descriptors==prepared.queue.descriptors&&
        s.queue.ready_indices==prepared.queue.ready_indices&&s.queue.capacity==prepared.queue.capacity&&
        s.staging==prepared.staging&&s.io_states==prepared.io_states&&s.staging_bytes==prepared.staging_bytes;
    if(!reuse){int rc=prepare_layer(s);if(rc)return rc;}
    p.storage=s;p.execution.bind(s.queue);
    p.execution.reset(++p.epoch);
    std::fill(p.final_index.begin(),p.final_index.end(),-1);std::fill(p.published.begin(),p.published.end(),0);
    p.generation=0;p.finalized=false;p.active=true;
    if(p.config.io_backend.begin) {
        int rc=p.config.io_backend.begin(p.config.io_backend.context,Impl::external_completion,&p,p.epoch);
        if(rc)p.active=false;
        return rc;
    }
    p.session=stream_begin_layer(p.engine,s.fd,p.extents.data(),p.config.group_count,
        {s.staging,p.config.staging_stride*p.config.group_count,s.io_states},nullptr,0);
    if(!p.session){p.active=false;return errno?errno:EIO;}
    return stream_observe(p.session,Impl::completion,&p);
}
int WeightManager::update_prediction(Span<GroupId> ids){return update_prediction(ids,{});}
int WeightManager::update_prediction(Span<GroupId> ids,PredictionUpdateOptions options) {
    auto& p=*impl_;if(!p.active||p.finalized||(ids.size&&!ids.data))return EINVAL;
    if(options.execution_epoch&&options.execution_epoch!=p.epoch)return ESTALE;
    std::fill(p.marks.begin(),p.marks.end(),0);p.priority.clear();
    for(size_t i=0;i<ids.size;i++) {
        int g=ids[i];if(g<0||uint32_t(g)>=p.config.group_count)return EINVAL;
        if(p.marks[g])continue;
        p.marks[g]=1;
        // Filter cache/READY/INFLIGHT before admission scans. Started reads are
        // non-preemptive; the backend still rechecks claimed state under lock.
        if(p.storage.cache_slots[g]<0 && __atomic_load_n(p.storage.io_states+g,__ATOMIC_ACQUIRE)==STREAM_EMPTY)
            p.priority.push_back(g);
    }
    if(p.priority.size()>options.io_budget)p.priority.resize(options.io_budget);
    int rc=p.config.io_backend.begin?p.config.io_backend.update(p.config.io_backend.context,{p.priority.data(),p.priority.size()}):stream_update_priority(p.session,p.priority.data(),p.priority.size());
    if(!rc)++p.generation;
    return rc;
}
int WeightManager::finalize(Span<GroupId> ids) {
    auto& p=*impl_;if(!p.active||p.finalized||(ids.size&&!ids.data)||ids.size<p.core_count)return EINVAL;
    std::fill(p.marks.begin(),p.marks.end(),0);p.final_reads.clear();uint32_t count=0;
    // Validate all membership before mutating the visible plan.
    for(size_t i=0;i<ids.size;i++) {
        int g=ids[i];if(g<0||uint32_t(g)>=p.config.group_count||p.marks[g]++)return EINVAL;
        if(!p.core_marks[g])++count;
    }
    for(uint32_t g=0;g<p.config.group_count;g++)if(p.core_marks[g]&&!p.marks[g])return EINVAL;
    if(count>p.storage.queue.capacity)return ENOSPC;
    if(p.storage.destination_slots) {
        std::fill(p.marks.begin(),p.marks.end(),0);
        for(size_t i=0;i<ids.size;i++) {
            int dest=p.storage.destination_slots[ids[i]];
            if(dest<0||uint32_t(dest)>=p.config.group_count||p.marks[dest]++)return EINVAL;
        }
    }
    FINALIZE_HOOK(0);
    {
        std::lock_guard<std::mutex> lock(p.mutex);uint32_t idx=0;
        for(size_t i=0;i<ids.size;i++) {
            int g=ids[i];if(p.core_marks[g])continue;
            int slot=p.storage.cache_slots[g];
            p.storage.queue.descriptors[idx]={uint32_t(g),p.layout[g].physical_group,
                slot<0?0u:uint32_t(slot),uint32_t(p.storage.destination_slots?p.storage.destination_slots[g]:p.core_count+idx),slot<0?SourceKind::Staging:SourceKind::Cache};
            p.final_index[g]=idx++;
            if(slot<0)p.final_reads.push_back(g);
        }
        FINALIZE_HOOK(1);
        p.execution.prepare(count);p.finalized.store(true,std::memory_order_release);
        for(size_t i=0;i<ids.size;i++) {
            int g=ids[i];if(p.core_marks[g])continue;
            if(p.storage.cache_slots[g]>=0) {
                p.published[g]=1;p.execution.append_unpublished(p.final_index[g]);
            }else if(__atomic_load_n(p.storage.io_states+g,__ATOMIC_ACQUIRE)==STREAM_READY) {
                SlotId source=p.config.io_backend.begin?p.config.io_backend.ready_slot(p.config.io_backend.context,g):g;
                if(source<0||uint32_t(source)>=p.config.group_count){p.execution.fail(EIO);return EIO;}
                p.storage.queue.descriptors[p.final_index[g]].source_slot=source;
                p.published[g]=1;p.execution.append_unpublished(p.final_index[g]);
            }
        }
        p.execution.activate();
    }
    FINALIZE_HOOK(2);
    // Commits the exact nonresident final set, reuses completed/inflight reads,
    // and immediately prioritizes new final reads using the production engine.
    StreamView view{};
    int rc=p.config.io_backend.begin?p.config.io_backend.consume(p.config.io_backend.context,{p.final_reads.data(),p.final_reads.size()}):stream_consume(p.session,p.final_reads.data(),p.final_reads.size(),&view,p.view_ids.data(),p.view_ids.size());
    if(rc)p.execution.fail(rc);
    return rc;
}
int WeightManager::retire() {
    auto& p=*impl_;if(!p.active)return EINVAL;
    int rc=p.config.io_backend.begin?p.config.io_backend.retire(p.config.io_backend.context):stream_end_layer(p.session); // drains callbacks BEFORE resetting mapping/epoch
    {std::lock_guard<std::mutex> lock(p.mutex);p.active=false;}return rc;
}
GroupState WeightManager::group_state(GroupId g) const {
    const auto& p=*impl_;if(!p.active||g<0||uint32_t(g)>=p.config.group_count)return {};
    uint32_t s=__atomic_load_n(p.storage.io_states+g,__ATOMIC_ACQUIRE);
    SlotId slot=(s==STREAM_READY||s==STREAM_INFLIGHT)?g:-1;
    if(slot>=0&&p.config.io_backend.begin)slot=p.config.io_backend.ready_slot(p.config.io_backend.context,g);
    return {p.storage.cache_slots[g],slot,s==STREAM_READY?IOState::Ready:s==STREAM_INFLIGHT?IOState::Inflight:IOState::Empty};
}
uint64_t WeightManager::prediction_generation() const {return impl_->generation;}
uint32_t WeightManager::epoch() const {return impl_->epoch;}
DeviceExecQueue WeightManager::queue() const {return impl_->storage.queue;}
Span<ExecDesc> WeightManager::core_descriptors() const {return {impl_->core_storage,impl_->core_count};}
int WeightManager::wait_io(int ms) {auto& p=*impl_;if(p.config.io_backend.begin)return p.config.io_backend.wait?p.config.io_backend.wait(p.config.io_backend.context,ms):ENOTSUP;return p.session?stream_wait(p.session,ms):EINVAL;}
uint64_t WeightManager::read_requests() const {auto& p=*impl_;if(p.config.io_backend.begin)return p.config.io_backend.requests?p.config.io_backend.requests(p.config.io_backend.context):0;StreamStatus s{};if(!impl_->session||stream_get_status(impl_->session,&s))return 0;return s.admissions;}
}

namespace leanstream::runtime { void WeightManager::cancel(int error) { if(impl_->active)impl_->execution.fail(error?error:ECANCELED); } }
