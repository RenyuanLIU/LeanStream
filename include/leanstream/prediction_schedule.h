#pragma once
#include "types.h"
#include <array>
#include <initializer_list>
#include <limits>
namespace leanstream::runtime {
enum class PredictionPoint:uint8_t {AttnInput,AttnQuarter,AttnHalf,AttnThreeQuarter,AttnOutput};
constexpr uint32_t InheritIOBudget=std::numeric_limits<uint32_t>::max();
struct PredictionPointConfig {PredictionPoint point;uint32_t io_budget=InheritIOBudget;};
struct PredictionSchedule {
 std::array<PredictionPointConfig,5> points{};uint32_t count=0;
 PredictionSchedule():PredictionSchedule({{PredictionPoint::AttnInput},{PredictionPoint::AttnOutput}}){}
 PredictionSchedule(std::initializer_list<PredictionPointConfig> values){count=values.size();if(count<=5){uint32_t i=0;for(auto v:values)points[i++]=v;}}
};
enum class AttentionVariant:uint8_t {Unsplit,Segmented};
struct AttentionSegment {uint32_t q_head_begin=0,q_head_end=0;PredictionPoint checkpoint=PredictionPoint::AttnOutput;};
struct AttentionExecutionPlan {
 AttentionVariant variant=AttentionVariant::Unsplit;uint32_t num_segments=1,internal_mask=0,enabled_mask=0;
 std::array<AttentionSegment,4> segments{};std::array<uint32_t,5> io_budget{};
 uint32_t query_heads=0,kv_heads=0,head_dim=0;
};
// Normalize order; never round an unsupported checkpoint boundary.
int compile_prediction_schedule(const PredictionSchedule&,uint32_t query_heads,uint32_t kv_heads,uint32_t head_dim,AttentionExecutionPlan*);
struct CheckpointMailbox {const void* data=nullptr;const uint32_t* ready_epoch=nullptr;};
// One persistent control worker owns this selector. GPU owns each ready flag.
// Caller keeps every slot immutable until the final job/epoch retires.
class LatestAvailable {
 AttentionExecutionPlan plan_{};uint32_t epoch_=0,seen_=0;int active_=-1;bool done_=false;
public:
 uint32_t executed=0,skipped=0,updates=0;
 void arm(const AttentionExecutionPlan& p,uint32_t epoch){plan_=p;epoch_=epoch;seen_=0;active_=-1;done_=false;executed=skipped=updates=0;}
 int next(const std::array<CheckpointMailbox,5>&);
 bool finish(const std::array<CheckpointMailbox,5>&); // true => immediately update I/O; false => stale
 bool done() const{return done_;}
};
}
