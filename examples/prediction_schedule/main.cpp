#include "leanstream/prediction_schedule.h"
#include <cstdio>
#include <cassert>
using namespace leanstream::runtime;
int main(){
 PredictionSchedule schedules[]={{{PredictionPoint::AttnInput},{PredictionPoint::AttnOutput}},{{PredictionPoint::AttnInput},{PredictionPoint::AttnHalf,64},{PredictionPoint::AttnOutput}},{{PredictionPoint::AttnInput},{PredictionPoint::AttnQuarter,32},{PredictionPoint::AttnHalf,64},{PredictionPoint::AttnThreeQuarter,128},{PredictionPoint::AttnOutput}}};
 for(auto& schedule:schedules){AttentionExecutionPlan plan;assert(!compile_prediction_schedule(schedule,32,8,128,&plan));std::printf("%s segments:",plan.variant==AttentionVariant::Unsplit?"Unsplit":"Segmented");for(unsigned i=0;i<plan.num_segments;i++)std::printf(" [%u,%u)",plan.segments[i].q_head_begin,plan.segments[i].q_head_end);
  uint32_t epochs[5]{};std::array<CheckpointMailbox,5> boxes{};for(int i=0;i<5;i++)boxes[i].ready_epoch=epochs+i;LatestAvailable worker;worker.arm(plan,1);
  // This example's fake predictor keeps up. Each real application completion
  // must immediately invoke LayerStep.update_prediction(priorities, options).
  for(int i=0;i<5;i++)if(plan.enabled_mask&(1u<<i)){__atomic_store_n(epochs+i,1,__ATOMIC_RELEASE);assert(worker.next(boxes)==i);assert(worker.finish(boxes));}
  std::printf(" predictor_updates=%u io_generations=%u\n",worker.updates,worker.updates);
 }
}
