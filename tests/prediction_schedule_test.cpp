#include "leanstream/prediction_schedule.h"
#include <cassert>
#include <cerrno>
#include <cstdio>
using namespace leanstream::runtime;
int main(){
 for(uint32_t mask=0;mask<8;mask++){
  PredictionSchedule s;s.count=0;s.points[s.count++]={PredictionPoint::AttnInput};for(int i=1;i<4;i++)if(mask&(1u<<(i-1)))s.points[s.count++]={PredictionPoint(i)};s.points[s.count++]={PredictionPoint::AttnOutput};
  AttentionExecutionPlan p;assert(!compile_prediction_schedule(s,32,8,128,&p));assert(p.internal_mask==mask&&p.num_segments==1+__builtin_popcount(mask));
  assert(p.variant==(mask?AttentionVariant::Segmented:AttentionVariant::Unsplit));uint32_t end=0;for(uint32_t i=0;i<p.num_segments;i++){assert(p.segments[i].q_head_begin==end);end=p.segments[i].q_head_end;}assert(end==32);
 }
 AttentionExecutionPlan p;assert(compile_prediction_schedule({{PredictionPoint::AttnInput}},32,8,128,&p)==EINVAL);
 assert(compile_prediction_schedule({{PredictionPoint::AttnOutput},{PredictionPoint::AttnOutput}},32,8,128,&p)==EINVAL);
 assert(compile_prediction_schedule({{PredictionPoint::AttnHalf},{PredictionPoint::AttnOutput}},12,3,128,&p)==ENOTSUP);
 assert(!compile_prediction_schedule({{PredictionPoint::AttnOutput},{PredictionPoint::AttnQuarter},{PredictionPoint::AttnHalf}},32,8,128,&p));assert(p.num_segments==3);
 uint32_t ready[5]{};std::array<CheckpointMailbox,5> boxes{};for(int i=0;i<5;i++)boxes[i].ready_epoch=ready+i;
 assert(!compile_prediction_schedule({{PredictionPoint::AttnQuarter},{PredictionPoint::AttnHalf},{PredictionPoint::AttnThreeQuarter},{PredictionPoint::AttnOutput}},32,8,128,&p));
 LatestAvailable worker;worker.arm(p,7);ready[1]=7;assert(worker.next(boxes)==1);ready[2]=ready[3]=7;assert(worker.next(boxes)==-1);assert(worker.finish(boxes));assert(worker.next(boxes)==3);assert(worker.finish(boxes));ready[4]=7;assert(worker.next(boxes)==4);assert(worker.finish(boxes)&&worker.done());assert(worker.updates==3&&worker.skipped==1);
 worker.arm(p,8);ready[1]=8;assert(worker.next(boxes)==1);ready[2]=ready[4]=8;assert(!worker.finish(boxes));assert(worker.next(boxes)==4);assert(worker.finish(boxes)&&worker.done());assert(worker.updates==1);
 puts("PASS all8 plans, invalid boundaries/schedules, latest-wins, mandatory final, no fake generations");
}
