#include "../../include/leanstream/prediction_schedule.h"
#include <cerrno>
namespace leanstream::runtime {
int compile_prediction_schedule(const PredictionSchedule& s,uint32_t q,uint32_t kv,uint32_t d,AttentionExecutionPlan* out){
 if(!out||!s.count||s.count>5||!q||!kv||!d||q%kv)return EINVAL;
 AttentionExecutionPlan p;p.query_heads=q;p.kv_heads=kv;p.head_dim=d;p.io_budget.fill(InheritIOBudget);
 for(uint32_t i=0;i<s.count;i++){auto v=s.points[i];uint32_t id=uint32_t(v.point);if(id>=5||(p.enabled_mask&(1u<<id)))return EINVAL;p.enabled_mask|=1u<<id;p.io_budget[id]=v.io_budget;}
 if(!(p.enabled_mask&(1u<<4)))return EINVAL;
 uint32_t begin=0,count=0;
 for(uint32_t id=1;id<4;id++)if(p.enabled_mask&(1u<<id)){
  if((uint64_t(q)*id)%4)return ENOTSUP;uint32_t end=q*id/4;if(end%(q/kv))return ENOTSUP;
  p.segments[count++]={begin,end,PredictionPoint(id)};begin=end;p.internal_mask|=1u<<(id-1);
 }
 p.segments[count++]={begin,q,PredictionPoint::AttnOutput};p.num_segments=count;p.variant=p.internal_mask?AttentionVariant::Segmented:AttentionVariant::Unsplit;*out=p;return 0;
}
int LatestAvailable::next(const std::array<CheckpointMailbox,5>& box){
 if(done_||active_>=0||!epoch_)return -1;
 int newest=-1;for(int id=4;id>=0;id--)if((plan_.enabled_mask&(1u<<id))&&!(seen_&(1u<<id))&&box[id].ready_epoch&&__atomic_load_n(box[id].ready_epoch,__ATOMIC_ACQUIRE)==epoch_){newest=id;break;}
 if(newest<0)return -1;
 // Unstarted lower-quality jobs are superseded; never overwrite active storage.
 for(int id=0;id<newest;id++)if((plan_.enabled_mask&(1u<<id))&&!(seen_&(1u<<id))){seen_|=1u<<id;skipped++;}
 active_=newest;seen_|=1u<<newest;return newest;
}
bool LatestAvailable::finish(const std::array<CheckpointMailbox,5>& box){
 if(active_<0)return false;int id=active_;active_=-1;executed++;
 if(id!=4&&box[4].ready_epoch&&__atomic_load_n(box[4].ready_epoch,__ATOMIC_ACQUIRE)==epoch_){skipped++;return false;}
 updates++;if(id==4)done_=true;return true;
}
}
extern "C" int ls_compile_attention_plan(const uint32_t* points,const uint32_t* budgets,int count,uint32_t q,uint32_t kv,uint32_t dim,uint32_t* words){
 using namespace leanstream::runtime;if(!points||!budgets||!words||count<1||count>5)return EINVAL;
 PredictionSchedule schedule;schedule.count=count;for(int i=0;i<count;i++)schedule.points[i]={PredictionPoint(points[i]),budgets[i]};
 AttentionExecutionPlan plan;int rc=compile_prediction_schedule(schedule,q,kv,dim,&plan);if(rc)return rc;
 words[0]=uint32_t(plan.variant);words[1]=plan.num_segments;words[2]=plan.internal_mask;words[3]=plan.enabled_mask;
 for(int i=0;i<4;i++){words[4+3*i]=plan.segments[i].q_head_begin;words[5+3*i]=plan.segments[i].q_head_end;words[6+3*i]=uint32_t(plan.segments[i].checkpoint);}
 for(int i=0;i<5;i++)words[16+i]=plan.io_budget[i];return 0;
}
