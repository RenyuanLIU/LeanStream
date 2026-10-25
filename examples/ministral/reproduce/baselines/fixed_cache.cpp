// Individual-neuron fixed-cache baseline. No group IDs or cache mutations.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <atomic>
#include <array>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <fcntl.h>
#include <mutex>
#include <thread>
#include <vector>
#include <unistd.h>
#include <sched.h>
#include <cstring>
#include <cerrno>
extern "C" void* ls_create(int,int,int,int,int);
extern "C" int ls_predict(void*,const void*,const float*,const void*,float,const int32_t*,int,int,int32_t*,float*,float*,float*);
extern "C" void ls_destroy(void*);
static constexpr int I=14336,K=2867,HOT=2437,BYTES=6144;
struct Layer {
 int fd=-1,dim=0,stride=0;void* predictor=nullptr;
 const uint8_t* weights=nullptr;const float* scales=nullptr;const int32_t* features=nullptr;
 uint16_t* hidden=nullptr;uint8_t* staging=nullptr;int32_t* desc=nullptr;
 cudaEvent_t input;
 std::array<int32_t,I> hot_slot;
 std::array<int32_t,K> selected,miss_ids,miss_slots;
 std::array<int8_t,1664> quant;
 int misses=0,error=0;std::atomic<int> cursor{0};
 uint64_t calls=0,bytes=0;
};
struct Engine {
 std::array<Layer,34> layers;std::mutex mutex;std::condition_variable work,done;
 std::thread predictor;std::vector<std::thread> readers;
 bool stop=false,job=false,finished=true;int layer=-1,epoch=0,remaining=0;
 Engine(){
  predictor=std::thread([this]{void* team=ls_create(I,512,K,12,4);if(team)ls_destroy(team);for(;;){int id;
   {std::unique_lock<std::mutex> l(mutex);work.wait(l,[&]{return stop||job;});if(stop)return;id=layer;job=false;}
   auto& x=layers[id];int rc=cudaEventSynchronize(x.input);
   float values[1664]={},maximum=0;
   for(int i=0;i<x.dim;i++){__half h;std::memcpy(&h,x.hidden+x.features[i],2);values[i]=__half2float(h);maximum=std::max(maximum,std::abs(values[i]));}
   float scale=maximum?maximum/127.f:1.f;
   x.quant.fill(0);for(int i=0;i<x.dim;i++)x.quant[i]=int8_t(std::max(-127.f,std::min(127.f,std::nearbyint(values[i]/scale))));
   if(!rc)rc=ls_predict(x.predictor,x.weights,x.scales,x.quant.data(),scale,nullptr,0,0,x.selected.data(),nullptr,nullptr,nullptr);
   x.misses=0;x.error=rc;x.calls=x.bytes=0;
   for(int j=0;j<K&&!rc;j++){
    int n=x.selected[j];if(n<0||n>=I){x.error=EINVAL;break;}
    int slot=x.hot_slot[n],source=slot<0;
    if(source){slot=x.misses;x.miss_ids[x.misses]=n;x.miss_slots[x.misses++]=slot;}
    int32_t* d=x.desc+5*j;d[0]=d[1]=n;d[2]=slot;d[3]=j;d[4]=source;
   }
   {std::lock_guard<std::mutex> l(mutex);x.cursor=0;
    if(x.error||!x.misses){finished=true;done.notify_all();}
    else{remaining=24;++epoch;work.notify_all();}}
  }});
  for(int t=0;t<24;t++)readers.emplace_back([this]{int seen=0;for(;;){int id;
   {std::unique_lock<std::mutex> l(mutex);work.wait(l,[&]{return stop||epoch!=seen;});if(stop)return;seen=epoch;id=layer;}
   auto& x=layers[id];uint64_t count=0,bytes=0;int error=0;
   for(;;){int j=x.cursor.fetch_add(1);if(j>=x.misses)break;ssize_t got;
    do{got=pread(x.fd,x.staging+size_t(x.miss_slots[j])*BYTES,BYTES,off_t(x.miss_ids[j])*BYTES);}while(got<0&&errno==EINTR);
    count++;if(got>0)bytes+=got;if(got!=BYTES)error=got<0?errno:EIO;
   }
   {std::lock_guard<std::mutex> l(mutex);x.calls+=count;x.bytes+=bytes;if(error)x.error=error;
    if(!--remaining){finished=true;done.notify_all();}}
  }});
 }
};
extern "C" {
void* fc_create(){return new Engine;}
int fc_bind(void* p,int id,const char* file,int dim,const int32_t* features,const void* weights,const float* scales,
            const int32_t* hot,uint16_t* hidden,uint8_t* staging,int32_t* desc){
 auto& e=*(Engine*)p;auto& x=e.layers[id];x.fd=open(file,O_RDONLY|O_DIRECT);if(x.fd<0)return errno;
 if(uintptr_t(staging)%512)return EINVAL;
 x.dim=dim;x.stride=(dim+31)/32*32;x.features=features;x.weights=(const uint8_t*)weights;x.scales=scales;
 x.hidden=hidden;x.staging=staging;x.desc=desc;x.hot_slot.fill(-1);
 for(int j=0;j<HOT;j++){if(hot[j]<0||hot[j]>=I||x.hot_slot[hot[j]]>=0)return EINVAL;x.hot_slot[hot[j]]=j;}
 cpu_set_t mask;sched_getaffinity(0,sizeof mask,&mask);
 x.predictor=ls_create(I,x.stride,K,12,4);sched_setaffinity(0,sizeof mask,&mask);
 if(!x.predictor)return EINVAL;
 return cudaEventCreateWithFlags(&x.input,cudaEventDisableTiming);
}
int fc_start(void* p,int id,const void* gpu,void* stream){
 auto& e=*(Engine*)p;auto& x=e.layers[id];std::lock_guard<std::mutex> l(e.mutex);
 if(!e.finished)return EBUSY;
 int rc=cudaMemcpyAsync(x.hidden,gpu,8192,cudaMemcpyDeviceToHost,(cudaStream_t)stream);
 if(!rc)rc=cudaEventRecord(x.input,(cudaStream_t)stream);if(rc)return rc;
 e.layer=id;e.finished=false;e.job=true;e.work.notify_all();return 0;
}
int fc_wait(void* p){auto& e=*(Engine*)p;std::unique_lock<std::mutex> l(e.mutex);e.done.wait(l,[&]{return e.finished;});return e.layers[e.layer].error;}
int fc_counts(void* p,int id,uint64_t* out){auto& x=((Engine*)p)->layers[id];out[0]=K;out[1]=K-x.misses;out[2]=x.misses;out[3]=x.bytes;out[4]=x.calls;return 0;}
void fc_destroy(void* p){auto& e=*(Engine*)p;{std::lock_guard<std::mutex> l(e.mutex);e.stop=true;e.work.notify_all();}
 e.predictor.join();for(auto& t:e.readers)t.join();for(auto& x:e.layers){if(x.predictor)ls_destroy(x.predictor);if(x.fd>=0)close(x.fd);if(x.hidden)cudaEventDestroy(x.input);}delete &e;}
}
