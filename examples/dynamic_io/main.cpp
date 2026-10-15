// Minimal real-SSD manager example: usage FILE GROUPS BYTES.
#include "leanstream/weight_manager.h"
#include <fcntl.h>
#include <unistd.h>
#include <cstdlib>
#include <vector>
#include <cstdio>
#include <chrono>
#include <thread>
using namespace leanstream::runtime;
int main(int argc,char** argv){
 if(argc!=4)return 2;int n=std::atoi(argv[2]),bytes=std::atoi(argv[3]);if(n<4||bytes<=0||bytes%512)return 2;
 int fd=open(argv[1],O_RDONLY|O_DIRECT);if(fd<0)return 2;void* arena=nullptr;
 if(posix_memalign(&arena,512,size_t(n)*bytes)){close(fd);return 2;}
 std::vector<GroupLayout> layout(n);std::vector<SlotId> cache(n,-1);std::vector<uint32_t> states(n+1),indices(n);std::vector<ExecDesc> desc(n);QueueControl ctl;
 for(int i=0;i<n;i++)layout[i]={uint32_t(i),uint32_t(i),uint64_t(i)*bytes,uint32_t(bytes)};
 void* hot=nullptr;if(posix_memalign(&hot,512,bytes)){close(fd);free(arena);return 1;}if(pread(fd,hot,bytes,0)!=bytes){perror("cache population");close(fd);free(arena);free(hot);return 1;}cache[0]=0; // initial cache population
 {WeightManager manager({uint32_t(n),1,uint64_t(bytes),uint64_t(bytes),512});
 LayerStorage storage{fd,{layout.data(),layout.size()},cache.data(),arena,states.data(),{}, {desc.data(),indices.data(),&ctl,uint32_t(n)}};
 int rc=manager.begin_layer(0,storage);bool begun=!rc;GroupId p0[]={0,1,2},p1[]={0,1,3},final[]={1,3};
 if(!rc)rc=manager.update_prediction({p0,3});
 // Refinement submits new priority immediately; no I/O completion wait.
 if(!rc)rc=manager.update_prediction({p1,3});if(!rc)rc=manager.finalize({final,2});if(!rc)rc=manager.wait_io(5000);
 // CPU-only example: no GPU retains these slots, so retirement is safe here.
 if(begun){int ended=manager.retire();if(!rc)rc=ended;}printf("status=%d generations=%llu read_requests=%llu ready=%u cache_reads=0\n",rc,(unsigned long long)manager.prediction_generation(),(unsigned long long)manager.read_requests(),ctl.ready_count);
 if(rc)return 1;}
 close(fd);free(arena);free(hot);return 0;
}
