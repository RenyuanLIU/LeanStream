// Real public Runtime + unchanged production pread backend + numeric CUDA MLP.
// Usage: runtime_cuda /absolute/path/to/NEW_demo_weights.bin
// Small format demonstration; its timing is never model performance evidence.
#include "leanstream/runtime.h"
#include "../../kernels/streaming_mlp_candidates.cuh"
#include "configs/example_kernel_lut.h"
#include "leanstream/predictor.h"
#include <fcntl.h>
#include <unistd.h>
#include <cassert>
#include <cstdio>
#include <vector>
#include <cmath>
#include <chrono>
#include <thread>
#include <atomic>
#ifdef STREAM_TESTING
static std::atomic<bool> injected_read_error{false};
extern "C" ssize_t stream_test_pread(int fd,void* bytes,size_t n,off_t offset){if(injected_read_error.load()){errno=EIO;return -1;}return pread(fd,bytes,n,offset);}
#endif
using namespace leanstream::runtime;
#define CUDA(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));abort();}}while(0)
template<class T>T* mapped(size_t n){T* p;CUDA(cudaHostAlloc(&p,n*sizeof(T),cudaHostAllocMapped));return p;}
template<class T>T* device(size_t n){T* p;CUDA(cudaMalloc(&p,n*sizeof(T)));return p;}
struct Backend {
    KernelSelector selector{leanstream::example::table()};
    uint32_t io_bucket=0; // Caller-provided illustrative state; not a Runtime progress measurement.
    MLPBuffers buffers{};cudaEvent_t finished;uint32_t width,group_size;cudaStream_t active_stream=nullptr;bool recorded=false;
    static int enqueue(void* ptr,uint32_t layer,const void* input,void* output,void* stream,
                       Span<ExecDesc> core,DeviceExecQueue q,uint32_t epoch,ConsumerMode mode){
        auto& b=*static_cast<Backend*>(ptr);b.buffers.input=static_cast<const __half*>(input);
        auto s=static_cast<cudaStream_t>(stream);
        auto rc=enqueue_selected_streaming_mlp(b.selector,{layer,b.io_bucket},core,q,b.buffers,static_cast<__half*>(output),epoch,b.width,b.group_size,mode,s);
        b.active_stream=s;auto event_rc=cudaEventRecord(b.finished,s);b.recorded=event_rc==cudaSuccess;
        return int(rc!=cudaSuccess?rc:event_rc);
    }
    static bool complete(void* ptr,uint32_t){auto& b=*static_cast<Backend*>(ptr);return (b.recorded?cudaEventQuery(b.finished):cudaStreamQuery(b.active_stream))==cudaSuccess;}
    static int commit(void*,uint32_t,Span<GroupId> ids){return ids.size==1&&ids[0]==0?0:EINVAL;}
};
int main(int argc,char** argv){
    if(argc!=2){fprintf(stderr,"Provide a NEW output file path\n");return 2;}
    constexpr uint32_t N=8,H=512,G=2,BYTES=G*3*H/2;
    int write_fd=open(argv[1],O_WRONLY|O_CREAT|O_EXCL,0600);if(write_fd<0){perror("create");return 2;}
    std::vector<uint8_t> packed(N*BYTES,0x11);assert(write(write_fd,packed.data(),packed.size())==ssize_t(packed.size()));close(write_fd);
    int fd=open(argv[1],O_RDONLY|O_DIRECT);if(fd<0){perror("O_DIRECT");return 2;}
    auto* stage=mapped<uint8_t>(N*BYTES);auto* states=mapped<uint32_t>(N+1);
    auto* desc=mapped<ExecDesc>(N);auto* core=mapped<ExecDesc>(1);auto* indices=mapped<uint32_t>(N);auto* control=mapped<QueueControl>(1);*control=QueueControl{};
    auto* counts=mapped<uint32_t>(N);for(uint32_t i=0;i<N;i++)counts[i]=0;
    auto* cache=device<uint8_t>(BYTES);CUDA(cudaMemcpy(cache,packed.data(),BYTES,cudaMemcpyHostToDevice));
    auto* scales=device<__half>(N*G*3*H/128);std::vector<__half> scale(N*G*3*H/128,__float2half(.05f));CUDA(cudaMemcpy(scales,scale.data(),scale.size()*2,cudaMemcpyHostToDevice));
    auto* input=device<__half>(H);std::vector<__half> x(H,__float2half(.1f));CUDA(cudaMemcpy(input,x.data(),H*2,cudaMemcpyHostToDevice));
    auto* output=device<__half>(H);auto* partial=device<float>(N*H);
    std::vector<GroupLayout> layout;SlotId slots[N];for(uint32_t i=0;i<N;i++){layout.push_back({i,i,uint64_t(i)*BYTES,BYTES});slots[i]=i? -1:0;}
    GroupId keep[]={0},refine[]={0,2,4,6},final[]={0,6,2,4};
    Backend backend;backend.width=H;backend.group_size=G;backend.buffers={input,cache,stage,scales,partial,counts,nullptr};CUDA(cudaEventCreateWithFlags(&backend.finished,cudaEventDisableTiming));
    cudaStream_t stream;CUDA(cudaStreamCreate(&stream));
    {
        Runtime runtime({N,4,BYTES,BYTES,512,ConsumerMode::Progressive},{&backend,Backend::enqueue,Backend::complete,Backend::commit});
        LayerStorage storage{fd,{layout.data(),layout.size()},slots,stage,states,{keep,1},{desc,indices,control,N},core};
        LayerStep step;assert(runtime.begin_layer(0,storage,&step)==0);
        // Synthetic predictor rows correspond directly to logical group IDs.
        float pw[N*2*32],px[32];int32_t priorities[4];
        for(auto& v:px)v=1;
        for(uint32_t r=0;r<N;r++)for(int j=0;j<64;j++)pw[r*64+j]=float(r+1)/32;
        void* predictor=ls_create(N,32,4,1,32);assert(predictor);
        assert(!ls_predict(predictor,pw,nullptr,px,1,nullptr,0,0,priorities,nullptr,nullptr,nullptr));
        assert(step.update_prediction({priorities,4})==0);ls_destroy(predictor);
        assert(step.enqueue_compute(input,output,stream)==0);
        assert(step.update_prediction({refine,4})==0);
        assert(step.finalize({final,4})==0);
        // Algorithmic completion at the application boundary, no per-layer timing sync.
        CUDA(cudaEventSynchronize(backend.finished));assert(step.commit({keep,1})==0);assert(runtime.drain()==0);
        for(uint32_t i=0;i<4;i++)assert(counts[i]==1);
        for(uint32_t i=4;i<N;i++)assert(counts[i]==0);
        std::vector<__half> result(H);CUDA(cudaMemcpy(result.data(),output,H*2,cudaMemcpyDeviceToHost));float gate=__half2float(__float2half(H*__half2float(x[0])*__half2float(scale[0])));
        float silu=__half2float(__float2half(gate/(1+expf(-gate))));
        float activation=__half2float(__float2half(silu*gate));
        float expected=__half2float(__float2half(4*G*activation*__half2float(scale[0])));
        for(auto y:result){assert(std::isfinite(__half2float(y)));assert(fabs(__half2float(y)-expected)<.005f);}
        assert(control->ready_count==3&&runtime.weights().prediction_generation()==2);
        // Adversarial real-backend failures; errors never become valid output.
        for(int failure=0;failure<3;failure++) {
#ifndef STREAM_TESTING
            if(failure==1)continue;
#endif
            int test_fd=dup(fd);assert(test_fd>=0);storage.fd=test_fd;
            assert(!runtime.begin_layer(0,storage,&step));assert(!step.enqueue_compute(input,output,stream));
            int expected=0;
            if(failure==0){GroupId invalid[]={6};assert(step.finalize({invalid,1})==EINVAL);expected=EINVAL;}
            else {
                if(failure==1){
#ifdef STREAM_TESTING
                    injected_read_error=true;
#endif
                    expected=EIO;}
                else {int writer=open(argv[1],O_WRONLY);assert(writer>=0);assert(!ftruncate(writer,BYTES));close(writer);expected=EIO;}
                GroupId missing[]={0,7};assert(!step.finalize({missing,2}));
            }
            auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
            while(!Backend::complete(&backend,0)){assert(std::chrono::steady_clock::now()<deadline);std::this_thread::sleep_for(std::chrono::milliseconds(1));}
            assert(__atomic_load_n(&control->error,__ATOMIC_ACQUIRE)==uint32_t(expected));
            assert(step.commit({keep,1})!=0);int rc=step.abort();assert(rc==0||rc==expected);
            if(test_fd>=0)close(test_fd);
#ifdef STREAM_TESTING
            injected_read_error=false;
#endif
        }
        // Restore this test-owned file and prove a new epoch can run normally.
        int writer=open(argv[1],O_WRONLY);assert(writer>=0);assert(write(writer,packed.data(),packed.size())==ssize_t(packed.size()));close(writer);
        storage.fd=fd;assert(!runtime.begin_layer(0,storage,&step));assert(!step.enqueue_compute(input,output,stream));assert(!step.finalize({final,4}));
        CUDA(cudaEventSynchronize(backend.finished));assert(!step.commit({keep,1}));assert(!runtime.drain());
        puts("PASS: finalize cancellation, pread/short-read errors, finite persistent exit, invalid output rejected, recovery");
        puts("PASS: public Runtime, immediate prediction updates, real O_DIRECT I/O, final-only INT4 MLP, A/B/reduction, exact execution counts");
    }
    close(fd);CUDA(cudaStreamDestroy(stream));CUDA(cudaEventDestroy(backend.finished));
    CUDA(cudaFree(input));CUDA(cudaFree(output));CUDA(cudaFree(partial));CUDA(cudaFree(cache));CUDA(cudaFree(scales));
    CUDA(cudaFreeHost(stage));CUDA(cudaFreeHost(states));CUDA(cudaFreeHost(desc));CUDA(cudaFreeHost(core));CUDA(cudaFreeHost(indices));CUDA(cudaFreeHost(control));CUDA(cudaFreeHost(counts));
}
