#include "streaming_mlp_candidates.cuh"
namespace leanstream::runtime {
cudaError_t enqueue_selected_streaming_mlp(const KernelSelector& selector,KernelRuntimeState state,
    Span<ExecDesc> core,DeviceExecQueue queue,MLPBuffers buffers,__half* output,uint32_t epoch,
    uint32_t width,uint32_t group_size,ConsumerMode mode,cudaStream_t stream){
    auto c=selector.plan(state).candidate;
    // v0.1 backend only implements the five worker configurations and entry point zero.
    if((c.workers!=1&&c.workers!=2&&c.workers!=4&&c.workers!=8&&c.workers!=16) ||
        c.cta_count||c.tile_size||c.unroll||c.group_batch||c.entry_point)c=kDefaultKernelCandidate;
    return enqueue_streaming_mlp(core,queue,buffers,output,epoch,width,group_size,c.workers,mode,stream);
}
}
