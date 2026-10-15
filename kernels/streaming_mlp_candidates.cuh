#pragma once
#include "streaming_mlp.cuh"
#include "leanstream/kernel_selector.h"
namespace leanstream::runtime {
// Call while preparing GPU submission. The caller supplies its latest cheap state.
// This helper neither waits for progress nor observes GPU state.
cudaError_t enqueue_selected_streaming_mlp(const KernelSelector&,KernelRuntimeState,
    Span<ExecDesc>,DeviceExecQueue,MLPBuffers,__half* output,uint32_t epoch,
    uint32_t width,uint32_t group_size,ConsumerMode,cudaStream_t);
}
