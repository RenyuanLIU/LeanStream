#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../include/leanstream/execution_queue.h"
namespace leanstream::runtime {
struct MLPBuffers {
    const __half* input;
    const uint8_t* cache;
    const uint8_t* staging;
    const __half* scales;
    float* partial; // deterministic per-group dest_slot, WIDTH float values
    uint32_t* executions; // optional tests only; nullptr in production
    uint64_t* first_compute; // optional GPU globaltimer per destination
};
// Exactly A -> B -> reduction on caller stream; no CPU callback launches.
cudaError_t enqueue_streaming_mlp(Span<ExecDesc> device_core, DeviceExecQueue device_queue,
    MLPBuffers, __half* output, uint32_t epoch, uint32_t width, uint32_t group_size,
    uint32_t workers, ConsumerMode, cudaStream_t);
}
