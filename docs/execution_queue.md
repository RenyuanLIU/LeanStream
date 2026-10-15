# Execution queue

`ExecDesc` carries logical group, physical group, source slot, destination slot, and `SourceKind`. Logical IDs determine semantics; physical IDs locate scale rows/file layout. Destination slots provide deterministic reduction order. Cache and staging byte addresses use their respective source slots.

`DeviceExecQueue` borrows descriptor storage, ready-index storage, a cache-line-aligned `QueueControl`, and capacity. These buffers must be visible and coherent to the GPU. CPU-only tests use ordinary host storage; CUDA examples use mapped pinned memory.

`ExecutionQueue::reset(epoch)` is legal only after the previous epoch's GPU users and callbacks retire. Prepare and fill unpublished descriptors, then activate/seal the plan. Serialized producers publish indices only after descriptor metadata and weight bytes are visible. CPU publication uses release ordering; CUDA consumers acquire at system scope. Do not reuse slots or change descriptors while the consumer holds them.

The queue tracks final total count separately from ready count. Progressive consumers claim published indices once. Batch-compatible consumers wait for batch publication; Auto conservatively follows batch behavior. Error publication terminates waiting consumers; failed output must be rejected. The application still must wait for GPU completion before freeing storage.

Queue publication allocates no storage. Reset, publication, and lifecycle sequencing are contracts, not a general-purpose concurrent-container API.
