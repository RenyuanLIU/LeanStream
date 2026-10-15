# Dynamic Weight / I/O Manager

`ManagerConfig` specifies logical group count, group byte size, staging stride, alignment, queue depth, and consumer mode. `LayerStorage` borrows the file descriptor, logical-to-physical layout, cache-slot map, staging storage, I/O state words, optional cached core IDs, and execution queue storage. Keep bindings immutable through the step. Provide at least `group_count * staging_stride` bytes of staging; set `staging_bytes` to enable explicit capacity validation.

`begin_layer` starts an epoch. `update_prediction` consumes ranked logical IDs and advances the prediction generation immediately. Newer generations replace/reorder unstarted speculative requests. Started reads are non-preemptive. READY groups and cache hits are reused. `PredictionUpdateOptions::io_budget` limits new EMPTY admissions; cache, READY, and INFLIGHT hits do not spend that budget.

`finalize(final_groups)` freezes required membership, independently of speculative order. Required core membership must remain consistent with the pre-enqueued core plan. Only final required groups become execution descriptors. Completion of speculative reads does not add execution membership. Storage completion publishes readiness and never launches GPU work. Finalization does not wait for file reads to finish.

`SourceKind::Cache` and `SourceKind::Staging` select the descriptor's storage source. Retirement requires the application to prove GPU completion first. `wait_io` is for diagnostics and CPU examples; do not insert it into the enqueue path. `prepare_layer` optionally registers immutable layout/plan metadata during idle time; bump generations when those bindings change.

The reference `src/io/stream_io.*` transport uses aligned `pread` workers. It preserves the scheduler's non-preemptive read semantics. Its aggregate `StreamStatus` snapshot uses a scheduler lock, includes speculative traffic, and is not an exact future-layer remaining-required count. It is not used to make kernel decisions. The public WeightManager exposes per-group state and read-request counts, but no exact constant-time remaining-required-I/O snapshot. A custom `IOBackend` can provide application-owned cheap progress separately without coupling selector policy to transport policy.

The v0.1 reference STREAM_IO backend uses fixed logical staging storage. Strict physical cache-capacity enforcement and bounded staging are not included. Do not interpret logical residency as a physical memory-capacity guarantee.

Run `python3 tests/run_io_example.py ./build/dynamic_io_example dynamic` for two immediate priority updates, final consumption, and retirement using a small synthetic aligned file. The CPU example has no GPU users and can retire after its diagnostic I/O wait. Runtime tests use controlled reads to verify replacement, reuse, final membership, cancellation, and recovery.
