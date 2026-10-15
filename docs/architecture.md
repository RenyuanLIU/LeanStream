# Architecture

The reusable control facade is `Runtime` / `LayerStep`. Each layer owns a `WeightManager`; the application supplies storage bindings and a `ComputeBackend`. Logical group identity is independent of physical file order and execution destination slots.

The predictor produces ordered logical IDs. `update_prediction` immediately submits a newer priority generation to I/O. The I/O engine runs independently. `finalize` freezes required membership, assigns deterministic execution destinations, and makes ready work visible through `ExecDesc` and `DeviceExecQueue`.

`ComputeBackend::enqueue` is the only normal GPU submission boundary. It receives layer, input/output, stream, core descriptors, queue, epoch, and consumer mode. The production Triton backend selects a backend candidate and submits the bound A, persistent B, and reduction on that stream. Without a measured table, it retains 32 persistent B CTAs. The CUDA reference helper has a separate four-worker fallback and can demonstrate LUT selection. Runtime has no knowledge of worker counts or candidate layout.

Prepare and submit future layer work while prior GPU work is running. Use the latest state already available at that point. Neither prediction callbacks nor storage callbacks submit GPU kernels; they update priorities and publish readiness. Finalization can occur after submission. The consumer waits for published final membership and ready descriptors on the GPU.

Keep buffers and mappings alive until GPU completion and transport callback retirement. `commit` installs application-selected resident groups through the backend and retires the step; `abort` also requires proof that submitted work has completed. `drain` is a lifecycle boundary, not a per-layer scheduling barrier. Serialize control operations for each LayerStep; only use the concurrency contracts explicitly provided by queue publication and transport callbacks.

The v0.1 backend uses fixed logical staging storage. Strict physical cache-capacity enforcement and bounded staging are separate ongoing optimizations.

The optional `examples/ministral` adapter preserves the validated REPAIRED_KNEE admitted-read union and original activation-based retention while linking public I/O and GPU components. It is model policy, not a second WeightManager or learned predictor inside the manager. See [production Triton](triton_backend.md).
