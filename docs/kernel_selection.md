# I/O-aware GPU kernel selection

LeanStream performs no kernel profiling on the decode critical path. The selection pipeline is:

```text
offline candidate characterization -> offline decision-table synthesis
                                                   |
latest available I/O progress                       v
  -> remaining-load ratio -> quantized q_n -> LUT[layer][q_n]
  -> candidate ID -> selected GPU configuration -> pre-enqueue
```

The runtime state is conceptually `s_n = (n, q_n)`: layer index and quantized remaining-I/O load. Candidate performance must be measured offline on the target platform and used offline to populate the LUT. Runtime only consumes that immutable table. There is no CUDA-event profiling, launch-time candidate benchmarking, trial execution, online search, or decode-time autotuning.

`KernelSelector::select(layer, io_bucket)` performs row-major direct indexing with constant-time range checks and direct candidate indexing. No allocation, strings, scans, locks, CUDA calls, or synchronization occur in the lookup. Table bytes and candidate registry are borrowed and must remain immutable and alive for the selector's lifetime. The caller must supply `layers * io_buckets` valid bytes. Use dense candidate IDs matching registry indices. The fallback is supplied by the backend. The CUDA reference catalog reserves ID 2 for four workers; the production Triton catalog reserves ID 3 for 32 persistent B CTAs.

`KernelCandidate` carries ID, workers, name, availability, and reserved backend configuration fields for CTA count, tile size, unroll, group batch, and entry-point ID. The CUDA reference catalog varies 1, 2, 4, 8, or 16 workers. The Triton catalog varies 4, 8, 16, or 32 persistent B CTAs, each with four warps per CTA. These are resource configurations of the same mathematical kernel. `KernelPlan` packages the selected configuration. A backend can interpret future fields without changing the selector contract.

`quantize_io_load(remaining, total, buckets=8)` returns `min(buckets-1, floor(min(remaining,total)*buckets/total))`. Bucket 0 is the least remaining load; bucket 7 is the largest for eight buckets. Exact full load and values above total map to the last bucket. Zero total or zero bucket count returns zero. Multiplication widens to 64 bits to avoid overflow. With eight buckets, interior boundaries are multiples of 1/8. Match helper bucket count to the offline table definition.

The public Runtime boundary has no cheap exact remaining-required-I/O counter. Per-group scans and locked transport diagnostics are unsuitable for selection. Supply the latest cheap application-owned `remaining,total` snapshot, a cheap estimate, or an already quantized bucket at preparation/submission time. Do not claim this API independently measures I/O state. If an integration has no progress feed, select the default configuration or use an explicitly documented static bucket. The end-to-end example uses illustrative bucket zero; it does not infer live remaining load.

Do not wait for a perfect future state. While layer L runs, the CPU can choose and submit future layer work from its latest available snapshot. Prediction completion updates priority only. I/O completion publishes ready work only. Neither completion launches GPU work. The GPU may wait on data readiness, but never on CPU profiling or candidate search.

`configs/example_kernel_lut.h` contains **synthetic placeholder values**, flattened as three layers by eight buckets in row-major order. It is an API demonstration, not a production-optimal or platform-tuned table. Entry `layer * io_buckets + bucket` contains a uint8 candidate ID. The example prints configurations without timing claims.

An absent table, invalid layer/bucket, invalid table entry, absent/unavailable candidate, ID mismatch, or zero worker count returns the backend default (32 persistent B CTAs for production Triton; four workers for CUDA reference). The CUDA wrapper also falls back for unsupported configuration fields or worker counts. The Triton backend requires a matching compiled B specialization for every candidate; an unbound candidate falls back to its bound 32-CTA default. It never changes only the grid of a differently specialized kernel. No inference failure is introduced solely by an invalid selection. Malformed borrowed storage pointers remain a caller error.

## Production fallback and explicit tables

Use `triton_selector()` / `TritonBackend` without a table for validated 32-CTA production execution. CUDA's four-worker default is not a production Triton default. `warps_per_cta=4` means 128 threads in the production Triton kernels; it does not mean four persistent CTAs.

Pass `KernelDecisionTable` explicitly to the C++ backend, or `offline_table=` to the Python `NativeBackend`, only after offline characterization and specialization preparation. The synthetic CUDA example table is never automatically consumed by the production backend. No measured production decision LUT is distributed. Default inference preserves the validated production configuration, without tuning or search.
