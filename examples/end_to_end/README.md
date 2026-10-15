# CUDA reference integration example

This synthetic example exercises the CUDA reference backend. It does not measure the production Triton path or model throughput. For real-model Progressive execution, use [the optional Ministral recipe](../ministral/README.md).

After the CUDA build:

```sh
python3 tests/run_io_example.py ./build-cuda/end_to_end_example cuda
```

The harness supplies a fresh file path and removes the synthetic data afterward. The example connects CPU predictor priorities to Runtime/WeightManager, immediate refinements to aligned file reads, final membership to ready descriptors, and KernelSelector to the same streaming MLP arithmetic. A compute backend uses an explicitly illustrative bucket zero; real integrations supply their latest cheap state without waiting.

The backend submits core, persistent consumer, and reduction before finalization. Timing-disabled completion events guard storage lifetime and commit. Correctness checks cover finite numeric results, exactly-once final groups, cancellation, short reads, invalid finalization, and recovery. Cache commit here only retains the already cached group 0; applications implement their own cache installation policy.
