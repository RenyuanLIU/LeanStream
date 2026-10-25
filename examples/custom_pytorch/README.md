# Optional PyTorch integration example

Keep your model; replace only its MLPs with the existing LeanStream runtime and GPU backend. No downloaded checkpoint or trained predictor is required for this tiny tutorial.

```sh
export PYTHONPATH="$PWD"
export PYTHONDONTWRITEBYTECODE=1
export TRITON_PTXAS_PATH=/usr/local/cuda-11.8/bin/ptxas
python3 integrations/torch/build.py --output /data/ls-build
python3 examples/custom_pytorch/run.py --mode inspect
python3 examples/custom_pytorch/run.py --mode validate --library /data/ls-build/libleanstream_torch.so --output /data/new-toy-run
```

Use a new output path on an O_DIRECT-capable SSD and the validated Jetson SM87 environment. `basic` waits for I/O; `progressive` publishes early priorities and pre-enqueues READY-driven GPU work before final selection. Both execute the real public Runtime/WeightManager/ExecutionQueue/STREAM_IO/Triton/native backend and compare independent numerical results. Prefill retains the original dense MLP; no low-memory-loading claim is made.

See [PyTorch integration](../../docs/integrating_pytorch.md) for manual configuration, predictor readiness, inspection, Qwen requirements, cleanup and the support matrix. See [custom Triton](../../docs/custom_triton.md) and [editable templates](../../templates/triton/README.md) for a complete arithmetic specialization. `attach_model.py` is an opt-in local-HF-model recipe requiring user-supplied compatible weights and predictors; it is not validated Qwen E2E support.
