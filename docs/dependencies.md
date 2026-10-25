# Dependencies and licenses

LeanStream project source is distributed under the MIT license in the repository root. No external dependency source tree or binary distribution is bundled. The production kernels use the public Triton language and CUDA interfaces; dependency names do not grant rights to redistribute model weights or datasets.

- Triton: upstream 3.1.0, MIT license. On Jetson, use an upstream aarch64 source build; a compatible wheel may not be available.
- PyTorch: BSD-style license and its third-party notices. The validated Jetson environment uses the vendor 2.1.0a0 build; do not replace a working Jetson CUDA/PyTorch installation just to run the examples.
- Transformers: Apache-2.0, validated model-loader version 4.41.1.
- safetensors: Apache-2.0, used only by the optional model loader.
- NumPy: BSD-3-Clause.
- CUDA Toolkit/driver: NVIDIA's separately licensed development/runtime dependencies; headers and libraries are consumed from the installation, not copied into this repository.
- OpenMP: supplied by the compiler/runtime distribution, required only for the included CPU predictor.

Retain each dependency's notices and comply with its redistribution terms when distributing a combined binary/environment. Project source contains no imported dependency license notices that have been removed. The mapped tensor bridge implements the DLPack ABI layout; it does not bundle the DLPack source distribution.

## Jetson environment

Keep a supported Jetson CUDA 11.8 and PyTorch installation. Use a separate Python environment that can access the vendor PyTorch packages, then install compatible Python dependencies:

```sh
python3 -m venv --system-site-packages /path/to/leanstream-env
/path/to/leanstream-env/bin/pip install 'numpy<2' 'transformers==4.41.1' safetensors
```

Build upstream Triton 3.1.0 for aarch64 following its build requirements. A source checkout needs build tools, compatible LLVM development/build resources, and sufficient disk space. The validated upstream source revision is `cf34004b8a67d290a962da166f5aa2fc66751326`:

```sh
git clone https://github.com/triton-lang/triton.git /path/to/triton
git -C /path/to/triton checkout cf34004b8a67d290a962da166f5aa2fc66751326
/path/to/leanstream-env/bin/pip install --no-build-isolation -e /path/to/triton/python
```

This is an external dependency checkout. No private LeanStream source is required. Check `import torch, triton; print(torch.__version__, triton.__version__)` before building the optional native backend. Standard CUDA paths such as `/usr/local/cuda-11.8/bin/nvcc` may need to be selected explicitly.

## Optional complete Ministral reproduction

Use the pinned model, nine checksummed v2 asset archives and exact software/hardware
recipe in [the reproduction guide](reproduce_ministral3_8b.md). This adds only
optional Hugging Face/tokenizers/scoring dependencies; generic CPU builds do not
import or download them. Production kernels are included in backends/triton/.
The default One-shot uses its distinct downloaded1638-feature early-input W4A8
parameters. No training or model conversion is part of the reproduction workflow.

The complete v2 run additionally verified tokenizers0.19.1 and huggingface-hub0.23.1.
The observed vendor torch build was2.1.0a0+41361538.nv23.6.
