# Optional reproduction notices

`scoring/coqa.py` preserves the historical scorer byte-for-byte (SHA256 `62ac82e8027162d134709355a00526ff5497deb5987beebb18b439b320e860ca`). Its source matches the [EleutherAI evaluation harness CoQA utility](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/coqa/utils.py), copyright 2020 EleutherAI. Its MIT license is retained in `scoring/LICENSE`. Transformer SQuAD metric functions are imported from the separately installed Apache-2.0 Transformers dependency; their source is not vendored here.

The historical baseline implementation is project source exported from revision `9533c348a5c4b3da796fb8ab4bd0e2303d578c99`. Native baseline source SHA256: `9f3c48943b67a2a43e42e27982caec6cd01b116d444f4cc93ddb2f3491ae4d5a`. Original Python source SHA256: `b7eb9e358b947a9396a4c1c3032346576e5d67e7160929dcea76ad9748351dfd`. Only Python imports and the mapped-helper library argument change in the public adapter; numerical function bodies and native source are unchanged.

Model-derived data and CoQA contents are not licensed by the project MIT license. See the reproduction guide for the separate, pending artifact redistribution review. No model weights or dataset contents are included in this source directory.

## Model-derived artifacts

The supported checkpoint is [cyankiwi/Ministral-3-8B-Instruct-2512-AWQ-4bit](https://huggingface.co/cyankiwi/Ministral-3-8B-Instruct-2512-AWQ-4bit/tree/e3a799844ba08f27fddbf118e154057eb0105889), pinned to revision `e3a799844ba08f27fddbf118e154057eb0105889`. Its model card credits Ton Cao (cyankiwi) for the quantized checkpoint, identifies Mistral AI's `mistralai/Ministral-3-8B-Instruct-2512` as its base, and declares Apache-2.0. The Apache-2.0 text is included in `notices/Apache-2.0.txt` and in the artifact collection as `LICENSES/Apache-2.0.txt`.

LeanStream's Group2 packed MLP files are reordered and requantized signed INT4/per-128 derivatives, with gate/up/down rows and predictor-dependent physical ordering. The individual-neuron baseline packed storage is also derived. These files are not unchanged upstream tensors. Preserve their identities, this modification attribution, applicable upstream notices and the Apache-2.0 license when distributing them. The upstream checkpoint remains a separate pinned download.

The project authors created the full/P1-lite calibrated predictor parameters and fixed-hot artifacts. Public distribution permission and the applicable artifact license still require explicit approval; source-code MIT licensing is not an automatic grant for these data files.

`metadata/predictor_capture.json` binds calibration records 0 and 1 to the historical CoQA documents; `coqa_selection.json` identifies both as MCTest records. The [original MCTest MSR-LA](https://github.com/mcobzarenco/mctest/blob/master/data/MCTest/LICENSE.pdf) includes same-terms requirements for redistributed data derivatives. Whether those requirements apply to the calibrated parameters/fixed-hot artifacts requires a rights determination before assigning their public artifact license. Project ownership and omission of raw passages do not by themselves resolve that question. No claim is made here that learned parameters are necessarily data derivatives, or that they are exempt.

## Dataset inputs

No CoQA passages, answers or prompt token IDs are included in release archives. Users explicitly download the original dataset and reconstruct the historical inputs locally. [CoQA's original terms](https://stanfordnlp.github.io/coqa/#license) include CC BY-SA 4.0, MSR-LA, RACE and CNN/Apache terms. Neither the source-code license nor tokenization replaces those terms. The historical final-question, multi-reference scorer is distinct from the official all-turn benchmark protocol.
