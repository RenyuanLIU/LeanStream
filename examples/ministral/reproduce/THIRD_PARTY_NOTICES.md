# Optional reproduction notices

`scoring/coqa.py` preserves the historical scorer byte-for-byte (SHA256 `62ac82e8027162d134709355a00526ff5497deb5987beebb18b439b320e860ca`). Its source matches the [EleutherAI evaluation harness CoQA utility](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/coqa/utils.py), copyright 2020 EleutherAI. Its MIT license is retained in `scoring/LICENSE`. Transformer SQuAD metric functions are imported from the separately installed Apache-2.0 Transformers dependency; their source is not vendored here.

The historical baseline implementation is project source exported from revision `9533c348a5c4b3da796fb8ab4bd0e2303d578c99`. Native baseline source SHA256: `9f3c48943b67a2a43e42e27982caec6cd01b116d444f4cc93ddb2f3491ae4d5a`. Original Python source SHA256: `b7eb9e358b947a9396a4c1c3032346576e5d67e7160929dcea76ad9748351dfd`. Only Python imports and the mapped-helper library argument change in the public adapter; numerical function bodies and native source are unchanged.

Model-derived data and CoQA contents are not licensed by the project MIT license. See the reproduction guide for the separate model/dataset terms described in the public v2 reproduction guide. No model weights or dataset contents are included in this source directory.

## Model-derived artifacts

The supported checkpoint is [cyankiwi/Ministral-3-8B-Instruct-2512-AWQ-4bit](https://huggingface.co/cyankiwi/Ministral-3-8B-Instruct-2512-AWQ-4bit/tree/e3a799844ba08f27fddbf118e154057eb0105889), pinned to revision `e3a799844ba08f27fddbf118e154057eb0105889`. Its model card credits Ton Cao (cyankiwi) for the quantized checkpoint, identifies Mistral AI's `mistralai/Ministral-3-8B-Instruct-2512` as its base, and declares Apache-2.0. The Apache-2.0 text is included in `notices/Apache-2.0.txt` and in the artifact collection as `LICENSES/Apache-2.0.txt`.

LeanStream's Group2 packed MLP files are reordered and requantized signed INT4/per-128 derivatives, with gate/up/down rows and predictor-dependent physical ordering. The individual-neuron baseline packed storage is also derived. These files are not unchanged upstream tensors. Preserve their identities, this modification attribution, applicable upstream notices and the Apache-2.0 license when distributing them. The upstream checkpoint remains a separate pinned download.

The project authors created the full/P1-lite calibrated predictor parameters and fixed-hot artifacts. The project owner authorizes public distribution subject to the retained Apache-2.0 model terms and MSR-LA same-terms for calibration-derived project contributions; source-code MIT licensing does not replace these terms.

`metadata/predictor_capture.json` and `coqa_selection.json` identify calibration records0 and1 as MCTest records. The original [MSR-LA](notices/MCTest-MSR-LA.pdf) permits distribution subject to its terms, including same-terms for data derivatives, modification notices and other conditions. The v2 package conservatively retains those same terms for calibration-derived project contributions and includes the original license, dated modification notices and Apache-2.0 model attribution. No claim of blanket MIT licensing or exemption for learned parameters is made. Raw data and paired activations are excluded. See the collection's LICENSE_OR_NOTICE for scope and provenance.

## Dataset inputs

No CoQA passages, answers or prompt token IDs are included in release archives. Users explicitly download the original dataset and reconstruct the historical inputs locally. [CoQA's original terms](https://stanfordnlp.github.io/coqa/#license) include CC BY-SA 4.0, MSR-LA, RACE and CNN/Apache terms. Neither the source-code license nor tokenization replaces those terms. The historical final-question, multi-reference scorer is distinct from the official all-turn benchmark protocol.

For v2, project owners authorize public distribution of owned contributions and
model-derived artifacts subject to the pinned upstream Apache-2.0 terms. The
recalibrated One-shot uses historical fixed500 input records0 and1 for diagnostic
training; they are not held out. Dataset text, answers, tokenized prompts and paired
activation tensors are excluded from all public archives. The dataset is obtained
independently and reconstructed to pinned hashes under its original source terms.
