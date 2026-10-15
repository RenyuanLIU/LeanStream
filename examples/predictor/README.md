# CPU predictor example

Run `./build/predictor_example` after the CPU build. Four synthetic FP32 gate/up rows and a 32-element input produce ranked logical IDs `3 2`. The program creates the reference predictor, predicts, verifies the result, and destroys the context. No model checkpoint is used. Creation pins the calling OpenMP worker; see [predictor contract](../../docs/predictor.md).
