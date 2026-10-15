#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// Linux CPU reference implementation. stride is a multiple of 32; precision: 32/16/8/4.
// Creation pins OpenMP workers to permitted CPUs; one context per concurrent caller.
void* ls_create(int rows,int stride,int k,int threads,int precision);
void ls_destroy(void* context);
int ls_available_cores(void);
int ls_row_prefetch(int distance);
// FP32/FP16: row-major [rows,2,stride]. INT8: same shape, signed bytes.
// INT4: [rows,stride] bytes, signed low/high nibbles for the first/second 16 coordinates of each 32-coordinate block,
// gate then up. Quantized scales: [rows,2]; input_scale must be positive.
// input uses matching precision (INT4 uses INT8 input). Output has k logical IDs.
// keep excludes nkeep IDs. gates/ups/scores are optional rows-length arrays.
// Returns 0 on success, negative on invalid input or nonfinite scores.
int ls_predict(void* context,const void* weights,const float* scales,const void* input,
    float input_scale,const int32_t* keep,int nkeep,int skip,int32_t* output,
    float* gates,float* ups,float* scores);
// Speculative INT4 subset only; ids/count identify rows, output has take IDs.
int ls_predict_subset(void* context,const void* weights,const float* scales,const void* input,
    float input_scale,const int32_t* ids,int count,int take,int32_t* output);
#ifdef __cplusplus
}
#endif
