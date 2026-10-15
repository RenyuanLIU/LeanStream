# Replaceable CPU predictor

The contract is hidden/intermediate representation -> predictor backend -> ordered logical group IDs -> `WeightManager::update_prediction`. WeightManager does not depend on this implementation, its quantization, or any learned architecture. Map predictor rows to the application's logical group namespace explicitly when they differ.

`include/leanstream/predictor.h` exposes the reference C ABI: `ls_create`, `ls_predict`, `ls_predict_subset`, and `ls_destroy`. Create one context per concurrent caller. Contexts preallocate scoring/ranking storage; retain model arrays outside the context. Creation configures OpenMP and pins workers to CPUs from the permitted affinity set, so initialize contexts deliberately before other work. It can return null when CPU binding is unavailable. No exception boundary for allocation failure is provided by this reference C ABI.

Rows must be positive, stride a positive multiple of 32, and `1 <= k <= rows`. Threads must fit the available affinity set. Precision is 32, 16, 8, or 4. FP32/FP16 rows contain gate followed by up, each stride elements; input has stride matching-precision values. INT8 weights have the same shape, with two float scales per row and a positive input scale. INT4 weights use stride bytes per row; for each 32-coordinate block, the low/high nibble multiplies the first/second 16 INT8 input coordinates. Gate bytes precede up bytes. Quantized values are signed; scales have shape `[rows,2]`.

Scores are absolute SiLU(gate) times up. IDs sort by descending score, with ascending ID breaking ties. `keep` excludes unique valid rows; `nkeep <= rows-k`. Output has k int32 IDs. Optional gates, ups, and scores arrays each hold rows floats. Scores for skipped excluded rows are not fresh outputs; do not consume them. All pointer capacities and lifetimes are caller responsibilities. Return zero means success; negative values indicate bad arguments or nonfinite scores. Provide finite supported-range quantized inputs to avoid integer overflow.

`ls_predict_subset` is a speculative INT4-only path over supplied unique row IDs, returning `take` IDs. Full prediction remains authoritative for final membership. `ls_row_prefetch` is an optional initialization-time hint; change it only with workers drained.

`predictor_example` uses four synthetic FP32 rows and asserts ranked IDs 3 and 2. It requires neither a checkpoint nor private data. Feed these IDs into a layer's immediate prediction update; finalize with the application's final membership policy independently.
