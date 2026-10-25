# Generation and performance media

The current README displays [genuine autoregressive Library generation](autoregressive/README.md),
using the v2 default recalibrated One-shot. It is not teacher-forced. Exact tokens,
timing, model/predictor identities, EOS behavior and rendering commands are documented there.

The older [fixed-token input](fixed_decode_input.json), [measured fixed-token trace](fixed_token_decode_trace.json)
and speed_comparison.gif remain historical performance diagnostics. They process
identical reference continuation inputs and do not measure independent generated
answers. They are no longer the README demonstration. The capture_fixed_tokens
command and make_speed_gif.py renderer remain available for that historical mode.

[Original free-generation diagnostics](diagnostics/README.md) retain failures of
the old512-feature One-shot. They do not describe the v2 default1638-feature predictor.
Historical pooled numbers and single-prompt GIF timings are distinct measurements.
