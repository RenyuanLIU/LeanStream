# Experimental prediction schedules

The optional schedule API is independent of the three primary runtime components. `PredictionSchedule` names enabled prediction points and per-point I/O budgets. `compile_prediction_schedule` validates the points and head boundaries, returning an `AttentionExecutionPlan`.

`AttnInput + AttnOutput` uses the ordinary unsplit attention path. Optional `AttnQuarter`, `AttnHalf`, and `AttnThreeQuarter` checkpoints may split attention into head segments and add overhead. Unsupported boundaries return an error instead of silently rounding. The default schedule keeps the unsplit behavior.

`LatestAvailable` is owned by one control worker. Mailbox ready epochs are GPU-published and acquire-observed; data and flags must remain valid through retirement. Intermediate checkpoints may be skipped when newer data is ready. `finish` tells the caller whether a completed result is still current enough to publish as a priority update. Final output is mandatory. Checkpoint completion never launches GPU work.

This release includes schedule compilation and checkpoint-selection helpers, not the model-specific attention kernels or production orchestration. Integrations implement the plan while preserving default attention semantics. `prediction_schedule_example` and the schedule tests demonstrate unsplit and segmented plans, invalid boundaries, latest-available selection, and final completion.
