# VAE Gene Expression Imputation

Single-file PyTorch research script (`vae-test.py`). No tests, no package structure. Modules not installable in this environment, so tests beyond syntax checking should not be run. Source control is also managed manually by user.

## Known Issues
(none currently open -- the previous item, "use _random_fill wherever we
initialize vamp pseudo-inputs to replace type-1 missing values (NaNs) with
random values from the same distribution," was addressed: tuning.py's
`run_trial`/`retrain` and sample_efficiency.py's `train_and_eval_one` now
initialize VampPriorVAE/DEQEncoderVampVAE pseudo-inputs from real data with
`_random_fill` applied to type-1 NaNs.)

