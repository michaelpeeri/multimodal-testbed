# VAE Gene Expression Imputation

Single-file PyTorch research script (`vae-test.py`). No tests, no package structure. Modules not installable in this environment, so tests beyond syntax checking should not be run. Source control is also managed manually by user.

## Known Issues
- Use _random_fill wherever we initizialize vamp pseudo-inputs to replace type-1 missing values (NaNs) with random values from the same distribution.

