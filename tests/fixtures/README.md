# R-authored fixtures

These files are produced by a real R session, not by Python. They are the ground
truth for `tests/test_r_export_parity.py`.

| File | What it is |
|---|---|
| `r_exported_pipeline.json` | Output of proximetricsR's `export_sklearn_model()` |
| `r_reference_spectra.csv` | Held-out raw spectra (20 x 256), the `predict()` input |
| `r_reference_predictions.csv` | The predictions **R** produced for those spectra |

## Why they are committed rather than generated

A Python script can only reproduce the dict R *intends* to write. It cannot
reproduce what `jsonlite::toJSON()` actually emits, nor can it detect a
preprocessing step whose chemotools equivalent computes something subtly
different from its proximetricsR original. Both have caused real bugs:
`prep_snv()` serialized as `"params": []` (a JSON array, which the deserializer
rejects), a missing step for the edge points proximetricsR's Savitzky-Golay drops
and chemotools keeps, and a sample-vs-population standard-deviation mismatch in
SNV. Only a file written by R, checked against numbers computed by R, catches
that class of problem.

## Regenerating

Needs R with a proximetricsR build that provides `export_sklearn_model()`
(branch `feat/openmodels-interop`). From the repo root:

```
Rscript scripts/build_r_fixture.R
```

Then re-run `pytest tests/test_r_export_parity.py`. If the predictions changed,
that is a finding, not a fixture to refresh blindly: work out which side moved
before committing the new numbers.

Generated with proximetricsR 0.7.1, R 4.6.1.
