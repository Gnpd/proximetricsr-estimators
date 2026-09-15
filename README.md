# proximetricsr-estimators

Small, sklearn-conventioned collection of estimator classes used to reconstruct
[proximetricsR](https://github.com/l-ramirez-lopez/proximetricsr) models
(BUCHI NIRWise-PLUS-compatible PLS/XLS chemometric models) as real, predict-ready
Python objects.

It follows the same discovery convention scikit-learn uses, so both can be registered
together with [openmodels](https://github.com/Gnpd/openmodels):

```python
from openmodels import SerializationManager, SklearnSerializer
from chemotools.utils.discovery import all_estimators as chemotools_estimators
from proximetricsr_estimators.utils.discovery import all_estimators as pr_estimators

manager = SerializationManager(
    SklearnSerializer(custom_estimators=[chemotools_estimators, pr_estimators])
)
model = manager.load("exported_from_R.json")
predictions = model.predict(X)
```

## Contents

- `proximetricsr_estimators.regression.NIRWiseLinearModel`: predict-equivalent
  reconstruction of a proximetricsR `spectral_fit` model (PLS or XLS, any
  algorithm variant, including `"nwp"`). Prediction is always affine
  (`(X - x_means_) @ coef_.T + intercept_`), so this single class covers every
  proximetricsR regression method — `fit_method`/`type`/`min_w`/`max_w` are kept
  as constructor params purely for provenance, not used by `predict()`.

- `proximetricsr_estimators.regression.ProximetricsPLS` /
  `.ProximetricsXLS`: full, from-scratch Python reimplementations of
  proximetricsR's PLS/XLS fitting engine (`fit_plsr()`/`fit_xlsr()`'s
  `"standard"`/`"modified"`/`"nwp"` variants — mirroring
  `src/processing_helpers.cpp::estimate_all_pls` exactly), not just
  predict-only reconstruction. Unlike `NIRWiseLinearModel`, these expose the
  real decomposition (`x_weights_`, `x_loadings_`, `x_scores_`,
  `x_rotations_`, `y_loadings_`), which is what's needed for applicability-domain
  diagnostics (e.g. via `chemotools.outliers`) or for supplying a full
  proximetricsR `spectral_fit` shape, not just bare prediction. Verified
  numerically against proximetricsR's own R/C++ engine on synthetic data — max
  absolute differences on the order of `1e-14`–`1e-16` (floating-point noise,
  not approximation) across every fitted attribute, for all three algorithm
  variants of both PLS and XLS. `"standard"` is additionally verified
  equivalent to `sklearn.cross_decomposition.PLSRegression(scale=False)` (and
  therefore to `chemotools.regression.PLSRegression`, a thin subclass of it) —
  prefer that class directly for `"standard"`-only use; these two exist mainly
  for `"modified"`/`"nwp"`, which have no scikit-learn equivalent.

## Installation

```bash
pip install -e .
```

## Development

```bash
pip install -e .
pip install pytest openmodels
pytest
```
