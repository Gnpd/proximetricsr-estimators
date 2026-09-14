# proximetricsr-estimators

Small, sklearn-conventioned collection of estimator classes used to reconstruct
[proximetricsR](https://github.com/l-ramirez-lopez/proximetricsr) models
(BUCHI NIRWise-PLUS-compatible PLS/XLS chemometric models) as real, predict-ready
Python objects.

This package exists so that proximetricsR-specific reconstruction logic doesn't
have to live inside [chemotools](https://github.com/paucablop/chemotools) itself.
It follows the same discovery convention chemotools uses, so both can be registered
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
  non-`"nwp"` algorithm variant). Prediction is always affine
  (`(X - x_means_) @ coef_.T + intercept_`), so this single class covers every
  proximetricsR regression method — `fit_method`/`type`/`min_w`/`max_w` are kept
  as constructor params purely for provenance, not used by `predict()`.

## Installation

```bash
pip install -e .
```

## Development

```bash
pip install -e ".[dev]"
pytest
```
