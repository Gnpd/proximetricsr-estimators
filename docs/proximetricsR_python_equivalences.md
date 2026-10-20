# proximetricsR ↔ chemotools / scikit-learn / proximetricsr-estimators equivalence

How each proximetricsR recipe step and model maps to Python, and whether
`export_sklearn_model()` can export it. Checked against `R/` in proximetricsR,
chemotools 0.4.4 and proximetricsr-estimators 0.1.0.

## Preprocessing steps (`prep_*`)

| proximetricsR | Python equivalent (package) | Parameter mapping | Exported by `export_sklearn_model()`? |
|---|---|---|---|
| `prep_snv()` | `chemotools.scatter.StandardNormalVariate` | none | ✅ R divides by `n−1`, numpy by `n`. The constant factor is folded into the model's `coef_`/`x_means_`. |
| `prep_derivative(m, w, p, algorithm = "savitzky-golay")` | `chemotools.derivative.SavitzkyGolay` | `window_length = w`, `polyorder = p`, `deriv = m`, `mode = "nearest"` | ✅ A `RangeCut` is added after it, because R drops `(w−1)/2` points on each edge. |
| `prep_derivative(..., algorithm = "gap-segment")` | `chemotools.derivative.NorrisWilliams` | probably `gap_size = w`, `window_length = p` (segment), `deriv = m` ⚠️ not verified | ❌ `kernel_` isn't reproduced yet |
| `prep_derivative(..., algorithm = "nwp")` | none (BUCHI NIRWise PLUS "DG" derivative) | — | ❌ |
| `prep_smooth(w, p, algorithm = "savitzky-golay")` | `chemotools.smooth.SavitzkyGolayFilter` | `window_length = w`, `polyorder = p`; `kernel_` is computed in R with `sgf()` | ✅ A `RangeCut` is added after it (edges dropped in R). |
| `prep_smooth(w, algorithm = "moving-average")` | `chemotools.smooth.MeanFilter` | `window_length = w` (default `mode = "nearest"`) | ✅ ⚠️ Possible problem: moving-average edges  |
| `prep_detrend(p)` | `chemotools.baseline.PolynomialCorrection` | `order = p`, `indices = None` (whole spectrum) | ✅ |
| `prep_transform(to = "absorbance")` | `chemotools.physics.IntensityConversion` | `input_unit = "reflectance"`, `output_unit = "pseudoabsorbance"` (A = −log10 R) | ✅ Can't come after `prep_snv()`. |
| `prep_transform(to = "reflectance")` | `chemotools.physics.IntensityConversion` | `input_unit = "pseudoabsorbance"`, `output_unit = "reflectance"` | ✅ Can't come after `prep_snv()`. |
| `prep_wav_trim(band)` | `chemotools.feature_selection.RangeCut` | `start = min(band)`, `end = max(band)`, `x_axis = <input grid>`; indices are taken from R's recorded output grid | ✅ |
| `prep_wav_trim(band, trim_constant_edges = TRUE)` | none (depends on the data) | — | ❌ |
| `prep_resample(grid)` | `chemotools.adaptation.XAxisInterpolator` | `common_x_axis = seq(min, max, by = res)` or the NeoSpectra grid; method: natural spline in R vs `"cubic"` default ⚠️ | ❌ needs the x-axis passed at every `predict()` |

## Models (`fit_*`)

| proximetricsR | Fitting in Python | Export / prediction in Python | Parameter mapping |
|---|---|---|---|
| `fit_plsr(ncomp, type = "standard")` | `proximetricsr_estimators.regression.ProximetricsPLS(type="standard")`; same results as `sklearn.cross_decomposition.PLSRegression(scale=False)` / `chemotools.regression.PLSRegression` | `proximetricsr_estimators.regression.NIRWiseLinearModel` ✅ | `n_components = ncomp` (the export uses `ncomp = final_ncomp`) |
| `fit_plsr(ncomp, type = "modified")` | `ProximetricsPLS(type="modified")` (no scikit-learn equivalent) | `NIRWiseLinearModel` ✅ | same |
| `fit_plsr(ncomp, type = "nwp")` | `ProximetricsPLS(type="nwp")` (same predictions as `"modified"`) | `NIRWiseLinearModel` ✅ | same |
| `fit_xlsr(ncomp, type, min_w, max_w)` (all three types) | `proximetricsr_estimators.regression.ProximetricsXLS` (no scikit-learn equivalent) | `NIRWiseLinearModel` ✅ | `n_components = ncomp`, `type`, `min_w`, `max_w` |

**What `NIRWiseLinearModel` holds:** `fit_method` (`"plsr"`/`"xlsr"`), `type`,
`ncomp`, `min_w`, `max_w`. These are recorded for provenance only. Prediction
uses the fitted attributes `x_means_`, `coef_` and `intercept_` in
`(X − x_means_) · coef_ + intercept_`. It can't be refitted in Python; use
`ProximetricsPLS`/`ProximetricsXLS` for that.

## Added by the exporter (no proximetricsR step behind them)

| Python | When |
|---|---|
| `sklearn.pipeline.Pipeline` | Always the outer container: steps are named `stepN_<method>`, then `model`. |
| `chemotools.feature_selection.RangeCut` (`stepN_<method>_trim`) | After a Savitzky-Golay derivative or smoothing step that made the spectrum shorter in R. |