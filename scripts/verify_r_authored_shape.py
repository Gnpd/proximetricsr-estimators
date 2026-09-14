"""Simulates the exact wire dict R/export_sklearn_model.R hand-authors for every
supported preprocessing step (as it would look after `toJSON()` -> file -> Python
`json.loads()`), to verify it deserializes and predicts correctly via openmodels +
chemotools + this package -- without needing an R interpreter.

Mirrors export_sklearn_model.R's `.translate_prep_step()` step-by-step, including
its fitted-attribute computations (verified against the actual chemotools source
for each supported class: StandardNormalVariate, SavitzkyGolay (derivative),
SavitzkyGolayFilter (smooth), MeanFilter, PolynomialCorrection, IntensityConversion,
RangeCut). `prep_derivative(algorithm="gap-segment")` (NorrisWilliams) is
intentionally excluded -- export_sklearn_model.R does not support it, since
NorrisWilliams precomputes an internal kernel_ that hasn't been verified.

SavitzkyGolayFilter's kernel_ here is computed via a Python/numpy transliteration
of proximetricsR's existing sgf() helper (R/proxiscout_write_model.R) -- cross-checked
separately against scipy.signal.savgol_coeffs(..., use="conv") for several
(window, polyorder) pairs; for deriv=0 the coefficient vector is symmetric, so
there is no conv-vs-dot ordering ambiguity for this mapping. This script cannot
invoke the real R sgf() directly, so the R-side implementation itself (not just
this reference formula) should still be spot-checked against a live R session
before relying on this mapping.
"""

import json

import numpy as np
from chemotools.utils.discovery import all_estimators as chemotools_all_estimators
from openmodels import SerializationManager, SklearnSerializer

from proximetricsr_estimators.utils.discovery import all_estimators as pr_all_estimators


def _find_index(target, axis):
    return int(np.argmin(np.abs(np.asarray(axis) - target)))


def range_cut_step(name, x_axis_in, start, end):
    start_index = _find_index(start, x_axis_in)
    end_index = _find_index(end, x_axis_in)
    selected = x_axis_in[start_index:end_index]
    return [
        name,
        {
            "estimator_class": "RangeCut",
            "params": {"start": start, "end": end, "x_axis": list(x_axis_in)},
            "param_types": {"x_axis": "ndarray"},
            "param_dtypes": {"x_axis": "float64"},
            "attributes": {
                "start_index_": start_index,
                "end_index_": end_index,
                "x_axis_": list(selected),
                "wavenumbers_": list(selected),
                "n_features_in_": len(x_axis_in),
            },
            "attribute_types": {
                "start_index_": "int",
                "end_index_": "int",
                "x_axis_": "ndarray",
                "wavenumbers_": "ndarray",
                "n_features_in_": "int",
            },
            "attribute_dtypes": {"x_axis_": "float64", "wavenumbers_": "float64"},
        },
    ], len(selected)


def snv_step(name, n_features_in):
    return [
        name,
        {
            "estimator_class": "StandardNormalVariate",
            "params": {},
            "attributes": {"n_features_in_": n_features_in},
            "attribute_types": {"n_features_in_": "int"},
        },
    ]


def savitzky_golay_derivative_step(name, n_features_in, w, p, m):
    return [
        name,
        {
            "estimator_class": "SavitzkyGolay",
            "params": {"window_length": w, "polyorder": p, "deriv": m},
            "attributes": {
                "window_length_": w,
                "polyorder_": p,
                "deriv_": m,
                "n_features_in_": n_features_in,
            },
            "attribute_types": {
                "window_length_": "int",
                "polyorder_": "int",
                "deriv_": "int",
                "n_features_in_": "int",
            },
        },
    ]


def sgf_kernel(p, n, m=0):
    """Python/numpy transliteration of proximetricsR's sgf(p, n, m) R helper
    (R/proxiscout_write_model.R), used to derive SavitzkyGolayFilter's kernel_."""
    k = n // 2 + 1  # 1-based center index, matching R's floor(n/2)+1
    offsets = np.arange(1, n + 1) - k
    powers = np.arange(0, p + 1)
    Ce = offsets[:, None].astype(float) ** powers[None, :]
    A = np.linalg.pinv(Ce)
    Fm = A[m, :].copy()
    if m > 0:
        import math

        Fm = Fm * math.factorial(m)
    return Fm


def savitzky_golay_filter_step(name, n_features_in, w, p):
    kernel = sgf_kernel(p=p, n=w, m=0)[::-1].copy()  # rev(), a no-op for m=0
    return [
        name,
        {
            "estimator_class": "SavitzkyGolayFilter",
            "params": {"window_length": w, "polyorder": p},
            "attributes": {
                "window_length_": w,
                "polyorder_": p,
                "kernel_": list(kernel),
                # _half_ = (window_length_ - 1) // 2 is a *private* attribute
                # chemotools._BaseFIRFilter.transform() reads directly (see
                # chemotools/smooth/_base.py::_apply_filter_1d). openmodels' own
                # attribute *extraction* skips leading-underscore attributes
                # (SklearnSerializer._extract_estimator_attributes), so a plain
                # Python-native round trip of a real SavitzkyGolayFilter would
                # actually drop this too -- it only works here because we set it
                # directly in the hand-authored dict, bypassing that extraction step.
                "_half_": (w - 1) // 2,
                "n_features_in_": n_features_in,
            },
            "attribute_types": {
                "window_length_": "int",
                "polyorder_": "int",
                "kernel_": "ndarray",
                "_half_": "int",
                "n_features_in_": "int",
            },
            "attribute_dtypes": {"kernel_": "float64"},
        },
    ]


def mean_filter_step(name, n_features_in, w):
    return [
        name,
        {
            "estimator_class": "MeanFilter",
            "params": {"window_length": w},
            "attributes": {"window_length_": w, "n_features_in_": n_features_in},
            "attribute_types": {"window_length_": "int", "n_features_in_": "int"},
        },
    ]


def polynomial_correction_step(name, n_features_in, order):
    return [
        name,
        {
            "estimator_class": "PolynomialCorrection",
            "params": {"order": order, "indices": None},
            "attributes": {
                "indices_": list(range(n_features_in)),
                "n_features_in_": n_features_in,
            },
            "attribute_types": {"indices_": "ndarray", "n_features_in_": "int"},
            "attribute_dtypes": {"indices_": "int32"},
        },
    ]


def intensity_conversion_step(name, n_features_in):
    return [
        name,
        {
            "estimator_class": "IntensityConversion",
            "params": {"input_unit": "reflectance", "output_unit": "pseudoabsorbance"},
            "attributes": {"n_features_in_": n_features_in},
            "attribute_types": {"n_features_in_": "int"},
        },
    ]


def model_step(x_means, coef, intercept, feature_names):
    return [
        "model",
        {
            "estimator_class": "NIRWiseLinearModel",
            "params": {
                "fit_method": "plsr",
                "type": "standard",
                "ncomp": 5,
                "min_w": None,
                "max_w": None,
            },
            "attributes": {
                "x_means_": list(x_means),
                "coef_": list(coef),
                "intercept_": intercept,
                "n_features_in_": len(x_means),
                "feature_names_in_": list(feature_names),
            },
            "attribute_types": {
                "x_means_": "ndarray",
                "coef_": "ndarray",
                "intercept_": "float",
                "n_features_in_": "int",
                "feature_names_in_": "ndarray",
            },
            "attribute_dtypes": {
                "x_means_": "float64",
                "coef_": "float64",
                "feature_names_in_": "object",
            },
        },
    ]


def build_doc(steps):
    step_types = [["str", s[1]["estimator_class"]] for s in steps]
    return {
        "estimator_class": "Pipeline",
        "params": {"steps": steps, "memory": None, "verbose": False},
        "param_types": {"steps": step_types, "memory": "NoneType", "verbose": "bool"},
        "metadata": {
            "producer_name": "sklearn",
            "producers": {"sklearn": "unknown", "chemotools": "unknown"},
            "domain": "sklearn",
            "openmodels_format_version": 2,
            "openmodels_version": "unknown",
            "created_at": "2026-01-01T00:00:00Z",
            "dependency_versions": {"numpy": "unknown", "scipy": "unknown"},
            "source": "proximetricsR",
        },
    }


def deserialize_and_predict(doc, X):
    manager = SerializationManager(
        SklearnSerializer(custom_estimators=[chemotools_all_estimators, pr_all_estimators])
    )
    round_tripped = json.loads(json.dumps(doc))
    restored = manager.model_serializer.deserialize(round_tripped)
    return restored, restored.predict(X)


def main():
    rng = np.random.default_rng(0)
    n0 = 60
    wavelengths0 = np.linspace(1000, 1700, n0)

    # --- Case A: RangeCut -> SNV -> SavitzkyGolay(derivative) -> model -----------
    step1, n1 = range_cut_step("step1_wav_trim", wavelengths0, 1100, 1600)
    axis1 = wavelengths0[
        _find_index(1100, wavelengths0) : _find_index(1600, wavelengths0)
    ]
    step2 = snv_step("step2_snv", n1)
    step3 = savitzky_golay_derivative_step("step3_derivative", n1, w=5, p=2, m=1)
    x_means = rng.normal(size=n1)
    coef = rng.normal(size=n1) * 0.01
    steps_a = [step1, step2, step3, model_step(x_means, coef, 5.0, axis1.astype(str))]
    X = rng.normal(loc=1.0, scale=0.1, size=(6, n0))
    restored, preds = deserialize_and_predict(build_doc(steps_a), X)
    assert preds.shape == (6,)
    print("Case A (RangeCut -> SNV -> SavitzkyGolay -> model): OK, predictions:", preds)

    # --- Case B: SavitzkyGolayFilter (savitzky-golay smooth) -----------------------
    step_sgf = savitzky_golay_filter_step("step1_smooth", n0, w=7, p=3)
    steps_b = [step_sgf, model_step(rng.normal(size=n0), rng.normal(size=n0) * 0.01, 1.0, wavelengths0.astype(str))]
    restored, preds = deserialize_and_predict(build_doc(steps_b), X)
    assert preds.shape == (6,)
    print("Case B (SavitzkyGolayFilter -> model): OK")

    # Extra check: the reconstructed SavitzkyGolayFilter step's transform() output
    # matches scipy.signal.savgol_filter directly (deriv=0), confirming kernel_'s
    # *values* are correct, not just that the Pipeline runs without error.
    from scipy.signal import savgol_filter

    sgf_transformer = restored.named_steps["step1_smooth"]
    manual = savgol_filter(X, window_length=7, polyorder=3, deriv=0, axis=1, mode="nearest")
    reconstructed = sgf_transformer.transform(X)
    np.testing.assert_array_almost_equal(reconstructed, manual, decimal=6)
    print("  kernel_ values verified against scipy.signal.savgol_filter directly: OK")

    # --- Case C: MeanFilter (moving-average smooth) --------------------------------
    step_mf = mean_filter_step("step1_smooth", n0, w=5)
    steps_c = [step_mf, model_step(rng.normal(size=n0), rng.normal(size=n0) * 0.01, 1.0, wavelengths0.astype(str))]
    restored, preds = deserialize_and_predict(build_doc(steps_c), X)
    assert preds.shape == (6,)
    print("Case C (MeanFilter -> model): OK")

    # --- Case D: PolynomialCorrection (detrend) -------------------------------------
    step_pc = polynomial_correction_step("step1_detrend", n0, order=2)
    steps_d = [step_pc, model_step(rng.normal(size=n0), rng.normal(size=n0) * 0.01, 1.0, wavelengths0.astype(str))]
    restored, preds = deserialize_and_predict(build_doc(steps_d), X)
    assert preds.shape == (6,)
    print("Case D (PolynomialCorrection -> model): OK")

    # --- Case E: IntensityConversion (transform to absorbance/pseudoabsorbance) ----
    step_ic = intensity_conversion_step("step1_transform", n0)
    steps_e = [step_ic, model_step(rng.normal(size=n0), rng.normal(size=n0) * 0.01, 1.0, wavelengths0.astype(str))]
    X_pos = np.abs(X) + 0.1  # reflectance must be > 0
    restored, preds = deserialize_and_predict(build_doc(steps_e), X_pos)
    assert preds.shape == (6,)
    print("Case E (IntensityConversion -> model): OK")

    print("\nAll verified mappings deserialize and predict without error.")


if __name__ == "__main__":
    main()
