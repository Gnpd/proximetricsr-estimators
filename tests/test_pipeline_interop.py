"""End-to-end verification that a Pipeline shaped like proximetricsR's
`export_sklearn_model()` output (chemotools preprocessing steps + a final
NIRWiseLinearModel) round-trips through openmodels and predicts correctly.

This is the Python-side "golden fixture" step of the proximetricsR <-> openmodels/
chemotools interop plan: it proves the JSON shape `export_sklearn_model()` targets
is real and loadable, using the exact same step classes/params the R exporter emits
for the recipe `prep_wav_trim() -> prep_snv() -> prep_derivative(algorithm=
"savitzky-golay")` + `fit_plsr()`.
"""

import numpy as np
import pytest

from proximetricsr_estimators.regression import NIRWiseLinearModel
from proximetricsr_estimators.utils.discovery import all_estimators as pr_estimators


def _chemotools_all_estimators():
    chemotools = pytest.importorskip("chemotools")  # noqa: F841
    from chemotools.utils.discovery import all_estimators

    return all_estimators


def test_chemotools_steps_used_by_export_sklearn_model_are_discoverable():
    """Every chemotools class R/export_sklearn_model.R's translator emits must
    actually exist and be discoverable via chemotools's own all_estimators()."""
    all_estimators = _chemotools_all_estimators()
    estimators = dict(all_estimators())

    expected = [
        "RangeCut",
        "StandardNormalVariate",
        "SavitzkyGolay",
        "NorrisWilliams",
        "SavitzkyGolayFilter",
        "MeanFilter",
        "PolynomialCorrection",
        "IntensityConversion",
    ]
    missing = [name for name in expected if name not in estimators]
    assert not missing, f"chemotools classes referenced by export_sklearn_model() not found: {missing}"


def test_exported_pipeline_shape_round_trips_and_predicts():
    """Builds the Pipeline for the recipe used in
    tests/testthat/test-export_sklearn_model.R's "produces the expected Pipeline
    JSON shape" test: prep_wav_trim -> prep_snv -> prep_derivative(savitzky-golay)
    -> fit_plsr(). Serializes it via openmodels with chemotools + this package
    registered, deserializes, and checks predict() runs and matches the original
    fitted pipeline bit-for-bit (it's the same Python objects at that point, so
    this mainly proves the openmodels round trip itself doesn't perturb anything).
    """
    openmodels = pytest.importorskip("openmodels")
    chemotools_all_estimators = _chemotools_all_estimators()
    from chemotools.feature_selection import RangeCut
    from chemotools.scatter import StandardNormalVariate
    from chemotools.derivative import SavitzkyGolay
    from sklearn.pipeline import Pipeline

    rng = np.random.default_rng(0)
    n_samples, n_features = 20, 40
    wavelengths = np.linspace(1100, 1600, n_features)
    X = rng.normal(loc=1.0, scale=0.1, size=(n_samples, n_features))
    y = rng.normal(size=n_samples)

    pipeline = Pipeline(
        [
            ("step1_wav_trim", RangeCut(start=1150, end=1550, x_axis=wavelengths)),
            ("step2_snv", StandardNormalVariate()),
            ("step3_derivative", SavitzkyGolay(window_length=5, polyorder=2, deriv=1)),
            (
                "model",
                NIRWiseLinearModel(fit_method="plsr", type="standard", ncomp=3),
            ),
        ]
    )
    pipeline.fit(X, y)

    manager = openmodels.SerializationManager(
        openmodels.SklearnSerializer(
            custom_estimators=[chemotools_all_estimators, pr_estimators]
        )
    )
    # Serialize via the underlying model serializer to inspect the raw dict shape
    # (the same shape R's export_sklearn_model() hand-authors, before JSON encoding).
    serialized_dict = manager.model_serializer.serialize(pipeline)
    assert serialized_dict["estimator_class"] == "Pipeline"

    # Also exercise the full string round trip (JSON), matching how a file
    # exported from R would actually be consumed.
    serialized_json = manager.serialize(pipeline, format_name="json")
    restored = manager.deserialize(serialized_json, format_name="json")
    original_predictions = pipeline.predict(X)
    restored_predictions = restored.predict(X)

    np.testing.assert_array_almost_equal(
        restored_predictions, original_predictions, decimal=8
    )
