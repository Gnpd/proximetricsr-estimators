"""Build the "golden fixture" JSON referenced by the proximetricsR <-> openmodels/
chemotools interop plan: the exact wire shape a Pipeline of
RangeCut -> StandardNormalVariate -> SavitzkyGolay -> NIRWiseLinearModel produces
when serialized by openmodels.

This is the ground-truth shape R's `export_sklearn_model()`
(proximetricsR/R/export_sklearn_model.R) must reproduce for the equivalent recipe
(`prep_wav_trim() -> prep_snv() -> prep_derivative(algorithm="savitzky-golay")` +
`fit_plsr()`) -- see tests/testthat/test-export_sklearn_model.R's "produces the
expected Pipeline JSON shape" test in proximetricsR.

Run from the proximetricsr-estimators repo root:
    python scripts/build_golden_fixture.py
"""

import json
from pathlib import Path

import numpy as np
from chemotools.derivative import SavitzkyGolay
from chemotools.feature_selection import RangeCut
from chemotools.scatter import StandardNormalVariate
from chemotools.utils.discovery import all_estimators as chemotools_all_estimators
from openmodels import SerializationManager, SklearnSerializer
from sklearn.pipeline import Pipeline

from proximetricsr_estimators.regression import NIRWiseLinearModel
from proximetricsr_estimators.utils.discovery import all_estimators as pr_all_estimators


def main() -> None:
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
            ("model", NIRWiseLinearModel(fit_method="plsr", type="standard", ncomp=3)),
        ]
    )
    pipeline.fit(X, y)

    manager = SerializationManager(
        SklearnSerializer(
            custom_estimators=[chemotools_all_estimators, pr_all_estimators]
        )
    )
    serialized_dict = manager.model_serializer.serialize(pipeline)

    out_path = Path(__file__).parent.parent / "tests" / "fixtures" / "golden_pipeline_example.json"
    out_path.write_text(json.dumps(serialized_dict, indent=2, sort_keys=True))
    print(f"Wrote {out_path}")

    # Sanity check via the real JSON string round trip (not the raw in-memory dict):
    # param_types/param_dtypes for tuple-valued params (like Pipeline's `steps`,
    # a list of (name, estimator) tuples) are plain Python tuples in the in-memory
    # dict and are only flattened to JSON-compatible lists by json.dumps/json.loads
    # -- convert_from_serializable's list-recursion branch requires that flattening
    # to fire, so a round trip that skips the JSON string (as serialize_dict/
    # deserialize_dict above do) is not representative of the real file-based path.
    serialized_json = manager.serialize(pipeline, format_name="json")
    restored = manager.deserialize(serialized_json, format_name="json")
    np.testing.assert_array_almost_equal(restored.predict(X), pipeline.predict(X), decimal=8)
    print("Round-trip predict() check: OK")


if __name__ == "__main__":
    main()
