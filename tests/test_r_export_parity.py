"""Parity against a real proximetricsR export.

The fixtures under ``tests/fixtures/`` are written by an actual R session (see
``scripts/build_r_fixture.R`` and ``tests/fixtures/README.md``), not simulated
here in Python. That distinction is the point of this module: a hand-authored
Python dict cannot reproduce what R's ``jsonlite::toJSON()`` actually emits, and
shape assertions alone cannot detect a preprocessing step whose Python
equivalent computes something subtly different from its R original. Both classes
of bug have shipped before. ``test_predictions_match_r`` is the backstop for
them: it loads the R file and checks the numbers.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

FIXTURES = Path(__file__).parent / "fixtures"
PIPELINE_JSON = FIXTURES / "r_exported_pipeline.json"


def _manager():
    openmodels = pytest.importorskip("openmodels")
    pytest.importorskip("chemotools")
    from chemotools.utils.discovery import all_estimators as chemotools_estimators

    from proximetricsr_estimators.utils.discovery import all_estimators as pr_estimators

    return openmodels.SerializationManager(
        openmodels.SklearnSerializer(
            custom_estimators=[chemotools_estimators, pr_estimators]
        )
    )


@pytest.fixture(scope="module")
def pipeline():
    return _manager().load(str(PIPELINE_JSON))


@pytest.fixture(scope="module")
def spectra():
    return np.loadtxt(FIXTURES / "r_reference_spectra.csv", delimiter=",", skiprows=1)


@pytest.fixture(scope="module")
def r_predictions():
    return np.loadtxt(
        FIXTURES / "r_reference_predictions.csv", delimiter=",", skiprows=1
    )


def test_predictions_match_r(pipeline, spectra, r_predictions):
    """The whole point of the export: Python must reproduce R's numbers."""
    predictions = pipeline.predict(spectra)

    assert predictions.shape == r_predictions.shape
    np.testing.assert_allclose(predictions, r_predictions, atol=1e-10)


def test_exported_steps_are_the_expected_classes(pipeline):
    """The fixture's recipe is prep_wav_trim -> prep_transform ->
    prep_derivative(savitzky-golay) -> prep_snv -> fit_plsr. The extra RangeCut
    after the derivative reproduces the edge points proximetricsR drops and
    chemotools, which pads instead, keeps."""
    assert [type(est).__name__ for _, est in pipeline.steps] == [
        "RangeCut",
        "IntensityConversion",
        "SavitzkyGolay",
        "RangeCut",
        "StandardNormalVariate",
        "NIRWiseLinearModel",
    ]


def test_wire_shape_matches_openmodels_format():
    """https://github.com/Gnpd/openmodels/blob/main/docs/format.md"""
    doc = json.loads(PIPELINE_JSON.read_text())

    assert doc["estimator_class"] == "Pipeline"
    assert "metadata" in doc

    for name, step in doc["params"]["steps"]:
        # metadata is root-only, never duplicated on nested sub-estimators
        assert "metadata" not in step, name
        # every param carries a type, so values JSON cannot represent natively
        # survive the round trip
        assert set(step.get("param_types", {})) == set(step.get("params", {})), name
        # openmodels indexes data["attribute_types"] directly rather than via
        # .get() once "attributes" is present, so a missing one is a KeyError
        if "attributes" in step:
            assert "attribute_types" in step, name


def test_metadata_records_its_python_producers():
    doc = json.loads(PIPELINE_JSON.read_text())
    metadata = doc["metadata"]

    # producer_name is the package owning the *outermost* class (Pipeline), not
    # the tool that wrote the file -- that is recorded in "source".
    assert metadata["producer_name"] == "sklearn"
    assert metadata["source"] == "proximetricsR"
    assert set(metadata["producers"]) == {
        "sklearn",
        "chemotools",
        "proximetricsr_estimators",
    }
    assert metadata["openmodels_format_version"] == 2
    # absent on purpose: R cannot know which scikit-learn will load the file, and
    # openmodels skips its version check when the field is missing rather than
    # warning on a placeholder.
    assert "producer_version" not in metadata


def test_predict_emits_no_warnings(pipeline, spectra):
    """Pins the wavenumbers_/feature_names_in_ decision: the model step is handed
    a plain array by the step before it, so feature_names_in_ would warn here on
    every call."""
    model = pipeline.steps[-1][1]
    assert not hasattr(model, "feature_names_in_")
    assert model.wavenumbers_.shape == (model.n_features_in_,)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipeline.predict(spectra)

    assert [str(w.message) for w in caught] == []


def test_predict_works_with_pandas_output(pipeline, spectra, r_predictions):
    """Under set_output("pandas") the transformers relabel columns "x0", "x1", ...
    With feature_names_in_ set on the model those would mismatch and raise."""
    pytest.importorskip("pandas")

    pipeline.set_output(transform="pandas")
    try:
        np.testing.assert_allclose(pipeline.predict(spectra), r_predictions, atol=1e-10)
    finally:
        pipeline.set_output(transform="default")
