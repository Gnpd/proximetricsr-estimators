"""Tests for proximetricsr_estimators.regression.NIRWiseLinearModel."""

import numpy as np
import pytest
from sklearn.utils.estimator_checks import check_estimator

from proximetricsr_estimators.regression import NIRWiseLinearModel
from proximetricsr_estimators.utils.discovery import all_estimators


def test_compliance_nirwise_linear_model():
    check_estimator(NIRWiseLinearModel())


def test_predict_matches_hand_computed_affine_result():
    # Arrange: attributes set directly, as openmodels' deserializer would do for
    # a model reconstructed from an R-exported file (not via fit()).
    model = NIRWiseLinearModel(fit_method="xlsr", type="standard", ncomp=2, min_w=3, max_w=15)
    model.x_means_ = np.array([1.0, 2.0, 3.0])
    model.coef_ = np.array([0.5, -1.0, 2.0])
    model.intercept_ = 10.0
    model.n_features_in_ = 3

    X = np.array(
        [
            [1.0, 2.0, 3.0],  # centered -> [0, 0, 0] -> pred = intercept
            [2.0, 2.0, 3.0],  # centered -> [1, 0, 0] -> pred = intercept + 0.5
            [1.0, 3.0, 4.0],  # centered -> [0, 1, 1] -> pred = intercept - 1.0 + 2.0
        ]
    )

    # Act
    predictions = model.predict(X)

    # Assert
    expected = np.array([10.0, 10.5, 11.0])
    np.testing.assert_array_almost_equal(predictions, expected, decimal=10)


def test_predict_before_fit_raises():
    model = NIRWiseLinearModel()
    with pytest.raises(Exception):
        model.predict(np.ones((2, 3)))


def test_fit_is_plain_ols_after_centering():
    # Arrange
    rng = np.random.default_rng(42)
    X = rng.normal(size=(50, 4))
    true_coef = np.array([1.0, -2.0, 0.5, 0.0])
    y = 3.0 + X @ true_coef

    model = NIRWiseLinearModel()

    # Act
    model.fit(X, y)
    predictions = model.predict(X)

    # Assert: exact recovery for a noiseless linear relationship. intercept_ is the
    # sample mean of y (matching proximetricsR's intercept = mean(Y) convention),
    # not necessarily the "true" 3.0 used to generate y from finite-sample X.
    np.testing.assert_array_almost_equal(predictions, y, decimal=8)
    np.testing.assert_array_almost_equal(model.coef_, true_coef, decimal=8)
    assert model.intercept_ == pytest.approx(y.mean(), abs=1e-8)


def test_provenance_params_do_not_affect_predict():
    X = np.array([[1.0, 1.0]])

    plsr_model = NIRWiseLinearModel(fit_method="plsr", type="standard", ncomp=5)
    xlsr_model = NIRWiseLinearModel(fit_method="xlsr", type="modified", ncomp=1, min_w=3, max_w=15)
    for model in (plsr_model, xlsr_model):
        model.x_means_ = np.array([0.0, 0.0])
        model.coef_ = np.array([1.0, 1.0])
        model.intercept_ = 0.0
        model.n_features_in_ = 2

    np.testing.assert_array_almost_equal(plsr_model.predict(X), xlsr_model.predict(X))


def test_all_estimators_discovers_nirwise_linear_model():
    estimators = dict(all_estimators())
    assert "NIRWiseLinearModel" in estimators
    assert estimators["NIRWiseLinearModel"] is NIRWiseLinearModel


def test_openmodels_round_trip():
    openmodels = pytest.importorskip("openmodels")

    model = NIRWiseLinearModel(fit_method="xlsr", type="standard", ncomp=2, min_w=3, max_w=15)
    model.x_means_ = np.array([1.0, 2.0, 3.0])
    model.coef_ = np.array([0.5, -1.0, 2.0])
    model.intercept_ = 10.0
    model.n_features_in_ = 3
    model.feature_names_in_ = np.array(["1000", "1002", "1004"], dtype=object)

    manager = openmodels.SerializationManager(
        openmodels.SklearnSerializer(custom_estimators=all_estimators)
    )
    serialized = manager.serialize(model)
    restored = manager.deserialize(serialized)

    X = np.array([[1.0, 2.0, 3.0], [2.0, 2.0, 3.0]])
    np.testing.assert_array_almost_equal(restored.predict(X), model.predict(X), decimal=10)
    assert restored.fit_method == "xlsr"
    assert restored.type == "standard"
    assert restored.ncomp == 2
    assert restored.min_w == 3
    assert restored.max_w == 15
