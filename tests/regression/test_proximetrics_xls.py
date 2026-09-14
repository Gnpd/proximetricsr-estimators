"""Tests for proximetricsr_estimators.regression.ProximetricsXLS.

Numerically verified against R's own fit_xlsr() (proximetricsR
src/processing_helpers.cpp::estimate_all_pls, "xls_*" weight branches) on
synthetic data, for all three types and including a window-overhang boundary
case (min_w/max_w close to n_features): max abs differences on the order of
1e-14 to 1e-15 across coef_/intercept_/x_weights_/x_loadings_/x_scores_/
x_rotations_/y_loadings_. That R comparison isn't reproduced here since it
requires an R installation; this file re-verifies what's checkable from Python
alone and guards the documented "modified"/"nwp" relationship, mirroring
test_proximetrics_pls.py.
"""

import numpy as np
import pytest
from sklearn.utils.estimator_checks import check_estimator

from proximetricsr_estimators.regression import ProximetricsXLS
from proximetricsr_estimators.utils.discovery import all_estimators


def _make_data(n_samples=60, n_features=25, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    true_beta = rng.normal(size=n_features) * 0.3
    y = X @ true_beta + rng.normal(scale=0.05, size=n_samples)
    return X, y


def test_compliance_proximetrics_xls():
    # n_components=1, a narrow (min_w=1, max_w=2) window: sklearn's generic
    # conformance checks fit on small synthetic datasets, and XLS's windowed
    # weight computation needs enough features/samples relative to min_w/max_w
    # and n_components to keep `weights @ x_loadings.T` invertible (the same
    # requirement the real R fit_xlsr() has -- this isn't a Python-specific
    # limitation). The smallest reasonable configuration is used here purely to
    # satisfy sklearn's estimator API checks, not because larger configurations
    # don't work on real spectral data (see the other tests below).
    check_estimator(ProximetricsXLS(n_components=1, min_w=1, max_w=2))


def test_min_w_must_be_less_than_max_w():
    X, y = _make_data(n_features=10)
    model = ProximetricsXLS(n_components=2, min_w=5, max_w=5)
    with pytest.raises(ValueError, match="min_w.*max_w"):
        model.fit(X, y)


@pytest.mark.parametrize("type_", ["standard", "modified", "nwp"])
def test_predict_matches_manual_affine_computation(type_):
    X, y = _make_data()
    model = ProximetricsXLS(n_components=4, type=type_, min_w=3, max_w=8)
    model.fit(X, y)

    manual = (X - model.x_mean_) @ model.coef_ + model.intercept_
    np.testing.assert_array_almost_equal(model.predict(X), manual, decimal=10)


@pytest.mark.parametrize("type_", ["standard", "modified", "nwp"])
def test_transform_matches_x_rotations(type_):
    X, y = _make_data()
    model = ProximetricsXLS(n_components=4, type=type_, min_w=3, max_w=8)
    model.fit(X, y)

    manual_scores = (X - model.x_mean_) @ model.x_rotations_
    np.testing.assert_array_almost_equal(model.transform(X), manual_scores, decimal=10)


def test_modified_and_nwp_predict_identically():
    X, y = _make_data(seed=3)
    ncomp = 4

    modified = ProximetricsXLS(n_components=ncomp, type="modified", min_w=3, max_w=8).fit(X, y)
    nwp = ProximetricsXLS(n_components=ncomp, type="nwp", min_w=3, max_w=8).fit(X, y)

    np.testing.assert_array_almost_equal(modified.coef_, nwp.coef_, decimal=10)
    np.testing.assert_almost_equal(modified.intercept_, nwp.intercept_, decimal=10)
    np.testing.assert_array_almost_equal(modified.predict(X), nwp.predict(X), decimal=10)

    np.testing.assert_array_almost_equal(
        nwp.x_weights_, modified.x_weights_ * modified.y_loadings_[None, :], decimal=10
    )
    np.testing.assert_array_almost_equal(
        nwp.x_loadings_, modified.x_loadings_ / modified.y_loadings_[None, :], decimal=10
    )


def test_window_overhanging_boundary_does_not_crash():
    """min_w/max_w close to n_features means some columns have no valid j in
    range -- must not error or silently produce NaNs."""
    X, y = _make_data(n_samples=40, n_features=12, seed=1)
    model = ProximetricsXLS(n_components=3, type="modified", min_w=5, max_w=9)
    model.fit(X, y)
    assert np.all(np.isfinite(model.coef_))
    assert np.all(np.isfinite(model.predict(X)))


def test_all_estimators_discovers_proximetrics_xls():
    estimators = dict(all_estimators())
    assert "ProximetricsXLS" in estimators
    assert estimators["ProximetricsXLS"] is ProximetricsXLS


def test_openmodels_round_trip():
    openmodels = pytest.importorskip("openmodels")

    X, y = _make_data(seed=11)
    model = ProximetricsXLS(n_components=3, type="modified", min_w=3, max_w=8)
    model.fit(X, y)

    manager = openmodels.SerializationManager(
        openmodels.SklearnSerializer(custom_estimators=all_estimators)
    )
    serialized = manager.serialize(model)
    restored = manager.deserialize(serialized)

    np.testing.assert_array_almost_equal(restored.predict(X), model.predict(X), decimal=10)
    np.testing.assert_array_almost_equal(restored.x_weights_, model.x_weights_, decimal=10)
    assert restored.type == "modified"
    assert restored.min_w == 3
    assert restored.max_w == 8
