"""Tests for proximetricsr_estimators.regression.ProximetricsPLS.

`"standard"` is checked against a real sklearn.cross_decomposition.PLSRegression
fit on the same data (numerically equivalent, verified once against R too -- see
the module docstring in _proximetrics_pls.py and the session record for the exact
cross-check against proximetricsR's own src/processing_helpers.cpp::estimate_all_pls
on synthetic data: max abs differences on the order of 1e-14 to 1e-16 across
coef_/intercept_/x_weights_/x_loadings_/x_scores_/x_rotations_/y_loadings_ for all
three types). That R comparison isn't reproduced here since it requires an R
installation; this file re-verifies the parts checkable from Python alone and
guards the documented "modified"/"nwp" relationship.
"""

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.utils.estimator_checks import check_estimator

from proximetricsr_estimators.regression import ProximetricsPLS
from proximetricsr_estimators.utils.discovery import all_estimators


def _make_data(n_samples=60, n_features=25, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    true_beta = rng.normal(size=n_features) * 0.3
    y = X @ true_beta + rng.normal(scale=0.05, size=n_samples)
    return X, y


def test_compliance_proximetrics_pls():
    check_estimator(ProximetricsPLS())


def test_clear_error_on_too_many_components():
    """n_components larger than min(n_samples, n_features) is checked upfront
    (see _pls_core.py::fit_pls_like -- numpy's inv() does not reliably raise on
    the resulting near-singular matrix, it silently returns numerically
    meaningless coefficients instead)."""
    X, y = _make_data(n_samples=30, n_features=5, seed=4)
    model = ProximetricsPLS(n_components=8, type="standard")
    with pytest.raises(ValueError, match="n_components.*exceeds"):
        model.fit(X, y)


@pytest.mark.parametrize("type_", ["standard", "modified", "nwp"])
def test_predict_matches_manual_affine_computation(type_):
    X, y = _make_data()
    model = ProximetricsPLS(n_components=4, type=type_)
    model.fit(X, y)

    manual = (X - model.x_mean_) @ model.coef_ + model.intercept_
    np.testing.assert_array_almost_equal(model.predict(X), manual, decimal=10)


@pytest.mark.parametrize("type_", ["standard", "modified", "nwp"])
def test_transform_matches_x_rotations(type_):
    X, y = _make_data()
    model = ProximetricsPLS(n_components=4, type=type_)
    model.fit(X, y)

    manual_scores = (X - model.x_mean_) @ model.x_rotations_
    np.testing.assert_array_almost_equal(model.transform(X), manual_scores, decimal=10)


def test_standard_matches_real_sklearn_pls_regression():
    """The one part of the R-verified claim re-checkable without R: "standard"
    reproduces classic NIPALS PLS1, so it must also match sklearn's own
    PLSRegression(scale=False) on arbitrary data, not just the one fixture used
    against R."""
    X, y = _make_data(seed=7)
    ncomp = 5

    ours = ProximetricsPLS(n_components=ncomp, type="standard")
    ours.fit(X, y)

    sk = PLSRegression(n_components=ncomp, scale=False)
    sk.fit(X, y)

    np.testing.assert_array_almost_equal(ours.coef_, np.asarray(sk.coef_).reshape(-1), decimal=8)
    np.testing.assert_almost_equal(ours.intercept_, np.asarray(sk.intercept_).reshape(-1)[0], decimal=8)

    # Weight/loading vectors are only unique up to a per-component sign in PLS;
    # compare direction (cosine similarity ~ +-1) rather than raw values.
    for i in range(ncomp):
        cos_w = np.dot(ours.x_weights_[:, i], sk.x_weights_[:, i]) / (
            np.linalg.norm(ours.x_weights_[:, i]) * np.linalg.norm(sk.x_weights_[:, i])
        )
        assert abs(abs(cos_w) - 1) < 1e-8


def test_modified_and_nwp_predict_identically():
    """Documented relationship: "nwp" only rescales the intermediate
    weights_/scores_/x_loadings_/x_rotations_ (NIRWise PLUS's slope correction
    for on-device score-space display) -- final predictions are identical to
    "modified" for the same data and n_components."""
    X, y = _make_data(seed=3)
    ncomp = 4

    modified = ProximetricsPLS(n_components=ncomp, type="modified").fit(X, y)
    nwp = ProximetricsPLS(n_components=ncomp, type="nwp").fit(X, y)

    np.testing.assert_array_almost_equal(modified.coef_, nwp.coef_, decimal=10)
    np.testing.assert_almost_equal(modified.intercept_, nwp.intercept_, decimal=10)
    np.testing.assert_array_almost_equal(modified.predict(X), nwp.predict(X), decimal=10)

    # And the documented rescaling relationship itself, component-wise.
    np.testing.assert_array_almost_equal(
        nwp.x_weights_, modified.x_weights_ * modified.y_loadings_[None, :], decimal=10
    )
    np.testing.assert_array_almost_equal(
        nwp.x_loadings_, modified.x_loadings_ / modified.y_loadings_[None, :], decimal=10
    )


def test_all_estimators_discovers_proximetrics_pls():
    estimators = dict(all_estimators())
    assert "ProximetricsPLS" in estimators
    assert estimators["ProximetricsPLS"] is ProximetricsPLS


def test_openmodels_round_trip():
    openmodels = pytest.importorskip("openmodels")

    X, y = _make_data(seed=11)
    model = ProximetricsPLS(n_components=3, type="modified")
    model.fit(X, y)

    manager = openmodels.SerializationManager(
        openmodels.SklearnSerializer(custom_estimators=all_estimators)
    )
    serialized = manager.serialize(model)
    restored = manager.deserialize(serialized)

    np.testing.assert_array_almost_equal(restored.predict(X), model.predict(X), decimal=10)
    np.testing.assert_array_almost_equal(restored.x_weights_, model.x_weights_, decimal=10)
    np.testing.assert_array_almost_equal(restored.x_rotations_, model.x_rotations_, decimal=10)
    assert restored.type == "modified"
    assert restored.n_components == 3
