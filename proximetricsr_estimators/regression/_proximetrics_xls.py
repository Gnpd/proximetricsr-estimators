"""
The module :mod:`proximetricsr_estimators.regression._proximetrics_xls` implements
a Python-native, verified reimplementation of proximetricsR's XLS (extended PLS)
fitting engine (``fit_xlsr()``'s ``"standard"``, ``"modified"`` and ``"nwp"``
algorithm variants), mirroring ``src/processing_helpers.cpp::estimate_all_pls``
(dispatched through ``get_pls_weights()``'s ``"xls_*"`` branches) exactly.
"""

# License: MIT

from numbers import Integral

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted, validate_data

from ._pls_core import fit_pls_like, xls_weights


class ProximetricsXLS(TransformerMixin, RegressorMixin, BaseEstimator):
    """Python-native reimplementation of proximetricsR's ``fit_xlsr()`` XLS engine.

    XLS ("extended" PLS) shares the exact same component-deflation, projection
    and ``"nwp"``-rescaling machinery as PLS (see
    :class:`~proximetricsr_estimators.regression.ProximetricsPLS`, which this
    class shares an internal core with) -- the only difference is how each
    component's weight vector is computed. Instead of correlating/covarying y
    directly against each spectral variable, XLS correlates/covaries y against
    the *difference* between pairs of variables ``min_w`` to ``max_w`` apart
    (``X[:, i] - X[:, j]``), accumulating each pair's contribution into
    ``w[i] += value`` / ``w[j] -= value``. This is a windowed, locally
    derivative-like weighting scheme with no equivalent in scikit-learn or
    chemotools.

    Parameters
    ----------
    n_components : int, default=2
        Number of XLS components.

    type : {"standard", "modified", "nwp"}, default="standard"
        Which proximetricsR ``fit_xlsr()`` algorithm variant to reproduce --
        see :class:`ProximetricsPLS` for what each variant means; the same
        covariance-vs-correlation-vs-nwp-rescaling distinctions apply here,
        just computed over variable-pair differences instead of single
        variables. As with PLS, ``"modified"`` and ``"nwp"`` produce identical
        predictions for the same data/``n_components``/``min_w``/``max_w`` --
        ``"nwp"`` only rescales the intermediate weights/scores/loadings.

    min_w : int, default=3
        Minimum window size: only pairs with ``j - i >= min_w`` contribute.

    max_w : int, default=15
        Maximum window size: only pairs with ``j - i <= max_w`` contribute.

    Attributes
    ----------
    x_mean_ : ndarray of shape (n_features,)
    x_weights_ : ndarray of shape (n_features, n_components)
    x_loadings_ : ndarray of shape (n_features, n_components)
    y_loadings_ : ndarray of shape (n_components,)
    x_scores_ : ndarray of shape (n_samples, n_components)
    x_rotations_ : ndarray of shape (n_features, n_components)
        Projection matrix such that ``x_scores_ == (X - x_mean_) @ x_rotations_``
        in one step (proximetricsR's ``projection_m``, transposed).
    coef_ : ndarray of shape (n_features,)
        Final regression coefficients, using all ``n_components``.
    intercept_ : float
    n_features_in_ : int
    """

    _parameter_constraints: dict = {
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "type": [StrOptions({"standard", "modified", "nwp"})],
        "min_w": [Interval(Integral, 1, None, closed="left")],
        "max_w": [Interval(Integral, 1, None, closed="left")],
    }

    def __init__(
        self,
        n_components: int = 2,
        type: str = "standard",
        min_w: int = 3,
        max_w: int = 15,
    ):
        self.n_components = n_components
        self.type = type
        self.min_w = min_w
        self.max_w = max_w

    def fit(self, X, y):
        """Fit the XLS model, reproducing proximetricsR's ``estimate_all_pls()``
        with XLS-style windowed-difference weights.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        self : ProximetricsXLS
        """
        self._validate_params()
        if self.min_w >= self.max_w:
            raise ValueError("'min_w' must be less than 'max_w'.")
        X, y = validate_data(self, X, y, ensure_2d=True, dtype=np.float64, y_numeric=True)

        result = fit_pls_like(
            X, y, self.n_components, self.type,
            weight_fn=lambda X_deflated, y_centered: xls_weights(
                X_deflated, y_centered, self.type, self.min_w, self.max_w
            ),
        )

        self.x_mean_ = result["x_mean"]
        self.x_weights_ = result["x_weights"]
        self.x_loadings_ = result["x_loadings"]
        self.y_loadings_ = result["y_loadings"]
        self.x_scores_ = result["x_scores"]
        self.x_rotations_ = result["x_rotations"]
        self.coef_ = result["coef"]
        self.intercept_ = result["intercept"]

        return self

    def predict(self, X):
        """Predict target values for ``X``, using all ``n_components``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
        """
        check_is_fitted(self, ["x_mean_", "coef_", "intercept_"])
        X = validate_data(self, X, ensure_2d=True, dtype=np.float64, reset=False)
        return (X - self.x_mean_) @ self.coef_ + self.intercept_

    def transform(self, X):
        """Project ``X`` onto the score space, in one step via ``x_rotations_``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        x_scores : ndarray of shape (n_samples, n_components)
        """
        check_is_fitted(self, ["x_mean_", "x_rotations_"])
        X = validate_data(self, X, ensure_2d=True, dtype=np.float64, reset=False)
        return (X - self.x_mean_) @ self.x_rotations_
