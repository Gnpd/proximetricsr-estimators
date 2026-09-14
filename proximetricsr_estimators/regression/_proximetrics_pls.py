"""
The module :mod:`proximetricsr_estimators.regression._proximetrics_pls` implements
a Python-native, verified reimplementation of proximetricsR's PLS fitting engine
(``fit_plsr()``'s ``"standard"``, ``"modified"`` and ``"nwp"`` algorithm variants),
mirroring ``src/processing_helpers.cpp::estimate_all_pls`` in the R package exactly.
"""

# License: MIT

from numbers import Integral

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted, validate_data

from ._pls_core import fit_pls_like, pls_weights


class ProximetricsPLS(TransformerMixin, RegressorMixin, BaseEstimator):
    """Python-native reimplementation of proximetricsR's ``fit_plsr()`` PLS engine.

    Unlike :class:`~proximetricsr_estimators.regression.NIRWiseLinearModel` (which
    only carries the final affine ``coef_``/``intercept_`` for predict-only
    reconstruction of an R-fitted model), this class performs the actual PLS
    decomposition in Python: ``x_weights_``, ``x_loadings_``, ``y_loadings_``,
    ``x_scores_`` and ``x_rotations_`` are all available, matching sklearn's own
    ``PLSRegression`` attribute naming. This is what a model needs to carry for
    applicability-domain diagnostics (leverage / Q-residual, e.g. via
    ``chemotools.outliers``) or for reconstructing a full proximetricsR
    ``spectral_fit`` (as needed by the ProxiMate ``.cal``/``.prj`` device-format
    writers), not just bare prediction.

    Parameters
    ----------
    n_components : int, default=2
        Number of PLS components.

    type : {"standard", "modified", "nwp"}, default="standard"
        Which proximetricsR ``fit_plsr()`` algorithm variant to reproduce:

        - ``"standard"``: covariance-based weights, one NIPALS iteration per
          component, X deflated only. Verified numerically equivalent (up to
          floating-point precision and PLS's inherent per-component sign
          ambiguity) to ``sklearn.cross_decomposition.PLSRegression(scale=False)``
          -- prefer that class (or ``chemotools.regression.PLSRegression``,
          a thin subclass of it) for ``"standard"`` fits; this class exists
          primarily for ``"modified"``/``"nwp"``, and supports ``"standard"``
          mainly so all three variants share one implementation and API,
          matching ``fit_plsr()``'s own ``type`` argument.
        - ``"modified"``: correlation-based weights (Shenk, J. S., & Westerhaus,
          M. O. (1991). *Crop Science*, 31(2), 409-413), unnormalized -- a
          different, real algorithm with no scikit-learn equivalent.
        - ``"nwp"``: identical fitted *predictions* to ``"modified"`` (verified:
          the two produce the same ``coef_``/``intercept_`` for the same data and
          ``n_components``) -- it additionally rescales ``x_weights_``/
          ``x_scores_``/``x_loadings_``/``x_rotations_`` by the per-component
          y-loading (NIRWise PLUS's "slope correction"), matching BUCHI's
          on-device score-space convention. Only relevant if you need those
          intermediate attributes to match NIRWise PLUS exactly; for prediction
          alone, ``"modified"`` and ``"nwp"`` are interchangeable.

    Attributes
    ----------
    x_mean_ : ndarray of shape (n_features,)
    x_weights_ : ndarray of shape (n_features, n_components)
    x_loadings_ : ndarray of shape (n_features, n_components)
    y_loadings_ : ndarray of shape (n_components,)
    x_scores_ : ndarray of shape (n_samples, n_components)
    x_rotations_ : ndarray of shape (n_features, n_components)
        Projection matrix such that ``x_scores_ == (X - x_mean_) @ x_rotations_``
        in one step (proximetricsR's ``projection_m``, transposed to match
        sklearn's ``x_rotations_`` convention).
    coef_ : ndarray of shape (n_features,)
        Final regression coefficients, using all ``n_components``.
    intercept_ : float
    n_features_in_ : int

    References
    ----------
    .. [1] Shenk, J. S., & Westerhaus, M. O. (1991). The application of near
        infrared reflectance spectroscopy (NIRS) to compositional analysis of
        agricultural products. Crop Science, 31(2), 409-413.
    """

    _parameter_constraints: dict = {
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "type": [StrOptions({"standard", "modified", "nwp"})],
    }

    def __init__(self, n_components: int = 2, type: str = "standard"):
        self.n_components = n_components
        self.type = type

    def fit(self, X, y):
        """Fit the PLS model, reproducing proximetricsR's ``estimate_all_pls()``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        self : ProximetricsPLS
        """
        self._validate_params()
        X, y = validate_data(self, X, y, ensure_2d=True, dtype=np.float64, y_numeric=True)

        result = fit_pls_like(
            X, y, self.n_components, self.type,
            weight_fn=lambda X_deflated, y_centered: pls_weights(X_deflated, y_centered, self.type),
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
