"""
The module :mod:`proximetricsr_estimators.regression._nirwise_linear_model`
implements a predict-equivalent reconstruction of a proximetricsR `spectral_fit`
model (PLS or XLS, BUCHI NIRWise-PLUS-compatible algorithms).
"""

# License: MIT

from numbers import Integral

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted, validate_data


class NIRWiseLinearModel(RegressorMixin, BaseEstimator):
    """Predict-equivalent reconstruction of a proximetricsR ``spectral_fit`` model.

    proximetricsR (an R package) fits PLS/XLS chemometric models reproducing
    BUCHI NIRWise PLUS calibration software, using several algorithm variants
    (``"standard"``, ``"modified"``, and a proprietary ``"nwp"`` exact-match mode).
    Regardless of variant, prediction from an already-fitted model is always affine:

        predict(X) = (X - x_means_) @ coef_.T + intercept_

    This class exists so that a model exported from proximetricsR (e.g. via
    ``openmodels``, with ``x_means_``/``coef_``/``intercept_`` set directly from the
    R-fitted values) can be loaded and used for prediction in Python as a real,
    working scikit-learn estimator, without re-implementing any of proximetricsR's
    fitting algorithms and without depending on R at predict time.

    ``fit_method``/``type``/``min_w``/``max_w`` are provenance-only constructor
    parameters, mirroring proximetricsR's ``fit_plsr()``/``fit_xlsr()`` — they are
    not used by ``predict()``, but let a reloaded model still report which R-side
    method produced it, instead of flattening that identity into a bare
    ``LinearRegression``.

    ``fit()`` is provided so this remains a fully-conformant, independently usable
    scikit-learn estimator: it performs a plain ordinary-least-squares fit after
    centering ``X`` by its column means (and ``y`` by its mean, matching
    proximetricsR's convention of ``intercept = mean(Y)``). This is **not** what
    produces the attributes on a model reconstructed from an R-exported file — those
    are set directly from the R-fitted coefficients by the deserializer. In
    particular, ``fit()`` does not reproduce PLS/XLS/NWP component-weighting; it is
    an honest, literal implementation of what ``predict()`` actually computes, not a
    re-implementation of proximetricsR's training algorithms.

    Parameters
    ----------
    fit_method : {"plsr", "xlsr"}, default="plsr"
        Which proximetricsR fitting function (``fit_plsr()``/``fit_xlsr()``)
        produced the original model. Provenance only.

    type : {"standard", "modified", "nwp"}, default="standard"
        The proximetricsR algorithm variant. Provenance only. Note that a model
        using ``"nwp"`` may still be loaded here (prediction is affine regardless),
        but such a model's *fitted* attributes were necessarily computed in R —
        this class cannot reproduce ``"nwp"`` fitting itself.

    ncomp : int, default=1
        Number of PLS/XLS components used by the original R-fitted model.
        Provenance only — ``predict()`` uses ``coef_`` directly and does not
        truncate to this many components.

    min_w : int or None, default=None
        For ``fit_method="xlsr"``, the minimum XLS window size used when fitting
        in R. Provenance only.

    max_w : int or None, default=None
        For ``fit_method="xlsr"``, the maximum XLS window size used when fitting
        in R. Provenance only.

    Attributes
    ----------
    x_means_ : ndarray of shape (n_features,)
        Column means of the training spectra, subtracted from ``X`` at predict
        time (proximetricsR's ``spectral_fit$x_means``).

    coef_ : ndarray of shape (n_features,)
        Regression coefficients for the selected number of components
        (proximetricsR's ``spectral_fit$coefficients[ncomp, ]``).

    intercept_ : float
        The regression intercept (proximetricsR's ``spectral_fit$intercept``,
        equal to ``mean(Y)``).

    n_features_in_ : int
        Number of spectral variables seen during fit.

    wavenumbers_ : ndarray of shape (n_features_in_,), optional
        The wavelength/wavenumber grid the coefficients correspond to
        (proximetricsR's final processed grid). Provenance only - set when the model
        is reconstructed from an R export, not by ``fit()``, and not read by
        ``predict()``.

        Deliberately *not* stored as scikit-learn's ``feature_names_in_``. Inside an
        exported pipeline the model step always receives the output of the preceding
        chemotools transformer, so that attribute can never be satisfied: with array
        input it warns on every ``predict()``, and under ``set_output("pandas")`` the
        upstream transformer relabels the columns ``"x0"``, ``"x1"``, ..., which then
        mismatch and raise ``ValueError``. Callers wanting the grid as strings can use
        ``model.wavenumbers_.astype(str)``.

    Examples
    --------
    >>> import numpy as np
    >>> from proximetricsr_estimators.regression import NIRWiseLinearModel
    >>> model = NIRWiseLinearModel(fit_method="plsr", type="standard", ncomp=3)
    >>> model.x_means_ = np.zeros(4)
    >>> model.coef_ = np.array([0.1, 0.2, -0.1, 0.05])
    >>> model.intercept_ = 1.5
    >>> model.n_features_in_ = 4
    >>> model.predict(np.ones((2, 4)))
    array([1.75, 1.75])
    """

    _parameter_constraints: dict = {
        "fit_method": [StrOptions({"plsr", "xlsr"})],
        "type": [StrOptions({"standard", "modified", "nwp"})],
        "ncomp": [Interval(Integral, 1, None, closed="left")],
        "min_w": [Interval(Integral, 1, None, closed="left"), None],
        "max_w": [Interval(Integral, 1, None, closed="left"), None],
    }

    def __init__(
        self,
        fit_method="plsr",
        type="standard",
        ncomp=1,
        min_w=None,
        max_w=None,
    ):
        self.fit_method = fit_method
        self.type = type
        self.ncomp = ncomp
        self.min_w = min_w
        self.max_w = max_w

    def fit(self, X, y):
        """Fit a plain OLS model after centering X and y by their means.

        See the class docstring: this is a literal implementation of what
        ``predict()`` computes, not a reproduction of proximetricsR's PLS/XLS/NWP
        fitting algorithms.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training spectra.

        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : NIRWiseLinearModel
            The fitted estimator.
        """
        self._validate_params()
        X, y = validate_data(self, X, y, ensure_2d=True, dtype=np.float64, y_numeric=True)

        self.x_means_ = X.mean(axis=0)
        self.intercept_ = float(y.mean())

        X_centered = X - self.x_means_
        y_centered = y - self.intercept_

        coef, *_ = np.linalg.lstsq(X_centered, y_centered, rcond=None)
        self.coef_ = coef

        return self

    def predict(self, X):
        """Predict target values for ``X``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Spectra to predict from.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        check_is_fitted(self, ["x_means_", "coef_", "intercept_"])
        X = validate_data(self, X, ensure_2d=True, dtype=np.float64, reset=False)

        X_centered = X - self.x_means_
        return X_centered @ self.coef_ + self.intercept_
