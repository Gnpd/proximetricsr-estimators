"""Shared PLS/XLS decomposition core for proximetricsR-compatible estimators.

Mirrors ``src/processing_helpers.cpp::estimate_all_pls`` in proximetricsR,
parametrized by a per-iteration weight function so that
:class:`~proximetricsr_estimators.regression.ProximetricsPLS` and
:class:`~proximetricsr_estimators.regression.ProximetricsXLS` share exactly the
same (independently, numerically verified against R) deflation, projection and
``"nwp"``-rescaling logic -- only the weight-vector computation differs between
PLS and XLS.
"""

from typing import Callable

import numpy as np


def fit_pls_like(
    X: np.ndarray,
    y: np.ndarray,
    n_components: int,
    type: str,
    weight_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> dict:
    """Run proximetricsR's ``estimate_all_pls()`` algorithm.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,)
    n_components : int
    type : {"standard", "modified", "nwp"}
        Only used here to decide whether to apply NIRWise PLUS's ``"nwp"``
        post-hoc rescaling of ``x_weights``/``x_scores``/``x_loadings``/
        ``x_rotations`` (a relabelling for on-device score-space display, not
        the "regression_coefs are computed before this happens, so it does not
        change ``coef_``" -- see ``ProximetricsPLS``'s docstring). The weight
        vector itself is entirely delegated to ``weight_fn``.
    weight_fn : callable(X_deflated, y_centered) -> ndarray of shape (n_features,)
        Computes one component's weight vector from the current (iteratively
        X-deflated) data and the (once, up front) mean-centered response.

    Returns
    -------
    dict
        Keys: ``x_mean``, ``x_weights`` (n_features, n_components),
        ``x_loadings`` (same shape), ``y_loadings`` (n_components,),
        ``x_scores`` (n_samples, n_components), ``x_rotations``
        (n_features, n_components), ``coef`` (n_features,), ``intercept``
        (float).
    """
    n_samples, n_features = X.shape
    ncomp = n_components

    # A deterministic upfront check rather than relying on numpy to detect
    # near-singularity later: np.linalg.inv() only raises on an *exactly*
    # singular matrix, so ncomp > min(n_samples, n_features) reliably produces
    # silently nonsensical (not NaN, not an error) coefficients instead of
    # failing -- verified: requesting more components than features gave
    # coefficients ~100x the true scale with no warning or error at all.
    if ncomp > min(n_samples, n_features):
        raise ValueError(
            f"n_components={ncomp} exceeds min(n_samples, n_features)="
            f"{min(n_samples, n_features)} ({n_samples} samples, {n_features} "
            "features). Requesting more components than that produces numerically "
            "meaningless coefficients rather than a clear failure, so this is "
            "checked upfront."
        )

    y_mean = float(y.mean())
    y_centered = y - y_mean
    x_mean = X.mean(axis=0)
    X_deflated = X - x_mean

    weights = np.zeros((ncomp, n_features))
    x_loadings = np.zeros((ncomp, n_features))
    scores = np.zeros((n_samples, ncomp))
    y_loadings = np.zeros(ncomp)

    for i in range(ncomp):
        w = weight_fn(X_deflated, y_centered)
        if not np.any(w):
            raise ValueError(
                f"Component {i + 1} of {ncomp} has an all-zero weight vector -- "
                "there is not enough independent information left in X (relative "
                "to n_components and, for XLS, min_w/max_w) to fit this many "
                "components. Try fewer n_components or, for XLS, a narrower "
                "min_w/max_w window relative to the number of features."
            )
        t = X_deflated @ w
        tt = t @ t
        q = (y_centered @ t) / tt
        p = (t @ X_deflated) / tt

        weights[i] = w
        scores[:, i] = t
        y_loadings[i] = q
        x_loadings[i] = p

        X_deflated = X_deflated - np.outer(t, p)

    try:
        projection_m = np.linalg.inv(weights @ x_loadings.T) @ weights
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            f"Could not fit {ncomp} component(s): the weights/loadings matrix is "
            "singular, meaning the requested components are not all linearly "
            "independent for this data. This usually means n_components is too "
            "large relative to the number of features (and, for XLS, the "
            "min_w/max_w window) -- try fewer n_components."
        ) from exc
    coef_path = np.cumsum(projection_m * y_loadings[:, None], axis=0)

    if type == "nwp":
        weights = weights * y_loadings[:, None]
        scores = scores * y_loadings[None, :]
        x_loadings = x_loadings / y_loadings[:, None]
        projection_m = projection_m * y_loadings[:, None]

    return {
        "x_mean": x_mean,
        "x_weights": weights.T,
        "x_loadings": x_loadings.T,
        "y_loadings": y_loadings,
        "x_scores": scores,
        "x_rotations": projection_m.T,
        "coef": coef_path[-1],
        "intercept": y_mean,
    }


def pls_weights(X_deflated: np.ndarray, y_centered: np.ndarray, type: str) -> np.ndarray:
    """One iteration's weight vector for plain PLS, matching ``get_pls_weights()``'s
    ``"pls_standard"``/``"pls_modified"``/``"pls_nwp"`` branches.

    ``"standard"``: covariance-based weights, L2-normalized (classic NIPALS
    PLS1; numerically equivalent to ``sklearn.cross_decomposition.PLSRegression``).

    ``"modified"``/``"nwp"``: correlation-based weights (Shenk & Westerhaus's
    Modified PLS), *not* L2-normalized -- each entry is a per-column Pearson
    correlation with y, bounded in [-1, 1] individually rather than the vector
    as a whole.
    """
    if type == "standard":
        w = y_centered @ X_deflated
        return w / np.linalg.norm(w)

    x_centered = X_deflated - X_deflated.mean(axis=0)
    x_std = x_centered.std(axis=0, ddof=0)
    y_c = y_centered - y_centered.mean()
    y_std = y_c.std(ddof=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        w = (y_c @ x_centered) / (len(y_centered) * y_std * x_std)
    return np.nan_to_num(w, nan=0.0)


def xls_weights(
    X_deflated: np.ndarray, y_centered: np.ndarray, type: str, min_w: int, max_w: int
) -> np.ndarray:
    """One iteration's weight vector for XLS, matching ``get_pls_weights()``'s
    ``"xls_standard"``/``"xls_modified"``/``"xls_nwp"`` branches: for each
    feature pair (i, j) with ``min_w <= j - i <= max_w``, accumulates the
    covariance (``"standard"``) or correlation (``"modified"``/``"nwp"``)
    between y and the *difference* column ``X[:, i] - X[:, j]`` into
    ``w[i] += value`` and ``w[j] -= value``. ``"standard"`` is additionally
    L2-normalized afterward (mirroring ``"pls_standard"``).
    """
    n_samples, n_features = X_deflated.shape
    w = np.zeros(n_features)

    y_c = y_centered - y_centered.mean()
    y_std = y_c.std(ddof=0)
    use_corr = type in ("modified", "nwp")

    for i in range(n_features):
        j_start = i + min_w
        j_end = min(i + max_w, n_features - 1)
        if j_start > j_end:
            continue
        j_range = np.arange(j_start, j_end + 1)

        diffs = X_deflated[:, i : i + 1] - X_deflated[:, j_range]
        diffs_c = diffs - diffs.mean(axis=0)
        cov_vals = (y_c @ diffs_c) / n_samples

        if use_corr:
            diff_std = diffs_c.std(axis=0, ddof=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                vals = cov_vals / (diff_std * y_std)
            vals = np.nan_to_num(vals, nan=0.0)
        else:
            vals = cov_vals

        w[i] += vals.sum()
        np.add.at(w, j_range, -vals)

    if type == "standard":
        w = w / np.linalg.norm(w)

    return w
