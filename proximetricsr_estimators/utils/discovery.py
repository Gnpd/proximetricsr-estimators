"""
The :mod:`proximetricsr_estimators.utils.discovery` module includes utilities to
discover estimator classes from the `proximetricsr_estimators` package.

Mirrors `chemotools.utils.discovery.all_estimators` so both can be passed together
as a list to `openmodels.SklearnSerializer(custom_estimators=[...])`.
"""

# Adapted from chemotools.utils.discovery, itself adapted from scikit-learn

import inspect
import pkgutil
import warnings
from importlib import import_module
from operator import itemgetter
from pathlib import Path

from sklearn.base import BaseEstimator

_MODULE_TO_IGNORE = {"tests", "utils"}


def _iter_proximetricsr_estimators_modules():
    """Yield importable proximetricsr_estimators submodules.

    Submodules that fail to import are skipped with a warning instead of
    propagating the error, so discovery keeps working for callers who don't
    need the affected submodule.
    """
    root = str(Path(__file__).parent.parent)  # proximetricsr_estimators package root
    for _, module_name, _ in pkgutil.walk_packages(
        path=[root], prefix="proximetricsr_estimators."
    ):
        module_parts = module_name.split(".")
        if any(part in _MODULE_TO_IGNORE for part in module_parts):
            continue

        try:
            yield import_module(module_name)
        except ImportError as exc:
            warnings.warn(
                f"Skipping '{module_name}' during discovery: {exc}",
                stacklevel=3,
            )


def all_estimators(type_filter=None):
    """Get a list of all estimators from `proximetricsr_estimators`.

    This function crawls the module and gets all classes that inherit from
    `BaseEstimator`. Classes that are defined in test modules are not included.

    Parameters
    ----------
    type_filter : {"classifier", "regressor", "cluster", "transformer"} \
            or list of such str, default=None
        Which kind of estimators should be returned. If None, no filter is
        applied and all estimators are returned.

    Returns
    -------
    estimators : list of tuples
        List of (name, class), where ``name`` is the class name as string and
        ``class`` is the actual type of the class.

    Examples
    --------
    >>> from proximetricsr_estimators.utils.discovery import all_estimators
    >>> estimators = all_estimators()
    >>> type(estimators)
    <class 'list'>
    """

    def is_abstract(c):
        return bool(getattr(c, "__abstractmethods__", False))

    all_classes = []
    for module in _iter_proximetricsr_estimators_modules():
        classes = inspect.getmembers(module, inspect.isclass)
        classes = [
            (name, cls)
            for name, cls in classes
            if not name.startswith("_") and cls.__module__ == module.__name__
        ]
        all_classes.extend(classes)

    all_classes = set(all_classes)

    estimators = [
        (name, cls)
        for name, cls in all_classes
        if issubclass(cls, BaseEstimator)
        and name != "BaseEstimator"
        and not is_abstract(cls)
    ]

    if type_filter is not None:
        from sklearn.base import ClassifierMixin, ClusterMixin, RegressorMixin, TransformerMixin

        if not isinstance(type_filter, list):
            type_filter = [type_filter]
        else:
            type_filter = list(type_filter)  # copy
        filtered_estimators = []
        filters = {
            "classifier": ClassifierMixin,
            "regressor": RegressorMixin,
            "transformer": TransformerMixin,
            "cluster": ClusterMixin,
        }
        for name, mixin in filters.items():
            if name in type_filter:
                type_filter.remove(name)
                filtered = [est for est in estimators if issubclass(est[1], mixin)]
                filtered_estimators.extend(filtered)
        estimators = filtered_estimators
        if type_filter:
            raise ValueError(
                "Parameter type_filter must be 'classifier', 'regressor', "
                f"'transformer', 'cluster' or None, got {repr(type_filter)}."
            )

    return sorted(set(estimators), key=itemgetter(0))
