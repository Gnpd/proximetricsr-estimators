"""
proximetricsr-estimators: sklearn-compatible estimators for reconstructing
proximetricsR (BUCHI NIRWise-PLUS-compatible) models in Python.
"""

from .regression import NIRWiseLinearModel, ProximetricsPLS, ProximetricsXLS

__all__ = [
    "NIRWiseLinearModel",
    "ProximetricsPLS",
    "ProximetricsXLS",
]
