"""Probability calibration for classifier outputs.

Calibrates a pre-fitted classifier on a held-out validation set.

Two methods:
    "isotonic" — IsotonicRegression (non-parametric). Better for larger val
                  sets (n >= 1000). Preferred default.
    "sigmoid"  — Platt scaling via a 1D LogisticRegression. Better for small
                  val sets.

Note: sklearn 1.8 removed cv="prefit" from CalibratedClassifierCV, so we
implement calibration manually to avoid version-specific API changes.

Usage:
    from src.models.calibration import calibrate_model, calibration_curve_data

    calibrated = calibrate_model(fitted_model, X_val, y_val)
    probs = calibrated.predict_proba(X_test)[:, 1]

    curve = calibration_curve_data(y_test, probs)
    # curve columns: mean_predicted, fraction_positive, n
"""

import logging

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve

log = logging.getLogger(__name__)

_VALID_METHODS = ("isotonic", "sigmoid")


# ---------------------------------------------------------------------------
# Calibrated model wrapper
# ---------------------------------------------------------------------------

class CalibratedModel:
    """A pre-fitted classifier with an isotonic or sigmoid calibration layer.

    Wraps any classifier that exposes predict_proba(X) → (n, 2) array.
    After calibration, predict_proba() applies the learned mapping from
    raw probabilities to calibrated probabilities.
    """

    def __init__(self, base_model, calibrator, method: str):
        self.base_model  = base_model
        self.calibrator  = calibrator
        self.method      = method

    def predict_proba(self, X) -> np.ndarray:
        """Return shape-(n, 2) array: [P(class 0), P(class 1)]."""
        raw = self.base_model.predict_proba(X)[:, 1]

        if self.method == "isotonic":
            cal = np.clip(self.calibrator.predict(raw), 0.0, 1.0)
        else:
            cal = np.clip(
                self.calibrator.predict_proba(raw.reshape(-1, 1))[:, 1],
                0.0, 1.0,
            )

        return np.column_stack([1.0 - cal, cal])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calibrate_model(
    model,
    X_cal,
    y_cal,
    method: str = "isotonic",
) -> CalibratedModel:
    """Wrap a pre-fitted classifier with probability calibration.

    Args:
        model:  A fitted classifier exposing predict_proba() → (n, 2).
        X_cal:  Feature matrix for calibration (held-out validation set).
        y_cal:  True binary labels for calibration.
        method: "isotonic" (default) or "sigmoid".

    Returns:
        CalibratedModel with the same predict_proba() interface.

    Raises:
        ValueError: If method is not recognised.
    """
    if method not in _VALID_METHODS:
        raise ValueError(
            f"method must be one of {_VALID_METHODS}, got {method!r}"
        )

    raw_probs = model.predict_proba(X_cal)[:, 1]

    if method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(raw_probs, y_cal)
    else:
        calibrator = LogisticRegression(C=1.0, max_iter=500, random_state=42)
        calibrator.fit(raw_probs.reshape(-1, 1), y_cal)

    log.info(
        "Calibrated model using %s on %d samples.", method, len(y_cal)
    )
    return CalibratedModel(model, calibrator, method)


def calibration_curve_data(
    y_true,
    y_prob,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compute calibration curve data for reporting.

    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities for the positive class.
        n_bins: Number of bins to divide the [0, 1] probability range.

    Returns:
        DataFrame with columns:
            mean_predicted   — mean predicted probability in each bin
            fraction_positive — actual positive rate in each bin
            n                — number of samples in each bin
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )

    bin_edges   = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges[1:-1])
    bin_counts  = np.bincount(bin_indices, minlength=n_bins)

    return pd.DataFrame({
        "mean_predicted":    mean_predicted_value,
        "fraction_positive": fraction_of_positives,
        "n":                 bin_counts[: len(mean_predicted_value)],
    })
