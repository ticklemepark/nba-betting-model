"""Tests for src/models/calibration.py."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.models.calibration import calibrate_model, calibration_curve_data


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_fitted_logit(n=200):
    """Return a simple fitted LogisticRegression on synthetic binary data."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, 3))
    y = (X[:, 0] + rng.standard_normal(n) * 0.5 > 0).astype(int)
    model = LogisticRegression(random_state=42).fit(X, y)
    return model, X, y


# ---------------------------------------------------------------------------
# calibrate_model
# ---------------------------------------------------------------------------

class TestCalibrateModel:
    def test_returns_fitted_calibrator(self):
        model, X, y = _make_fitted_logit()
        cal = calibrate_model(model, X, y)
        # Must expose predict_proba
        probs = cal.predict_proba(X)
        assert probs.shape == (len(X), 2)

    def test_probs_in_zero_one(self):
        model, X, y = _make_fitted_logit()
        cal = calibrate_model(model, X, y)
        probs = cal.predict_proba(X)[:, 1]
        assert float(probs.min()) >= 0.0
        assert float(probs.max()) <= 1.0

    def test_probs_sum_to_one(self):
        model, X, y = _make_fitted_logit()
        cal = calibrate_model(model, X, y)
        probs = cal.predict_proba(X)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    def test_sigmoid_method_accepted(self):
        model, X, y = _make_fitted_logit()
        cal = calibrate_model(model, X, y, method="sigmoid")
        assert cal.predict_proba(X).shape == (len(X), 2)

    def test_invalid_method_raises(self):
        model, X, y = _make_fitted_logit()
        with pytest.raises(ValueError, match="method must be"):
            calibrate_model(model, X, y, method="platt_wrong")


# ---------------------------------------------------------------------------
# calibration_curve_data
# ---------------------------------------------------------------------------

class TestCalibrationCurveData:
    def test_returns_dataframe(self):
        rng = np.random.default_rng(1)
        y_true = rng.integers(0, 2, size=300)
        y_prob = rng.uniform(0, 1, size=300)
        result = calibration_curve_data(y_true, y_prob)
        assert isinstance(result, pd.DataFrame)

    def test_has_expected_columns(self):
        rng = np.random.default_rng(2)
        y_true = rng.integers(0, 2, size=300)
        y_prob = rng.uniform(0, 1, size=300)
        result = calibration_curve_data(y_true, y_prob)
        assert "mean_predicted" in result.columns
        assert "fraction_positive" in result.columns
        assert "n" in result.columns

    def test_mean_predicted_in_zero_one(self):
        rng = np.random.default_rng(3)
        y_true = rng.integers(0, 2, size=300)
        y_prob = rng.uniform(0, 1, size=300)
        result = calibration_curve_data(y_true, y_prob)
        assert (result["mean_predicted"] >= 0.0).all()
        assert (result["mean_predicted"] <= 1.0).all()

    def test_fraction_positive_in_zero_one(self):
        rng = np.random.default_rng(4)
        y_true = rng.integers(0, 2, size=300)
        y_prob = rng.uniform(0, 1, size=300)
        result = calibration_curve_data(y_true, y_prob)
        assert (result["fraction_positive"] >= 0.0).all()
        assert (result["fraction_positive"] <= 1.0).all()
