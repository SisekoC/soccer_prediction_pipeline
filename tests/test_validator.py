# tests/test_validator.py
"""Tests for the validator module."""

import pytest
import numpy as np
from validation.validator import Validator


def test_validator_initialization(tmp_path):
    """Test validator initializes with optional history path."""
    history_file = tmp_path / "history.json"
    validator = Validator(history_path=str(history_file))
    assert validator.history == []


def test_validate_single_prediction_probabilities():
    """Test validation with probability predictions."""
    validator = Validator()
    pred = np.array([0.7, 0.2, 0.1])
    actual = "H"
    class_mapping = {0: "H", 1: "D", 2: "A"}

    result = validator.validate(pred, actual, class_mapping=class_mapping)

    assert result["correct"] is True
    assert result["predicted"] == "H"
    assert result["actual"] == "H"
    assert "log_loss_contribution" in result
    assert "brier_score_contribution" in result
    assert result["confidence"] == 0.7


def test_validate_single_prediction_class():
    """Test validation with hard class predictions."""
    validator = Validator()
    pred = 1  # class index
    actual = 1
    result = validator.validate(pred, actual)

    assert result["correct"] is True
    assert result["predicted"] == "1"
    assert result["actual"] == "1"
    assert "log_loss_contribution" not in result  # not computed without probabilities


def test_validate_batch():
    """Test batch validation."""
    validator = Validator()
    predictions = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.3, 0.3, 0.4]])
    actuals = ["H", "D", "A"]
    class_mapping = {0: "H", 1: "D", 2: "A"}

    results = validator.validate_batch(predictions, actuals, class_mapping=class_mapping, store=False)

    assert len(results) == 3
    assert results[0]["correct"] is True
    assert results[1]["correct"] is True
    assert results[2]["correct"] is True  # since highest prob is A (index 2)


def test_aggregate_metrics():
    """Test aggregate metrics calculation."""
    validator = Validator()
    validator.history = [
        {"accuracy_contribution": 1, "log_loss_contribution": 0.5, "brier_score_contribution": 0.2},
        {"accuracy_contribution": 0, "log_loss_contribution": 2.0, "brier_score_contribution": 0.6},
        {"accuracy_contribution": 1, "log_loss_contribution": 0.3, "brier_score_contribution": 0.1},
    ]

    metrics = validator.get_aggregate_metrics()
    assert metrics["accuracy"] == 2/3
    assert metrics["avg_log_loss"] == pytest.approx((0.5+2.0+0.3)/3)
    assert metrics["avg_brier_score"] == pytest.approx((0.2+0.6+0.1)/3)
    assert metrics["n_validations"] == 3