# tests/test_model.py
"""Tests for model module."""

import pytest
import numpy as np
from models.model import PredictionModel, EnsembleModel
from models.hyperparameters import get_hyperparameters


def test_prediction_model_initialization():
    """Test that PredictionModel initializes with different model types."""
    model = PredictionModel(model_type="random_forest")
    assert model.model_type == "random_forest"
    assert model.model is not None

    model = PredictionModel(model_type="logistic_regression")
    assert model.model_type == "logistic_regression"

    with pytest.raises(ValueError):
        PredictionModel(model_type="unknown_model")


def test_prediction_model_train_and_predict(sample_features_df):
    """Test training and prediction."""
    X, y = sample_features_df
    model = PredictionModel(model_type="random_forest")
    model.train(X, y)
    preds = model.predict_proba(X)
    assert preds.shape == (100, 3)  # 3 classes
    labels = model.predict(X)
    assert labels.shape == (100,)


def test_model_save_load(tmp_path, sample_features_df):
    """Test saving and loading a model."""
    X, y = sample_features_df
    model = PredictionModel(model_type="logistic_regression")
    model.train(X, y)

    save_path = tmp_path / "model.pkl"
    model.save(str(save_path))
    assert save_path.exists()

    loaded = PredictionModel(model_path=str(save_path))
    assert loaded.model is not None
    preds_loaded = loaded.predict_proba(X)
    preds_original = model.predict_proba(X)
    np.testing.assert_array_equal(preds_loaded, preds_original)


def test_ensemble_model():
    """Test ensemble model with dummy callables."""
    def dummy_model1(x):
        return np.array([[0.8, 0.2]])

    def dummy_model2(x):
        return np.array([[0.6, 0.4]])

    ensemble = EnsembleModel(weights={"model1": 0.7, "model2": 0.3})
    ensemble.add_model("model1", dummy_model1)
    ensemble.add_model("model2", dummy_model2)

    X = np.random.randn(1, 40)
    pred = ensemble.predict_proba(X)
    expected = (0.8 * 0.7 + 0.6 * 0.3, 0.2 * 0.7 + 0.4 * 0.3)
    np.testing.assert_almost_equal(pred[0], expected)


def test_hyperparameters():
    """Test hyperparameter function."""
    params = get_hyperparameters("random_forest")
    assert "n_estimators" in params
    assert "max_depth" in params

    params = get_hyperparameters("xgboost")
    assert "learning_rate" in params

    with pytest.raises(ValueError):
        get_hyperparameters("unknown")