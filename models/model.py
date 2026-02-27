# models/model.py
"""Model training, prediction, and ensemble classes for soccer prediction pipeline."""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from typing import Optional, Dict, Any, Union, List, Callable

from .hyperparameters import get_hyperparameters
from ..utils.logger import get_logger
from ..utils.exceptions import ModelError

logger = get_logger(__name__)


class PredictionModel:
    """
    Unified interface for training and predicting with various classifiers.
    Supports scikit‑learn compatible models and XGBoost/LightGBM/CatBoost.
    """

    def __init__(
        self,
        model_type: str = "random_forest",
        hyperparameters: Optional[Dict[str, Any]] = None,
        model_path: Optional[str] = None,
    ):
        """
        Args:
            model_type: Type of model. Supported:
                'random_forest', 'xgboost', 'lightgbm', 'catboost', 'logistic_regression'
            hyperparameters: Optional override of default hyperparameters.
            model_path: If provided, load model from this path instead of creating new.
        """
        self.model_type = model_type
        self.hyperparameters = hyperparameters or get_hyperparameters(model_type)
        self.model = None

        if model_path:
            self.load(model_path)
        else:
            self._initialize_model()

    def _initialize_model(self) -> None:
        """Create a new model instance based on model_type."""
        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(**self.hyperparameters)
        elif self.model_type == "xgboost":
            try:
                import xgboost as xgb
                self.model = xgb.XGBClassifier(**self.hyperparameters)
            except ImportError:
                raise ModelError(
                    "XGBoost not installed. Please install with 'pip install xgboost'"
                )
        elif self.model_type == "lightgbm":
            try:
                import lightgbm as lgb
                self.model = lgb.LGBMClassifier(**self.hyperparameters)
            except ImportError:
                raise ModelError(
                    "LightGBM not installed. Please install with 'pip install lightgbm'"
                )
        elif self.model_type == "catboost":
            try:
                from catboost import CatBoostClassifier
                self.model = CatBoostClassifier(**self.hyperparameters)
            except ImportError:
                raise ModelError(
                    "CatBoost not installed. Please install with 'pip install catboost'"
                )
        elif self.model_type == "logistic_regression":
            self.model = LogisticRegression(**self.hyperparameters)
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")
        logger.info(f"Initialised {self.model_type} model")

    def train(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> None:
        """
        Train the model on the provided features and labels.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Target labels (n_samples,).
        """
        if self.model is None:
            self._initialize_model()
        logger.info(f"Training {self.model_type} on {X.shape[0]} samples with {X.shape[1]} features")
        self.model.fit(X, y)
        logger.info("Training complete")

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Feature matrix.

        Returns:
            Array of shape (n_samples, n_classes) with probabilities.
        """
        if self.model is None:
            raise ModelError("Model not loaded or trained")
        return self.model.predict_proba(X)

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Feature matrix.

        Returns:
            Array of predicted class labels.
        """
        if self.model is None:
            raise ModelError("Model not loaded or trained")
        return self.model.predict(X)

    def evaluate(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
    ) -> Dict[str, float]:
        """
        Evaluate model performance on given test data.

        Returns:
            Dictionary with metrics: accuracy, log_loss, (and roc_auc if binary).
        """
        y_pred_proba = self.predict_proba(X)
        y_pred = self.predict(X)
        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "log_loss": log_loss(y, y_pred_proba),
        }
        # ROC AUC only for binary classification
        if y_pred_proba.shape[1] == 2:
            metrics["roc_auc"] = roc_auc_score(y, y_pred_proba[:, 1])
        return metrics

    def save(self, path: str) -> None:
        """Save the model to disk using joblib."""
        if self.model is None:
            raise ModelError("No model to save")
        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """Load a model from disk."""
        self.model = joblib.load(path)
        logger.info(f"Model loaded from {path}")


# Type alias for callable models (e.g., LLM scorers)
ModelOrCallable = Union[PredictionModel, Callable[..., np.ndarray]]


class EnsembleModel:
    """
    Combines multiple models (ML models or callable scorers) via weighted averaging.
    Useful for hybrid ML + LLM consensus predictions.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Args:
            weights: Dictionary mapping model names to weights.
                     If None, equal weights are used (1.0 for each).
        """
        self.models: Dict[str, ModelOrCallable] = {}
        self.weights: Dict[str, float] = weights or {}
        self.model_names: List[str] = []

    def add_model(self, name: str, model: ModelOrCallable, weight: Optional[float] = None) -> None:
        """
        Add a model to the ensemble.

        Args:
            name: Unique identifier for the model.
            model: Either a PredictionModel instance or a callable that accepts
                   input X and optional additional_inputs, returning class probabilities (n_samples, n_classes).
            weight: Weight for this model. If None, uses the value from initial weights dict,
                    or 1.0 if not present.
        """
        if name in self.models:
            raise ValueError(f"Model with name '{name}' already exists in ensemble")
        self.models[name] = model
        self.model_names.append(name)
        if weight is not None:
            self.weights[name] = weight
        elif name not in self.weights:
            # Default weight = 1.0
            self.weights[name] = 1.0
        logger.info(f"Added model '{name}' with weight {self.weights[name]}")

    def predict_proba(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        additional_inputs: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """
        Compute weighted average of probabilities from all models.

        Args:
            X: Feature matrix for ML models. For callable models, they may ignore this
               if they rely on other inputs (e.g., LLM scorers).
            additional_inputs: A dictionary that can be passed to callable models
                               (e.g., raw text for LLM scorers). The callable should
                               accept (X, additional_inputs) or just additional_inputs.

        Returns:
            Array of shape (n_samples, n_classes) with averaged probabilities.
        """
        if not self.models:
            raise ModelError("Ensemble has no models")

        weighted_sum = None
        total_weight = 0.0

        for name in self.model_names:
            model = self.models[name]
            weight = self.weights[name]

            # Get predictions from this model
            if isinstance(model, PredictionModel):
                pred = model.predict_proba(X)
            elif callable(model):
                # Callable should accept (X, additional_inputs) or just X
                if additional_inputs is not None:
                    # Try with both args, fallback to just X
                    try:
                        pred = model(X, additional_inputs)
                    except TypeError:
                        pred = model(X)
                else:
                    pred = model(X)
            else:
                raise ModelError(f"Model '{name}' is neither PredictionModel nor callable")

            # Initialize weighted_sum with correct shape
            if weighted_sum is None:
                weighted_sum = pred * weight
            else:
                weighted_sum += pred * weight
            total_weight += weight

        if total_weight == 0:
            raise ModelError("Total weight is zero")
        return weighted_sum / total_weight

    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        additional_inputs: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """
        Predict class labels by taking argmax of weighted probabilities.
        """
        proba = self.predict_proba(X, additional_inputs)
        return np.argmax(proba, axis=1)