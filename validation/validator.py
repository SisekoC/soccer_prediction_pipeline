# validation/validator.py
"""Prediction validation and performance tracking."""

import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, roc_auc_score

from ..utils.logger import get_logger

logger = get_logger(__name__)


class Validator:
    """
    Validates model predictions against actual outcomes.
    Can handle single predictions or accumulate history for aggregate metrics.
    Supports probability predictions (preferred) or hard class predictions.
    """

    def __init__(self, history_path: Optional[str] = None):
        """
        Args:
            history_path: Optional path to JSON file for storing validation history.
                          If provided, history is loaded and new entries are appended.
        """
        self.history_path = Path(history_path) if history_path else None
        self.history: List[Dict[str, Any]] = []
        if self.history_path and self.history_path.exists():
            self._load_history()

    def _load_history(self) -> None:
        """Load validation history from JSON file."""
        try:
            with open(self.history_path, 'r') as f:
                self.history = json.load(f)
            logger.info(f"Loaded {len(self.history)} validation records from {self.history_path}")
        except Exception as e:
            logger.error(f"Failed to load validation history: {e}")
            self.history = []

    def _save_history(self) -> None:
        """Append current history to JSON file."""
        if not self.history_path:
            return
        try:
            # Ensure directory exists
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.history_path, 'w') as f:
                json.dump(self.history, f, indent=2)
            logger.debug(f"Validation history saved to {self.history_path}")
        except Exception as e:
            logger.error(f"Failed to save validation history: {e}")

    def validate(
        self,
        prediction: Union[np.ndarray, List[float], float],
        actual: Union[str, int],
        match_id: Optional[str] = None,
        predicted_class: Optional[Union[str, int]] = None,
        class_mapping: Optional[Dict[int, str]] = None,
        store: bool = True,
    ) -> Dict[str, Any]:
        """
        Validate a single prediction against the actual outcome.

        Args:
            prediction: Model output. Can be:
                - Probability array of shape (n_classes,) or (1, n_classes)
                - Single probability for binary case (if class mapping provided)
                - Scalar (treated as predicted class)
            actual: Actual outcome. Can be string ('H', 'D', 'A') or integer class index.
            match_id: Optional identifier for this prediction (e.g., match ID).
            predicted_class: If prediction is probabilities, you can optionally provide
                             the predicted class (otherwise derived by argmax).
            class_mapping: Mapping from integer indices to outcome strings.
                           Example: {0: 'H', 1: 'D', 2: 'A'}.
            store: Whether to store this validation in history.

        Returns:
            Dictionary with validation results:
                - correct: boolean
                - accuracy_contribution: 1 if correct else 0
                - predicted_class: class label
                - actual_class: class label
                - confidence: max probability (if probabilities provided)
                - log_loss_contribution: log loss value (if probabilities provided)
                - brier_score_contribution: Brier score (if probabilities provided)
                - timestamp: ISO datetime
                - match_id: provided ID
        """
        # Convert inputs to standard form
        timestamp = datetime.now().isoformat()

        # Handle probability array
        if isinstance(prediction, (np.ndarray, list)):
            pred_array = np.array(prediction).flatten()
            if pred_array.ndim == 1:
                # Single sample, multiple classes
                n_classes = len(pred_array)
                predicted_class_idx = int(np.argmax(pred_array))
                confidence = float(pred_array[predicted_class_idx])
                probas = pred_array
            else:
                raise ValueError("Prediction array must be 1D (single sample)")

            # Map predicted class index to label if mapping provided
            if class_mapping:
                predicted_class_label = class_mapping.get(predicted_class_idx, str(predicted_class_idx))
            else:
                predicted_class_label = str(predicted_class_idx)
        else:
            # Scalar prediction – treat as class
            predicted_class_label = str(prediction)
            confidence = None
            probas = None

        # Determine actual class label
        actual_label = str(actual)

        # If probabilities and class mapping, compute actual class index for metrics
        actual_class_idx = None
        if class_mapping and probas is not None:
            # Reverse mapping: label -> index
            inv_mapping = {v: k for k, v in class_mapping.items()}
            actual_class_idx = inv_mapping.get(actual_label, None)

        # Compute metrics
        correct = (predicted_class_label == actual_label)

        result = {
            "match_id": match_id,
            "timestamp": timestamp,
            "actual": actual_label,
            "predicted": predicted_class_label,
            "correct": correct,
            "accuracy_contribution": 1.0 if correct else 0.0,
        }

        # Add probability‑based metrics if available
        if probas is not None and actual_class_idx is not None:
            # Log loss contribution for this sample (negative log prob of true class)
            # To avoid log(0), clip probabilities
            eps = 1e-15
            prob_true = np.clip(probas[actual_class_idx], eps, 1 - eps)
            log_loss_contrib = -np.log(prob_true)
            result["log_loss_contribution"] = float(log_loss_contrib)

            # Brier score contribution (sum of squared errors)
            # For multi-class, it's sum over classes (one-hot encoded)
            one_hot = np.zeros_like(probas)
            one_hot[actual_class_idx] = 1.0
            brier_contrib = np.sum((probas - one_hot) ** 2)
            result["brier_score_contribution"] = float(brier_contrib)

            # ROC AUC can't be computed per sample, so not included here
            result["confidence"] = confidence

        # Store in history if requested
        if store:
            self.history.append(result)
            self._save_history()

        return result

    def validate_batch(
        self,
        predictions: Union[np.ndarray, List],
        actuals: Union[List, np.ndarray, pd.Series],
        match_ids: Optional[List[str]] = None,
        class_mapping: Optional[Dict[int, str]] = None,
        store: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Validate multiple predictions.

        Args:
            predictions: Array of shape (n_samples, n_classes) or (n_samples,) of class predictions.
            actuals: List/array of actual outcomes.
            match_ids: Optional list of identifiers.
            class_mapping: Mapping from class indices to labels.
            store: Whether to store all validations.

        Returns:
            List of result dictionaries per sample.
        """
        n = len(actuals)
        if isinstance(predictions, np.ndarray) and predictions.ndim == 2:
            # Probabilities for each sample
            proba_mode = True
        else:
            proba_mode = False

        results = []
        for i in range(n):
            pred = predictions[i] if proba_mode else predictions[i] if isinstance(predictions, list) else predictions
            actual = actuals[i]
            match_id = match_ids[i] if match_ids else None
            result = self.validate(
                prediction=pred,
                actual=actual,
                match_id=match_id,
                class_mapping=class_mapping,
                store=False,  # Store after loop to avoid multiple file writes
            )
            results.append(result)

        if store:
            self.history.extend(results)
            self._save_history()

        return results

    def get_aggregate_metrics(self, history_subset: Optional[List[Dict]] = None) -> Dict[str, float]:
        """
        Compute aggregate metrics over validation history.

        Args:
            history_subset: Optional subset of history to use; if None, use all history.

        Returns:
            Dictionary with metrics: accuracy, avg_log_loss, avg_brier_score, etc.
        """
        data = history_subset if history_subset is not None else self.history
        if not data:
            logger.warning("No validation data for aggregate metrics")
            return {}

        n = len(data)
        accuracy = np.mean([d["accuracy_contribution"] for d in data])

        metrics = {
            "accuracy": accuracy,
            "n_validations": n,
        }

        # Log loss (if available)
        if "log_loss_contribution" in data[0]:
            avg_log_loss = np.mean([d["log_loss_contribution"] for d in data])
            metrics["avg_log_loss"] = avg_log_loss

        # Brier score (if available)
        if "brier_score_contribution" in data[0]:
            avg_brier = np.mean([d["brier_score_contribution"] for d in data])
            metrics["avg_brier_score"] = avg_brier

        return metrics

    def clear_history(self) -> None:
        """Clear in‑memory history and delete history file if exists."""
        self.history = []
        if self.history_path and self.history_path.exists():
            self.history_path.unlink()
        logger.info("Validation history cleared")