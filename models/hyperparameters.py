# models/hyperparameters.py
"""Default hyperparameter sets for supported model types."""

from typing import Dict, Any

def get_hyperparameters(model_type: str = "xgboost") -> Dict[str, Any]:
    """
    Return a dictionary of default hyperparameters.
    
    Args:
        model_type: One of 'xgboost', 'random_forest', 'logistic_regression',
                   'lightgbm', 'catboost', 'ensemble'
    """
    if model_type == "xgboost":
        return {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 0,
            "reg_alpha": 0,
            "reg_lambda": 1,
            "random_state": 42,
            "verbosity": 0,
        }
    elif model_type == "random_forest":
        return {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "random_state": 42,
            "n_jobs": -1,
        }
    elif model_type == "lightgbm":
        return {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "num_leaves": 31,
            "random_state": 42,
        }
    elif model_type == "catboost":
        return {
            "iterations": 100,
            "depth": 6,
            "learning_rate": 0.1,
            "random_state": 42,
            "verbose": False,
        }
    elif model_type == "logistic_regression":
        return {
            "C": 1.0,
            "penalty": "l2",
            "solver": "lbfgs",
            "max_iter": 1000,
            "random_state": 42,
        }
    elif model_type == "ensemble":
        # Placeholder for hybrid ensemble
        return {
            "xgboost_weight": 0.6,
            "llm_weight": 0.4,
        }
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")