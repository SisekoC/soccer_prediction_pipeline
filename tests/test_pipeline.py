# tests/test_pipeline.py
"""Tests for the pipeline orchestrator."""

import pytest
from pipeline.orchestrator import PredictionPipeline
from data.data_loader import DataLoader
from features.feature_engineering import FeatureEngineering
from models.model import PredictionModel, EnsembleModel
from validation.validator import Validator
from presentation.presenter import Presenter


def test_pipeline_initialization():
    """Test that pipeline initializes with default components."""
    pipeline = PredictionPipeline()
    assert pipeline.data_loader is not None
    assert pipeline.validator is not None
    assert pipeline.feature_eng is not None
    assert pipeline.result_validator is not None
    assert pipeline.presenter is not None


def test_pipeline_run_without_model(mocker, sample_match_data):
    """Test pipeline run when no model is provided."""
    # Mock DataLoader to return sample data
    mocker.patch.object(DataLoader, "load_match_data", return_value=sample_match_data)

    pipeline = PredictionPipeline()
    result = pipeline.run("Arsenal", "Chelsea", "2026-03-01")

    assert result["match_info"]["home_team"] == "Arsenal"
    assert result["features"] is not None
    assert result["prediction"] is None


def test_pipeline_run_with_model(mocker, sample_match_data, sample_features_df):
    """Test pipeline run with a model."""
    mocker.patch.object(DataLoader, "load_match_data", return_value=sample_match_data)
    # Mock feature engineering to return dummy features
    mock_features = sample_features_df[0].iloc[0]
    mocker.patch.object(FeatureEngineering, "create_features", return_value=mock_features)

    model = PredictionModel(model_type="logistic_regression")
    pipeline = PredictionPipeline(model=model)
    result = pipeline.run("Arsenal", "Chelsea", "2026-03-01")

    assert result["prediction"] is not None


def test_pipeline_batch_run(mocker, sample_match_data):
    """Test batch processing."""
    mocker.patch.object(DataLoader, "load_match_data", return_value=sample_match_data)
    mocker.patch.object(FeatureEngineering, "create_features", return_value=pd.Series(np.random.random(40)))

    pipeline = PredictionPipeline()
    matches = [("Arsenal", "Chelsea", "2026-03-01"), ("Liverpool", "ManCity", "2026-03-02")]
    results = pipeline.run_batch(matches, delay=0)

    assert len(results) == 2