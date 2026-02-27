# tests/test_feature_engineering.py
"""Tests for feature engineering module."""

import pytest
import pandas as pd
import numpy as np
from features.feature_engineering import FeatureEngineering
from features.anchors import AnchorStore
from features.transformers import TeamFormTransformer, PlayerAttributesTransformer


def test_feature_engineering_initialization():
    """Test that FeatureEngineering initializes with anchor store."""
    fe = FeatureEngineering()
    assert fe.anchor_store is not None
    assert fe.bi_encoder is not None
    assert fe.cross_encoder is not None
    assert fe.top_k == 10


def test_create_features_with_empty_text(feature_eng):
    """Test that empty input returns zero scores."""
    fe = FeatureEngineering()
    scores = fe.create_features("")
    assert isinstance(scores, pd.Series)
    assert len(scores) == 40
    assert (scores == 0).all()


def test_create_features_with_sample_text(feature_eng, sample_match_data):
    """Test feature creation with sample news text."""
    fe = FeatureEngineering()
    text = sample_match_data["news_text"]
    scores = fe.create_features(text)
    assert isinstance(scores, pd.Series)
    assert len(scores) == 40
    # Scores should be between 0 and 1
    assert all(0 <= s <= 1 for s in scores)


def test_team_form_transformer(sample_match_data):
    """Test that TeamFormTransformer runs without error."""
    transformer = TeamFormTransformer()
    # Add some dummy data to sample_match_data
    sample_match_data["football_data"]["home_recent"] = {
        "matches": [
            {"homeTeam": {"id": 57}, "score": {"fullTime": {"home": 2, "away": 1}}, "winner": "HOME_TEAM"}
        ]
    }
    sample_match_data["football_data"]["away_recent"] = {
        "matches": [
            {"awayTeam": {"id": 61}, "score": {"fullTime": {"away": 0, "home": 0}}, "winner": "DRAW"}
        ]
    }
    result = transformer.transform(sample_match_data)
    assert isinstance(result, pd.Series)
    assert "home_points_last_5" in result.index