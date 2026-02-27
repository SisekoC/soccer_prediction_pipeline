# tests/conftest.py
"""Shared fixtures and mocks for all tests."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import json

from config.settings import settings
from data.data_loader import DataLoader
from features.anchors import AnchorStore
from models.model import PredictionModel
from validation.validator import Validator
from presentation.presenter import Presenter


@pytest.fixture
def sample_anchor_scores() -> pd.Series:
    """Return a sample Series of 40 anchor scores."""
    np.random.seed(42)
    scores = np.random.random(40)
    # Use the actual anchor names from settings
    # In practice, we'd load from settings, but for test we'll use dummy names
    names = [f"anchor_{i}" for i in range(40)]
    return pd.Series(scores, index=names)


@pytest.fixture
def sample_match_data() -> dict:
    """Return a sample raw data dictionary as returned by DataLoader."""
    return {
        "match_info": {
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "date": "2026-03-01",
        },
        "football_data": {
            "home_team": {"id": 57, "name": "Arsenal"},
            "away_team": {"id": 61, "name": "Chelsea"},
            "home_recent": {"matches": []},
            "away_recent": {"matches": []},
            "head_to_head": {},
        },
        "footystats": {"home": {}, "away": {}},
        "sofifa": {"home_players": [], "away_players": []},
        "sports_odds": [],
        "news_text": "Arsenal are in good form. Chelsea have injury concerns.",
    }


@pytest.fixture
def temp_cache_dir() -> Path:
    """Create a temporary directory for caching."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_data_loader(monkeypatch, temp_cache_dir):
    """Provide a DataLoader instance with mocked API calls."""
    loader = DataLoader(cache_dir=str(temp_cache_dir))
    # Monkeypatch the internal API clients if needed
    return loader


@pytest.fixture
def sample_features_df() -> pd.DataFrame:
    """Create a small feature DataFrame for training tests."""
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(100, 40), columns=[f"f{i}" for i in range(40)])
    y = pd.Series(np.random.choice([0, 1, 2], 100))
    return X, y


@pytest.fixture
def sample_prediction() -> np.ndarray:
    """Return a sample prediction array (probabilities for 3 classes)."""
    return np.array([[0.7, 0.2, 0.1]])