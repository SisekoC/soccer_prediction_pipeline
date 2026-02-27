# tests/test_data_loader.py
"""Tests for the data loader module."""

import pytest
from pathlib import Path
from data.data_loader import DataLoader
from data.api_client import (
    FootballDataClient,
    FootyStatsClient,
    SoFIFAClient,
    SportsOddsClient,
    BBCScraperClient,
)


def test_data_loader_initialization(temp_cache_dir):
    """Test that DataLoader initializes with cache directory."""
    loader = DataLoader(cache_dir=str(temp_cache_dir))
    assert loader.cache_dir.exists()
    assert loader.cache_dir.name == "api_responses"


def test_cache_key_generation(mock_data_loader):
    """Test that cache keys are generated consistently."""
    key1 = mock_data_loader._cache_key("test", "arg1", "arg2")
    key2 = mock_data_loader._cache_key("test", "arg1", "arg2")
    assert key1 == key2
    assert key1.name.endswith(".json")
    assert "test_" in key1.name


def test_load_match_data_missing_api_keys(monkeypatch, temp_cache_dir):
    """Test that loader handles missing API keys gracefully."""
    # Unset API keys
    monkeypatch.setattr("config.settings.football_data_api_key", None)
    monkeypatch.setattr("config.settings.footystats_api_key", None)
    monkeypatch.setattr("config.settings.sports_odds_api_key", None)

    loader = DataLoader(cache_dir=str(temp_cache_dir))
    data = loader.load_match_data("Arsenal", "Chelsea", "2026-03-01")
    assert data is not None
    assert "football_data" in data
    assert "footystats" in data
    assert "sofifa" in data
    assert "sports_odds" in data
    assert "news_text" in data


def test_extract_news_text(mock_data_loader):
    """Test that news text extraction works."""
    html = "<html><body><main><p>Team news: player fit.</p></main></body></html>"
    text = mock_data_loader._extract_news_text(html, "")
    assert "Team news: player fit." in text