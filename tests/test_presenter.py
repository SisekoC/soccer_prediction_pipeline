# tests/test_presenter.py
"""Tests for the presenter module."""

import pytest
import pandas as pd
import numpy as np
from presentation.presenter import Presenter


def test_presenter_initialization():
    """Test presenter initializes with default format."""
    p = Presenter()
    assert p.output_format == "json"
    assert p.output_file is None


def test_json_format(sample_anchor_scores):
    """Test JSON output format."""
    p = Presenter(output_format="json")
    output = p.format(
        home_team="Arsenal",
        away_team="Chelsea",
        match_date="2026-03-01",
        features=sample_anchor_scores,
        prediction=np.array([[0.7, 0.2, 0.1]]),
        actual="H",
    )
    import json
    data = json.loads(output)
    assert data["match"]["home_team"] == "Arsenal"
    assert "features" in data
    assert "prediction" in data
    assert data["actual"] == "H"


def test_human_format(sample_anchor_scores):
    """Test human-readable output format."""
    p = Presenter(output_format="human")
    output = p.format(
        home_team="Arsenal",
        away_team="Chelsea",
        match_date="2026-03-01",
        features=sample_anchor_scores,
        prediction=np.array([[0.7, 0.2, 0.1]]),
    )
    assert "Arsenal vs Chelsea" in output
    assert "Top 5 influencing factors" in output
    assert "Prediction: Home 70.0%" in output


def test_csv_format(sample_anchor_scores):
    """Test CSV output format for single match."""
    p = Presenter(output_format="csv")
    output = p.format(
        home_team="Arsenal",
        away_team="Chelsea",
        match_date="2026-03-01",
        features=sample_anchor_scores,
        prediction=np.array([[0.7, 0.2, 0.1]]),
        actual="H",
    )
    assert "home_team,away_team,date,actual" in output
    assert "Arsenal,Chelsea,2026-03-01,H" in output


def test_markdown_format(sample_anchor_scores):
    """Test Markdown output format."""
    p = Presenter(output_format="markdown")
    output = p.format(
        home_team="Arsenal",
        away_team="Chelsea",
        match_date="2026-03-01",
        features=sample_anchor_scores,
        prediction=np.array([[0.7, 0.2, 0.1]]),
        actual="H",
    )
    assert "# Match: Arsenal vs Chelsea" in output
    assert "## Semantic Anchor Scores" in output
    assert "| Anchor | Score |" in output


def test_batch_csv_format(sample_anchor_scores):
    """Test batch CSV output."""
    p = Presenter(output_format="csv")
    results = [
        {
            "match_info": {"home_team": "Arsenal", "away_team": "Chelsea", "date": "2026-03-01"},
            "features": sample_anchor_scores,
            "prediction": np.array([[0.7, 0.2, 0.1]]),
            "actual": "H",
        },
        {
            "match_info": {"home_team": "Liverpool", "away_team": "ManCity", "date": "2026-03-02"},
            "features": sample_anchor_scores,
            "prediction": np.array([[0.3, 0.3, 0.4]]),
            "actual": "A",
        },
    ]
    output = p.format_batch(results)
    # Should have two rows plus header
    lines = output.strip().split('\n')
    assert len(lines) == 3
    assert "Arsenal,Chelsea" in lines[1]
    assert "Liverpool,ManCity" in lines[2]


def test_file_output(tmp_path, sample_anchor_scores):
    """Test writing to file."""
    output_file = tmp_path / "output.json"
    p = Presenter(output_format="json", output_file=str(output_file))
    result = p.format(
        home_team="Arsenal",
        away_team="Chelsea",
        match_date="2026-03-01",
        features=sample_anchor_scores,
    )
    assert result is None  # no return when writing to file
    assert output_file.exists()
    with open(output_file) as f:
        data = json.load(f)
    assert data["match"]["home_team"] == "Arsenal"