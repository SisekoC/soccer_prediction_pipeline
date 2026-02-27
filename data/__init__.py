# data/__init__.py
"""Data acquisition and validation modules for soccer data sources."""

from .api_client import (
    FootballDataClient,
    FootyStatsClient,
    SoFIFAClient,
    SportsOddsClient,
    BBCScraperClient,
    BaseAPIClient,
)
from .data_loader import DataLoader
from .data_validator import DataValidator

__all__ = [
    "DataLoader",
    "DataValidator",
    "FootballDataClient",
    "FootyStatsClient",
    "SoFIFAClient",
    "SportsOddsClient",
    "BBCScraperClient",
    "BaseAPIClient",
]