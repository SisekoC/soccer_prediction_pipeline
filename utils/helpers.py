# utils/helpers.py
"""Miscellaneous helper functions used across the pipeline."""

import re
from datetime import datetime
from typing import Optional


def normalize_team_name(name: str) -> str:
    """
    Convert team name to a slug suitable for URLs or filenames.
    Example: "Manchester United" -> "manchester-united"
    """
    # Lowercase, replace spaces with hyphens, remove special characters
    name = name.lower().strip()
    name = re.sub(r"[^\w\s-]", "", name)  # Remove punctuation
    name = re.sub(r"[\s]+", "-", name)    # Replace spaces with hyphens
    return name


def parse_date(date_str: Optional[str]) -> Optional[str]:
    """
    Attempt to parse a date string into YYYY-MM-DD format.
    If parsing fails or input is None, return None.
    """
    if not date_str:
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y%m%d"):
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division, returning default if denominator is zero."""
    if denominator == 0:
        return default
    return numerator / denominator


def chunk_list(lst, chunk_size):
    """Yield successive chunks from a list."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]