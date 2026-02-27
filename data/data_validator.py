"""
Data validation module for raw API responses.
Checks structure, types, and ranges, and fills missing optional fields with defaults.
"""

from typing import Any, Dict, Optional
import logging

from ..utils.exceptions import DataValidationError
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DataValidator:
    """
    Validates and cleans the raw data dictionary from DataLoader.
    Each validation method focuses on one data source.
    """

    def __init__(self, strict_mode: bool = False):
        """
        Args:
            strict_mode: If True, raise exceptions on non‑critical validation failures.
                         If False, only log warnings and fill defaults.
        """
        self.strict_mode = strict_mode

    def validate(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point: validates all sections of the raw data.
        Returns a cleaned copy of the data (with defaults applied).
        """
        # Work on a copy to avoid mutating the original
        data = raw_data.copy()

        # Validate top-level structure
        self._validate_top_level(data)

        # Validate each section
        data['football_data'] = self._validate_football_data(data.get('football_data', {}))
        data['footystats'] = self._validate_footystats(data.get('footystats', {}))
        data['sofifa'] = self._validate_sofifa(data.get('sofifa', {}))
        data['sports_odds'] = self._validate_sports_odds(data.get('sports_odds', {}))
        data['news_text'] = self._validate_news_text(data.get('news_text', ''))

        # Final sanity check
        self._post_validation_checks(data)

        return data

    def _validate_top_level(self, data: Dict[str, Any]) -> None:
        """Ensure top-level keys exist."""
        required_keys = ['match_info', 'football_data', 'footystats', 'sofifa', 'sports_odds', 'news_text']
        for key in required_keys:
            if key not in data:
                if self.strict_mode:
                    raise DataValidationError(f"Missing top-level key: {key}")
                else:
                    logger.warning(f"Top-level key '{key}' missing; adding empty placeholder.")
                    data[key] = {} if key != 'news_text' else ''

    def _validate_football_data(self, fb_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate Football-Data.org section."""
        validated = fb_data.copy()
        # Expected keys: home_team, away_team, home_recent, away_recent, head_to_head
        expected = ['home_team', 'away_team', 'home_recent', 'away_recent', 'head_to_head']
        for key in expected:
            if key not in validated:
                logger.warning(f"football_data missing '{key}'; adding empty dict.")
                validated[key] = {}
        # Further nested validation could be added (e.g., ensure home_team has 'id')
        return validated

    def _validate_footystats(self, fs_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate Footystats section."""
        validated = fs_data.copy()
        # Expected: 'home', 'away' (each containing stats dict)
        if 'home' not in validated:
            logger.warning("footystats missing 'home'; adding empty dict.")
            validated['home'] = {}
        if 'away' not in validated:
            logger.warning("footystats missing 'away'; adding empty dict.")
            validated['away'] = {}
        # Optionally check for common stats like 'xG_for'
        return validated

    def _validate_sofifa(self, sofifa_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate SoFIFA section."""
        validated = sofifa_data.copy()
        if 'home_players' not in validated:
            logger.warning("sofifa missing 'home_players'; adding empty list.")
            validated['home_players'] = []
        if 'away_players' not in validated:
            logger.warning("sofifa missing 'away_players'; adding empty list.")
            validated['away_players'] = []
        # If players list exists, validate each player has numeric attributes
        for side in ['home_players', 'away_players']:
            if isinstance(validated[side], dict) and 'players' in validated[side]:
                # Some APIs return {"players": [...]}
                players = validated[side]['players']
            elif isinstance(validated[side], list):
                players = validated[side]
            else:
                players = []
            for player in players:
                self._validate_player(player)
        return validated

    def _validate_player(self, player: Dict[str, Any]) -> None:
        """Check individual player attributes."""
        # Ensure required numeric fields exist; if missing, fill with default 50
        for attr in ['pace', 'stamina', 'strength']:
            if attr not in player:
                logger.debug(f"Player {player.get('name', 'unknown')} missing '{attr}'; defaulting to 50.")
                player[attr] = 50
            else:
                # Range check
                val = player[attr]
                if not isinstance(val, (int, float)) or val < 0 or val > 100:
                    logger.warning(f"Player {player.get('name', 'unknown')} has invalid {attr}={val}; defaulting to 50.")
                    player[attr] = 50

    def _validate_sports_odds(self, odds_data: Any) -> Any:
        """Validate sports odds section (can be list or dict)."""
        if isinstance(odds_data, list):
            # Validate each match in list
            for match in odds_data:
                self._validate_odds_match(match)
            return odds_data
        elif isinstance(odds_data, dict):
            self._validate_odds_match(odds_data)
            return odds_data
        else:
            logger.warning(f"sports_odds is unexpected type {type(odds_data)}; returning empty list.")
            return []

    def _validate_odds_match(self, match: Dict[str, Any]) -> None:
        """Validate a single odds match object."""
        if 'bookmakers' not in match or not match['bookmakers']:
            logger.debug("Odds match missing bookmakers; cannot compute probabilities.")
            # No further validation needed; downstream transformers will handle missing odds
        else:
            # Could check that prices are positive floats
            for bm in match['bookmakers']:
                for market in bm.get('markets', []):
                    for outcome in market.get('outcomes', []):
                        price = outcome.get('price')
                        if price is not None and (not isinstance(price, (int, float)) or price <= 0):
                            logger.warning(f"Invalid odds price {price}; may affect probability calculation.")

    def _validate_news_text(self, text: Any) -> str:
        """Ensure news_text is a string."""
        if not isinstance(text, str):
            logger.warning(f"news_text is not a string (type {type(text)}); converting to empty string.")
            return ""
        return text

    def _post_validation_checks(self, data: Dict[str, Any]) -> None:
        """
        Final cross‑field checks, e.g., ensure team names match across sections.
        """
        match_info = data.get('match_info', {})
        home_team = match_info.get('home_team')
        away_team = match_info.get('away_team')
        if not home_team or not away_team:
            logger.error("match_info missing home_team or away_team")
            if self.strict_mode:
                raise DataValidationError("match_info incomplete")
        # Additional cross-checks could be added here