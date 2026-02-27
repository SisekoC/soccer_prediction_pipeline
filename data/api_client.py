"""
API clients for various soccer data sources.
All clients use free tiers and include rate limiting awareness.
"""

import time
import requests
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ..config.settings import settings
from ..utils.exceptions import APIError, DataLoadError
from ..utils.logger import get_logger

logger = get_logger(__name__)


class BaseAPIClient(ABC):
    """Base class for all API clients with common retry logic and session handling."""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout: int = 10):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": settings.user_agent})
        if api_key:
            self._set_auth()
    
    @abstractmethod
    def _set_auth(self):
        """Set authentication headers or parameters."""
        pass
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(requests.exceptions.RequestException),
        after=lambda retry_state: logger.warning(f"Retrying API call (attempt {retry_state.attempt_number})")
    )
    def _get(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Make a GET request with retries."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        try:
            resp = self.session.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            if resp.status_code == 429:
                # Rate limit - wait and retry (handled by tenacity)
                logger.warning("Rate limited, will retry...")
                raise
            logger.error(f"HTTP error {resp.status_code} from {url}: {resp.text[:200]}")
            raise APIError(f"HTTP {resp.status_code} from {url}") from e
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {url}: {e}")
            raise APIError(f"Request failed: {e}") from e
    
    def get(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Public get method with logging."""
        logger.debug(f"Calling {self.__class__.__name__} endpoint: {endpoint}")
        return self._get(endpoint, params)


class FootballDataClient(BaseAPIClient):
    """
    Client for football-data.org (free tier: 10 requests/min, limited leagues).
    """
    
    def __init__(self):
        super().__init__(
            base_url="https://api.football-data.org/v4",
            api_key=settings.football_data_api_key
        )
    
    def _set_auth(self):
        self.session.headers.update({"X-Auth-Token": self.api_key})
    
    def get_team(self, team_id: int) -> Dict[str, Any]:
        """Get team information."""
        return self.get(f"teams/{team_id}")
    
    def get_matches(self, team_id: int, limit: int = 10) -> Dict[str, Any]:
        """Get recent matches for a team."""
        return self.get(f"teams/{team_id}/matches", params={"limit": limit, "status": "FINISHED"})
    
    def get_head_to_head(self, home_id: int, away_id: int, limit: int = 5) -> Dict[str, Any]:
        """Get head-to-head matches."""
        return self.get(f"teams/{home_id}/matches", params={"opponent": away_id, "limit": limit, "status": "FINISHED"})


class FootyStatsClient(BaseAPIClient):
    """
    Client for Footystats API (free tier includes Premier League data).
    Docs: https://www.footystats.org/api
    """
    
    def __init__(self):
        super().__init__(
            base_url="https://api.footystats.org",
            api_key=settings.footystats_api_key
        )
    
    def _set_auth(self):
        # Footystats uses query parameter 'key'
        self.session.params = {"key": self.api_key}
    
    def get_team_season_stats(self, team_id: int, season: str) -> Dict[str, Any]:
        """Get advanced stats for a team in a given season."""
        return self.get("team", params={"team_id": team_id, "season": season})
    
    def get_league_matches(self, league_id: int, season: str) -> Dict[str, Any]:
        """Get matches for a league season."""
        return self.get("matches", params={"league_id": league_id, "season": season})
    
    def get_team_fixtures(self, team_id: int) -> Dict[str, Any]:
        """Get upcoming fixtures."""
        return self.get("fixtures", params={"team_id": team_id})


class SoFIFAClient(BaseAPIClient):
    """
    Client for SoFIFA / CRSet API (free, rate-limited).
    Provides player attributes.
    """
    
    def __init__(self):
        # Using CRSet API (https://crset.cyclic.app) as a free alternative
        super().__init__(
            base_url="https://crset.cyclic.app/api/v1",
            api_key=None  # No API key needed for this free tier
        )
    
    def _set_auth(self):
        pass  # No auth required
    
    def search_players(self, name: str) -> Dict[str, Any]:
        """Search for a player by name."""
        return self.get("players/search", params={"name": name})
    
    def get_team_players(self, team_name: str) -> Dict[str, Any]:
        """Get players of a team (fuzzy search)."""
        return self.get("players/team", params={"team": team_name})
    
    def get_player(self, player_id: int) -> Dict[str, Any]:
        """Get player details by ID."""
        return self.get(f"players/{player_id}")


class SportsOddsClient(BaseAPIClient):
    """
    Client for The Odds API (free tier: 500 requests/month).
    Docs: https://the-odds-api.com/
    """
    
    def __init__(self):
        super().__init__(
            base_url="https://api.the-odds-api.com/v4",
            api_key=settings.sports_odds_api_key
        )
    
    def _set_auth(self):
        # API key passed as query parameter
        self.session.params = {"apiKey": self.api_key}
    
    def get_sports(self) -> Dict[str, Any]:
        """Get list of available sports."""
        return self.get("sports")
    
    def get_odds(self, sport: str = "soccer_epl", regions: str = "uk", markets: str = "h2h,spreads") -> Dict[str, Any]:
        """
        Get odds for a sport.
        Returns list of matches with odds from bookmakers.
        """
        params = {
            "regions": regions,
            "markets": markets,
            "oddsFormat": "decimal"
        }
        return self.get(f"sports/{sport}/odds", params=params)
    
    def get_scores(self, sport: str = "soccer_epl", days_from: int = 1) -> Dict[str, Any]:
        """Get recent scores."""
        return self.get(f"sports/{sport}/scores", params={"daysFrom": days_from})


class BBCScraperClient(BaseAPIClient):
    """
    Scraper for BBC Sport team pages (no API key, requires respectful crawling).
    Inherits from BaseAPIClient to reuse retry logic but overrides _get to return HTML.
    """
    
    def __init__(self):
        super().__init__(
            base_url="https://www.bbc.com/sport/football/teams",
            api_key=None
        )
        self.session.headers.update({
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
        })
    
    def _set_auth(self):
        pass
    
    def _get(self, endpoint: str, params: Optional[Dict] = None) -> str:
        """Override to return HTML text instead of JSON."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        try:
            resp = self.session.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            return resp.text
        except requests.exceptions.RequestException as e:
            logger.error(f"Scraper request failed for {url}: {e}")
            raise APIError(f"Scraper failed: {e}") from e
    
    def get_team_page(self, team_slug: str) -> str:
        """Get HTML of team page."""
        return self._get(team_slug)