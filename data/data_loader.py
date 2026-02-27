"""
Data loader that aggregates data from multiple soccer APIs.
Implements caching and fallback logic.
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from ..config.settings import settings
from ..utils.logger import get_logger
from ..utils.exceptions import DataLoadError
from .api_client import (
    FootballDataClient,
    FootyStatsClient,
    SoFIFAClient,
    SportsOddsClient,
    BBCScraperClient
)
from bs4 import BeautifulSoup

logger = get_logger(__name__)


class DataLoader:
    """
    Aggregates data from multiple soccer APIs for a given match.
    Uses a cache to avoid repeated calls.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir or settings.cache_dir) / "api_responses"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize API clients (only if API keys are available)
        self.football_data = None
        self.footystats = None
        self.sofifa = None
        self.sports_odds = None
        self.bbc_scraper = None
        
        if settings.football_data_api_key:
            self.football_data = FootballDataClient()
        else:
            logger.warning("Football-Data.org API key not set.")
        
        if settings.footystats_api_key:
            self.footystats = FootyStatsClient()
        else:
            logger.warning("Footystats API key not set.")
        
        if settings.sports_odds_api_key:
            self.sports_odds = SportsOddsClient()
        else:
            logger.warning("Sports Odds API key not set.")
        
        # SoFIFA/CRSet is free, no key needed
        self.sofifa = SoFIFAClient()
        self.bbc_scraper = BBCScraperClient()
    
    def _cache_key(self, source: str, *identifiers) -> Path:
        """Generate a cache file path."""
        key = hashlib.md5("_".join(map(str, identifiers)).encode()).hexdigest()
        return self.cache_dir / f"{source}_{key}.json"
    
    def _load_from_cache(self, cache_path: Path, max_age_hours: int = 24) -> Optional[Dict]:
        """Load cached data if it exists and is not too old."""
        if cache_path.exists():
            with open(cache_path, 'r') as f:
                data = json.load(f)
            # Check age
            cached_time = data.get('_cached_at', 0)
            if time.time() - cached_time < max_age_hours * 3600:
                return data.get('data')
            else:
                logger.debug(f"Cache expired for {cache_path}")
        return None
    
    def _save_to_cache(self, cache_path: Path, data: Dict):
        """Save data to cache with timestamp."""
        to_save = {
            '_cached_at': time.time(),
            'data': data
        }
        with open(cache_path, 'w') as f:
            json.dump(to_save, f)
    
    def _fetch_from_source(self, source_name: str, fetch_func, cache_identifiers: tuple, **kwargs) -> Optional[Dict]:
        """
        Generic fetch with cache check and save.
        """
        cache_path = self._cache_key(source_name, *cache_identifiers)
        
        # Try cache
        cached = self._load_from_cache(cache_path)
        if cached is not None:
            logger.info(f"Using cached {source_name} data")
            return cached
        
        # Fetch
        try:
            data = fetch_func(**kwargs)
            if data:
                self._save_to_cache(cache_path, data)
                logger.info(f"Fetched and cached {source_name} data")
            return data
        except Exception as e:
            logger.error(f"Failed to fetch from {source_name}: {e}")
            return None
    
    def load_match_data(self, home_team: str, away_team: str, match_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Main method: load all data for a match.
        Returns a dictionary with keys for each data source.
        """
        result = {
            "match_info": {
                "home_team": home_team,
                "away_team": away_team,
                "date": match_date
            },
            "football_data": {},
            "footystats": {},
            "sofifa": {},
            "sports_odds": {},
            "news_text": None
        }
        
        # We need team IDs for some APIs. This is a simplified approach.
        # In production, you'd maintain a mapping or search by name.
        home_id = self._get_team_id(home_team)
        away_id = self._get_team_id(away_team)
        
        # 1. Football-Data.org
        if self.football_data and home_id and away_id:
            # Get team info
            home_info = self._fetch_from_source(
                "football_data_team",
                self.football_data.get_team,
                (home_id,),
                team_id=home_id
            )
            away_info = self._fetch_from_source(
                "football_data_team",
                self.football_data.get_team,
                (away_id,),
                team_id=away_id
            )
            
            # Get recent matches
            home_matches = self._fetch_from_source(
                "football_data_matches",
                self.football_data.get_matches,
                (home_id,),
                team_id=home_id,
                limit=10
            )
            away_matches = self._fetch_from_source(
                "football_data_matches",
                self.football_data.get_matches,
                (away_id,),
                team_id=away_id,
                limit=10
            )
            
            # Head-to-head
            h2h = self._fetch_from_source(
                "football_data_h2h",
                self.football_data.get_head_to_head,
                (home_id, away_id),
                home_id=home_id,
                away_id=away_id,
                limit=5
            )
            
            result["football_data"] = {
                "home_team": home_info,
                "away_team": away_info,
                "home_recent": home_matches,
                "away_recent": away_matches,
                "head_to_head": h2h
            }
        
        # 2. Footystats
        if self.footystats:
            # For demo, we'll use team name search (Footystats has search by name)
            # Ideally you'd map team names to Footystats IDs.
            home_stats = self._fetch_from_source(
                "footystats_team",
                self.footystats.get_team_season_stats,
                (home_team, "2025"),
                team_id=home_team,  # simplified: pass name, API may handle
                season="2025"
            )
            away_stats = self._fetch_from_source(
                "footystats_team",
                self.footystats.get_team_season_stats,
                (away_team, "2025"),
                team_id=away_team,
                season="2025"
            )
            result["footystats"] = {
                "home": home_stats,
                "away": away_stats
            }
        
        # 3. SoFIFA / player attributes
        # Get team players (approximate)
        home_players = self._fetch_from_source(
            "sofifa_team",
            self.sofifa.get_team_players,
            (home_team,),
            team_name=home_team
        )
        away_players = self._fetch_from_source(
            "sofifa_team",
            self.sofifa.get_team_players,
            (away_team,),
            team_name=away_team
        )
        result["sofifa"] = {
            "home_players": home_players,
            "away_players": away_players
        }
        
        # 4. Sports Odds
        if self.sports_odds:
            # Assume sport = soccer_epl; in practice determine league from match info
            odds = self._fetch_from_source(
                "sports_odds",
                self.sports_odds.get_odds,
                (home_team, away_team),
                sport="soccer_epl"
            )
            result["sports_odds"] = odds
        
        # 5. News scraping (BBC)
        # Normalize team names for URL slug
        home_slug = home_team.lower().replace(' ', '-')
        away_slug = away_team.lower().replace(' ', '-')
        
        home_html = self._fetch_from_source(
            "bbc_team",
            self.bbc_scraper.get_team_page,
            (home_slug,),
            team_slug=home_slug
        )
        away_html = self._fetch_from_source(
            "bbc_team",
            self.bbc_scraper.get_team_page,
            (away_slug,),
            team_slug=away_slug
        )
        
        # Extract text from HTML
        news_text = self._extract_news_text(home_html, away_html)
        result["news_text"] = news_text
        
        # Add a small delay to be polite
        time.sleep(2)
        
        return result
    
    def _get_team_id(self, team_name: str) -> Optional[int]:
        """
        Map team name to Football-Data.org team ID.
        In production, you'd have a local mapping or search endpoint.
        This is a placeholder.
        """
        # Dummy mapping for common teams
        mapping = {
            "Manchester United": 66,
            "Liverpool": 64,
            "Arsenal": 57,
            "Chelsea": 61,
            "Manchester City": 65,
            "Tottenham Hotspur": 73,
            # ... add more
        }
        return mapping.get(team_name)
    
    def _extract_news_text(self, home_html: Optional[str], away_html: Optional[str]) -> str:
        """Extract plain text from BBC HTML pages."""
        text_parts = []
        for html in [home_html, away_html]:
            if html:
                try:
                    soup = BeautifulSoup(html, 'html.parser')
                    main = soup.find('main')
                    if main:
                        paras = main.find_all('p')
                        text = ' '.join(p.get_text() for p in paras)
                        text_parts.append(text)
                except Exception as e:
                    logger.error(f"Error parsing HTML: {e}")
        return ' '.join(text_parts) if text_parts else ""