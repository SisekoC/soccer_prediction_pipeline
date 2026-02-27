from pydantic_settings import BaseSettings
from typing import Dict, Optional

class Settings(BaseSettings):
    # API keys
    football_data_api_key: Optional[str] = None
    footystats_api_key: Optional[str] = None
    sports_odds_api_key: Optional[str] = None
    
    # Data storage
    cache_dir: str = "./data/cache/"          # Default local path
    
    # Other settings
    user_agent: str = "Mozilla/5.0 (compatible; SoccerBot/1.0)"
    log_level: str = "INFO"
    
    # Anchor definitions
    anchor_definitions: Dict[str, str] = {
        "Injury-prone players": "The team has several key players who are frequently injured...",
        # ... full list as before
    }
    
    class Config:
        env_file = ".env"