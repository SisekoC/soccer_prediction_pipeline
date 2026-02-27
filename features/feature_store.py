"""
Feature store for persisting computed semantic anchor scores.
Allows quick retrieval without re‑running the NLP pipeline.
"""

import json
from pathlib import Path
from typing import Optional, Union
import pandas as pd

from ..utils.logger import get_logger

logger = get_logger(__name__)


class FeatureStore:
    """
    Persistent storage for pre‑computed feature vectors.

    Features are stored as JSON files under a base directory,
    organised by entity type and version. The base directory is
    created if it does not exist.

    Example structure:
        ./data/feature_store/
            match/
                v1/
                    arsenal_chelsea_2026-02-21.json
                v2/
                    arsenal_chelsea_2026-02-21.json
            team/
                v1/
                    liverpool.json
    """

    def __init__(self, storage_path: Union[str, Path] = "./data/feature_store/"):
        """
        Args:
            storage_path: Root directory for the feature store.
        """
        self.base_path = Path(storage_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Feature store initialised at {self.base_path}")

    def _feature_path(self, entity_type: str, entity_id: str, version: str = "v1") -> Path:
        """
        Construct the full file path for a given entity and version.

        Args:
            entity_type: e.g., 'match', 'team', 'league'
            entity_id:   Unique identifier (must be filesystem‑safe)
            version:     Feature version (default 'v1')

        Returns:
            Path object pointing to the JSON file.
        """
        # Sanitise entity_id to prevent path traversal attacks
        safe_id = entity_id.replace("/", "_").replace("\\", "_")
        return self.base_path / entity_type / version / f"{safe_id}.json"

    def get_features(self, entity_type: str, entity_id: str, version: str = "v1") -> Optional[pd.Series]:
        """
        Retrieve features for an entity if they exist in the store.

        Args:
            entity_type: Type of entity (e.g., 'match')
            entity_id:   Unique identifier (e.g., 'arsenal_chelsea_2026-02-21')
            version:     Feature version (default 'v1')

        Returns:
            pandas Series with concept names as index, or None if not found.
        """
        path = self._feature_path(entity_type, entity_id, version)
        if not path.exists():
            logger.debug(f"No cached features at {path}")
            return None

        try:
            with open(path, "r") as f:
                data = json.load(f)
            # Expected JSON structure: {'concept_names': list, 'scores': list, 'version': str}
            series = pd.Series(data['scores'], index=data['concept_names'])
            logger.debug(f"Loaded features from {path}")
            return series
        except (json.JSONDecodeError, KeyError, IOError) as e:
            logger.error(f"Failed to load features from {path}: {e}")
            return None

    def store_features(self, entity_type: str, entity_id: str, features: pd.Series, version: str = "v1"):
        """
        Store a feature vector for an entity.

        Args:
            entity_type: Type of entity (e.g., 'match')
            entity_id:   Unique identifier (e.g., 'arsenal_chelsea_2026-02-21')
            features:    pandas Series with index = concept names, values = scores
            version:     Feature version (default 'v1')
        """
        path = self._feature_path(entity_type, entity_id, version)
        # Ensure the parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'concept_names': features.index.tolist(),
            'scores': features.values.tolist(),
            'version': version
        }

        try:
            with open(path, "w") as f:
                json.dump(data, f)
            logger.info(f"Stored features for {entity_type}/{entity_id} (version {version})")
        except IOError as e:
            logger.error(f"Failed to store features to {path}: {e}")
            raise