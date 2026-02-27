# models/model_registry.py
"""Registry for saving, loading, and versioning trained models."""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from .model import PredictionModel
from ..utils.logger import get_logger
from ..utils.exceptions import ModelError

logger = get_logger(__name__)


class ModelRegistry:
    """
    Simple file‑based model registry with versioning.
    Each model version is stored in a subdirectory with metadata.
    """

    def __init__(self, registry_path: str = "./models/registry/"):
        """
        Args:
            registry_path: Root directory for the model registry.
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)

    def save_model(
        self,
        model: PredictionModel,
        model_name: str,
        version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save a model with metadata.

        Args:
            model: The trained PredictionModel instance.
            model_name: Name of the model (e.g., 'match_outcome').
            version: Version string (auto‑generated if not provided).
            metadata: Additional info (e.g., feature list, training date).

        Returns:
            The version string.
        """
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")

        model_dir = self.registry_path / model_name / version
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save the model file
        model_path = model_dir / "model.pkl"
        model.save(str(model_path))

        # Save metadata
        meta = {
            "model_name": model_name,
            "version": version,
            "model_type": model.model_type,
            "hyperparameters": model.hyperparameters,
            "saved_at": datetime.now().isoformat(),
            **(metadata or {}),
        }
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Model {model_name}:{version} saved to {model_dir}")
        return version

    def load_model(
        self, model_name: str, version: Optional[str] = None
    ) -> tuple[PredictionModel, Dict[str, Any]]:
        """
        Load a model and its metadata.

        Args:
            model_name: Name of the model.
            version: Specific version; if None, load the latest.

        Returns:
            Tuple of (PredictionModel instance, metadata dict).
        """
        model_dir = self.registry_path / model_name
        if not model_dir.exists():
            raise ModelError(f"No models found for name '{model_name}'")

        if version is None:
            # Get latest version (lexicographically highest, assuming timestamps)
            versions = sorted([d.name for d in model_dir.iterdir() if d.is_dir()])
            if not versions:
                raise ModelError(f"No versions found for model '{model_name}'")
            version = versions[-1]

        model_path = model_dir / version / "model.pkl"
        metadata_path = model_dir / version / "metadata.json"

        if not model_path.exists():
            raise ModelError(f"Model file not found: {model_path}")

        # Load metadata
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

        # Infer model type from metadata
        model_type = metadata.get("model_type", "random_forest")
        hyperparams = metadata.get("hyperparameters", {})

        model = PredictionModel(model_type=model_type, hyperparameters=hyperparams)
        model.load(str(model_path))

        logger.info(f"Loaded model {model_name}:{version}")
        return model, metadata

    def list_models(self) -> Dict[str, list]:
        """List all available models and their versions."""
        result = {}
        for model_dir in self.registry_path.iterdir():
            if model_dir.is_dir():
                versions = [d.name for d in model_dir.iterdir() if d.is_dir()]
                result[model_dir.name] = versions
        return result

    def delete_model(self, model_name: str, version: Optional[str] = None) -> None:
        """Delete a specific model version or entire model if version is None."""
        target = self.registry_path / model_name
        if not target.exists():
            raise ModelError(f"Model '{model_name}' not found")

        if version is None:
            shutil.rmtree(target)
            logger.info(f"Deleted entire model '{model_name}'")
        else:
            version_path = target / version
            if not version_path.exists():
                raise ModelError(f"Version '{version}' not found for model '{model_name}'")
            shutil.rmtree(version_path)
            logger.info(f"Deleted {model_name}:{version}")