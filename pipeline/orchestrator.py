# pipeline/orchestrator.py
"""Main pipeline orchestrator that ties together all components."""

import time
from typing import Optional, Dict, Any

from ..config.settings import settings
from ..data.data_loader import DataLoader
from ..data.validator import DataValidator
from ..features.feature_engineering import FeatureEngineering
from ..models.model import PredictionModel, EnsembleModel
from ..validation.validator import Validator
from ..presentation.presenter import Presenter
from ..utils.logger import get_logger
from ..utils.exceptions import PipelineError, DataLoadError, ModelError

logger = get_logger(__name__)


class PredictionPipeline:
    """
    End‑to‑end prediction pipeline for soccer matches.
    Coordinates data loading, validation, feature engineering,
    model prediction, result validation, and presentation.
    """

    def __init__(
        self,
        data_loader: Optional[DataLoader] = None,
        validator: Optional[DataValidator] = None,
        feature_eng: Optional[FeatureEngineering] = None,
        model: Optional[PredictionModel] = None,
        ensemble: Optional[EnsembleModel] = None,
        result_validator: Optional[Validator] = None,
        presenter: Optional[Presenter] = None,
    ):
        """
        Initialize pipeline components. If not provided, default instances are created.

        Args:
            data_loader: Fetches raw data from APIs.
            validator: Validates raw data structure.
            feature_eng: Converts raw data to features (semantic scores + structured).
            model: Primary ML model for prediction.
            ensemble: Optional ensemble that can combine multiple models/LLMs.
            result_validator: Compares predictions with actual outcomes.
            presenter: Formats output for end users.
        """
        self.data_loader = data_loader or DataLoader()
        self.validator = validator or DataValidator(strict_mode=False)
        self.feature_eng = feature_eng or FeatureEngineering()
        self.model = model
        self.ensemble = ensemble
        self.result_validator = result_validator or Validator()
        self.presenter = presenter or Presenter()

        if not self.model and not self.ensemble:
            logger.warning("No model or ensemble provided – pipeline will only generate features.")

    def run(
        self,
        home_team: str,
        away_team: str,
        match_date: Optional[str] = None,
        actual_result: Optional[str] = None,
        use_ensemble: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute the full pipeline for a given match.

        Args:
            home_team: Name of the home team.
            away_team: Name of the away team.
            match_date: Date of the match (YYYY-MM-DD). Optional.
            actual_result: Actual match outcome (e.g., 'H', 'D', 'A') for validation.
            use_ensemble: If True, use ensemble model instead of single model.
            **kwargs: Additional arguments passed to components (e.g., force_refresh for data).

        Returns:
            Dictionary containing:
                - match_info
                - features (Series)
                - prediction (if model available)
                - validation (if actual_result provided)
                - formatted_output (string)
        """
        start_time = time.time()
        logger.info(f"Starting pipeline for {home_team} vs {away_team} ({match_date or 'unknown date'})")

        try:
            # ---- Step 1: Load raw data ----
            logger.info("Step 1/5: Loading raw data...")
            raw_data = self.data_loader.load_match_data(
                home_team, away_team, match_date, **kwargs
            )
            if not raw_data:
                raise DataLoadError(f"No data returned for {home_team} vs {away_team}")

            # ---- Step 2: Validate raw data ----
            logger.info("Step 2/5: Validating raw data...")
            validated_data = self.validator.validate(raw_data)

            # ---- Step 3: Feature engineering ----
            logger.info("Step 3/5: Engineering features...")
            features = self.feature_eng.create_features(validated_data)

            # ---- Step 4: Model prediction ----
            prediction = None
            if use_ensemble and self.ensemble:
                logger.info("Step 4/5: Making ensemble prediction...")
                # Prepare additional inputs for ensemble (e.g., raw text for LLM scorers)
                additional_inputs = {
                    "news_text": validated_data.get("news_text", ""),
                    "match_info": validated_data.get("match_info", {}),
                }
                prediction = self.ensemble.predict_proba(
                    features.values.reshape(1, -1),
                    additional_inputs=additional_inputs
                )
            elif self.model:
                logger.info("Step 4/5: Making model prediction...")
                prediction = self.model.predict_proba(features.values.reshape(1, -1))
            else:
                logger.warning("No model available – skipping prediction step.")

            # ---- Step 5: Validation against actual result ----
            validation_result = None
            if actual_result and prediction is not None:
                logger.info("Step 5/5: Validating prediction against actual result...")
                validation_result = self.result_validator.validate(
                    prediction, actual_result
                )

            # ---- Presentation ----
            output = self.presenter.format(
                home_team=home_team,
                away_team=away_team,
                match_date=match_date,
                features=features,
                prediction=prediction,
                actual=actual_result,
                validation=validation_result,
            )

            elapsed = time.time() - start_time
            logger.info(f"Pipeline completed in {elapsed:.2f} seconds")

            return {
                "match_info": {
                    "home_team": home_team,
                    "away_team": away_team,
                    "date": match_date,
                },
                "features": features,
                "prediction": prediction.tolist() if prediction is not None else None,
                "validation": validation_result,
                "formatted_output": output,
            }

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise PipelineError(f"Pipeline execution failed: {e}") from e

    def run_batch(
        self,
        matches: list,
        use_ensemble: bool = False,
        delay: int = 2,
    ) -> list:
        """
        Run pipeline for multiple matches.

        Args:
            matches: List of tuples (home_team, away_team, match_date, actual_result).
            use_ensemble: Whether to use ensemble model.
            delay: Seconds to wait between matches (respect API rate limits).

        Returns:
            List of result dictionaries.
        """
        results = []
        for i, match in enumerate(matches):
            if len(match) == 4:
                home, away, date, actual = match
            elif len(match) == 3:
                home, away, date = match
                actual = None
            else:
                home, away = match
                date = None
                actual = None

            logger.info(f"Processing batch item {i+1}/{len(matches)}: {home} vs {away}")
            result = self.run(home, away, date, actual, use_ensemble)
            results.append(result)

            if i < len(matches) - 1 and delay > 0:
                logger.debug(f"Waiting {delay}s before next request...")
                time.sleep(delay)

        return results