#!/usr/bin/env python3
"""
CLI entry point for the soccer prediction pipeline.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path if running as script
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.orchestrator import PredictionPipeline
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Semantic soccer match prediction pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --home Arsenal --away Chelsea --date 2026-03-01
  python main.py --home Liverpool --away ManCity --actual H --format human
  python main.py --batch matches.csv --output results.json
        """,
    )

    # Single match mode
    parser.add_argument("--home", help="Home team name")
    parser.add_argument("--away", help="Away team name")
    parser.add_argument("--date", help="Match date (YYYY-MM-DD)")
    parser.add_argument("--actual", help="Actual result (H/D/A) for validation")

    # Batch mode
    parser.add_argument("--batch", help="CSV file with columns: home,away,date,actual")

    # Output options
    parser.add_argument(
        "--format",
        choices=["json", "human", "csv", "markdown"],
        default="human",
        help="Output format (default: human)",
    )
    parser.add_argument("--output", help="Output file path (optional)")

    # Model options
    parser.add_argument(
        "--model",
        default="xgboost",
        help="Model type (xgboost, random_forest, logistic_regression)",
    )
    parser.add_argument("--model-path", help="Path to pre-trained model file")
    parser.add_argument("--ensemble", action="store_true", help="Use ensemble model")

    # Other
    parser.add_argument(
        "--cache-dir",
        default=settings.cache_dir,
        help="Cache directory for API responses",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    return parser.parse_args()


def read_batch_file(path: str):
    """Read batch CSV file and return list of match tuples."""
    import pandas as pd

    df = pd.read_csv(path)
    required = {"home", "away"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required}")
    matches = []
    for _, row in df.iterrows():
        date = row.get("date") if "date" in df.columns else None
        actual = row.get("actual") if "actual" in df.columns else None
        matches.append((row["home"], row["away"], date, actual))
    return matches


def main():
    args = parse_args()

    # Configure logging
    if args.verbose:
        import logging

        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    # Override cache dir if provided
    if args.cache_dir:
        from config.settings import settings as global_settings

        global_settings.cache_dir = args.cache_dir

    # Initialize pipeline
    from models.model import PredictionModel, EnsembleModel

    model = None
    ensemble = None
    if args.ensemble:
        ensemble = EnsembleModel()
        # In a real scenario, you'd add models to the ensemble here
        # For now, we just have an empty ensemble
        logger.info("Using ensemble model (empty, no predictions will be made)")
    elif args.model_path:
        model = PredictionModel(model_type=args.model, model_path=args.model_path)
    else:
        # Create untrained model (will not be used for prediction)
        model = PredictionModel(model_type=args.model)
        logger.warning(
            "No model loaded – pipeline will generate features only (no prediction)"
        )

    pipeline = PredictionPipeline(
        model=model,
        ensemble=ensemble if args.ensemble else None,
    )

    # Run batch or single
    if args.batch:
        matches = read_batch_file(args.batch)
        logger.info(f"Running batch for {len(matches)} matches")
        results = pipeline.run_batch(matches)
        # Output batch results
        from presentation.presenter import Presenter

        presenter = Presenter(output_format=args.format, output_file=args.output)
        presenter.format_batch(results)
    else:
        if not args.home or not args.away:
            logger.error("Both --home and --away required for single match mode")
            sys.exit(1)
        result = pipeline.run(
            home_team=args.home,
            away_team=args.away,
            match_date=args.date,
            actual_result=args.actual,
            use_ensemble=args.ensemble,
        )
        # Output
        from presentation.presenter import Presenter

        presenter = Presenter(output_format=args.format, output_file=args.output)
        output = presenter.format(
            home_team=args.home,
            away_team=args.away,
            match_date=args.date,
            features=result["features"],
            prediction=result["prediction"],
            actual=args.actual,
            validation=result.get("validation"),
        )
        if output:
            print(output)


if __name__ == "__main__":
    main()