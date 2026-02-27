# Soccer Semantic Prediction Pipeline

A production‑grade pipeline that combines multiple data sources (APIs + web scraping) with semantic NLP to generate 40 interpretable features for soccer match prediction. Supports hybrid ML + LLM ensembles for state‑of‑the‑art accuracy.

## Features

- Fetches data from multiple free‑tier APIs: Football-Data.org, Footystats, SoFIFA, The Odds API, and BBC Sport.
- Caches responses to Google Drive (or local disk) to avoid repeated downloads.
- Validates raw data structure and fills missing fields with defaults.
- Computes 40 semantic anchor scores using a hybrid bi‑encoder + cross‑encoder (MPNet + RoBERTa).
- Extracts structured features (team form, player attributes, betting odds) via modular transformers.
- Trains/predicts with XGBoost, Random Forest, Logistic Regression, or an ensemble that combines ML with LLM outputs.
- Validates predictions against actual results and tracks metrics over time.
- Presents results in JSON, human‑readable, CSV, or Markdown.
- Fully containerised with Docker.

## Project Structure

soccer-prediction-pipeline/
├── config/ # Settings and feature definitions
├── data/ # API clients, data loader, validator
├── features/ # Semantic anchors, transformers, feature engineering
├── models/ # ML models, ensemble, registry
├── pipeline/ # Orchestrator
├── validation/ # Prediction validator
├── presentation/ # Output formatters
├── utils/ # Logging, exceptions, helpers
├── tests/ # Unit tests
├── main.py # CLI entry point
├── requirements.txt # Dependencies
├── Dockerfile # Container setup
└── README.md

## Setup

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

Copy .env.example to .env and fill in your API keys (some are optional).

(Optional) Install additional ML libraries:
    !pip install xgboost lightgbm catboost

## Usage

Single match prediction: 
    bash --- python main.py --home Arsenal --away Chelsea --date 2026-03-01 --format human

With actual result (validation):
    bash --- python main.py --home Liverpool --away ManCity --actual H --format json --output result.json

Batch prediction from CSV
    bash --- python main.py --batch matches.csv --output results.json --format json
    The CSV should have columns home,away (and optionally date,actual).

Using an ensemble (ML + LLM): 
    bash --- python main.py --ensemble --home Arsenal --away Chelsea
    Note: You need to implement LLM scorers and add them to the ensemble.

## Configuration

Set environment variables in .env:

Variable	                    Required	                        Description
FOOTBALL_DATA_API_KEY	        No	                                API key for football-data.org
FOOTYSTATS_API_KEY	            No	                                API key for footystats.org
SPORTS_ODDS_API_KEY	            No	                                API key for the-odds-api.com
CACHE_DIR	                    No	                                Directory for cached API responses (default: ./data/cache/)
LOG_LEVEL	                    No	                                Logging level: DEBUG, INFO, WARNING, ERROR (default: INFO)

## Testing

Run the test suite with:
    bash --- pytest

## Docker

Build the image: 
    bash --- docker build -t soccer-pipeline .

Run a container:
    bash --- docker run --rm -v $(pwd)/data:/app/data soccer-pipeline --home Arsenal --away Chelsea

## License

MIT

