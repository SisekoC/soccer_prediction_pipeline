# pipeline/__init__.py
"""Pipeline orchestration: coordinates data flow through all stages."""

from .orchestrator import PredictionPipeline

__all__ = ["PredictionPipeline"]