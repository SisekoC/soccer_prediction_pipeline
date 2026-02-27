# utils/exceptions.py
"""Custom exceptions for the pipeline."""


class PipelineError(Exception):
    """Base exception for all pipeline errors."""
    pass


class DataLoadError(PipelineError):
    """Raised when data cannot be loaded from any source."""
    pass


class DataValidationError(PipelineError):
    """Raised when raw data fails validation checks."""
    pass


class FeatureEngineeringError(PipelineError):
    """Raised during feature computation."""
    pass


class ModelError(PipelineError):
    """Raised when model training or prediction fails."""
    pass


class APIError(PipelineError):
    """Raised when an external API call fails (HTTP errors, rate limits)."""
    pass