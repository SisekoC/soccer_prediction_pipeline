# features/transformers.py
from abc import ABC, abstractmethod
from typing import Any, Dict, Union
import pandas as pd

class FeatureTransformer(ABC):
    """Base class for all feature transformers."""
    
    @abstractmethod
    def transform(self, data: Dict[str, Any]) -> Union[pd.Series, pd.DataFrame]:
        """
        Extract features from the raw data dictionary returned by DataLoader.
        
        Args:
            data: The master data dictionary containing all API responses.
        
        Returns:
            A pandas Series (or DataFrame) of features.
        """
        pass