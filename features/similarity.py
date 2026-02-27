import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def compute_concept_scores(text: str, model, anchor_embeddings: np.ndarray, concept_names: list) -> pd.Series:
    """Return Series of cosine similarities for the given text."""
    if not text or not text.strip():
        return pd.Series(0.0, index=concept_names)
    
    text_emb = model.encode([text])
    similarities = cosine_similarity(text_emb, anchor_embeddings).flatten()
    return pd.Series(similarities, index=concept_names)