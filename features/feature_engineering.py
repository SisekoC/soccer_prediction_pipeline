import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import CrossEncoder

from .anchors import AnchorStore
from ..utils.logger import get_logger

logger = get_logger(__name__)


class FeatureEngineering:
    """
    Converts raw match text into 40 semantic anchor scores using a hybrid
    bi‑encoder (retrieval) + cross‑encoder (reranking) approach.
    """
    def __init__(self, top_k: int = 10):
        """
        Args:
            top_k: Number of top anchors to rerank with cross‑encoder.
        """
        self.top_k = top_k
        
        # 1. Bi‑encoder anchor store (for fast retrieval)
        self.anchor_store = AnchorStore()
        self.bi_encoder = self.anchor_store.model
        self.anchor_embeddings = self.anchor_store.get_embeddings()
        self.anchor_sentences = self.anchor_store.get_sentences()
        self.concept_names = self.anchor_store.get_names()
        
        # 2. Cross‑encoder for precise reranking
        #    Using a model trained on semantic textual similarity (STSb)
        self.cross_encoder = CrossEncoder('cross-encoder/stsb-roberta-large')
        logger.info(f"FeatureEngineering initialized with top_k={top_k}")
    
    def create_features(self, raw_text: str) -> pd.Series:
        """
        Compute 40 anchor similarity scores for the input text.
        Returns a pandas Series indexed by concept names.
        """
        if not raw_text or len(raw_text.strip()) == 0:
            logger.warning("Empty input text, returning zero scores")
            return pd.Series(0.0, index=self.concept_names)
        
        # ---- Stage 1: Bi‑encoder retrieval ----
        # Encode input text
        text_emb = self.bi_encoder.encode([raw_text])
        # Compute cosine similarity with all anchor embeddings
        initial_sims = cosine_similarity(text_emb, self.anchor_embeddings).flatten()
        
        # ---- Stage 2: Cross‑encoder reranking ----
        # Select top_k anchors with highest initial similarity
        if self.top_k < len(self.concept_names):
            top_indices = np.argsort(initial_sims)[-self.top_k:]
        else:
            top_indices = np.arange(len(self.concept_names))
        
        # Prepare input pairs for cross‑encoder: (text, anchor_sentence)
        pairs = [[raw_text, self.anchor_sentences[i]] for i in top_indices]
        
        # Get refined scores from cross‑encoder (returns values between 0 and 5, typically)
        reranked_scores = self.cross_encoder.predict(pairs)
        
        # Normalize cross‑encoder scores to [0,1] range if needed
        # (The model outputs similarity scores roughly in [0,5]; we can min‑max scale
        #  but since we only care about relative ranking, we can use as is.
        #  For consistency with initial cosine similarities, we might clip to [0,1].)
        reranked_scores = np.clip(reranked_scores / 5.0, 0, 1)  # scale to 0-1
        
        # Build final scores: use reranked scores for top indices, keep initial for others
        final_scores = initial_sims.copy()
        for idx, score in zip(top_indices, reranked_scores):
            final_scores[idx] = score
        
        # Return as pandas Series
        return pd.Series(final_scores, index=self.concept_names)