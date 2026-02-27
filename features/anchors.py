from sentence_transformers import SentenceTransformer
import numpy as np
from ..config.settings import settings

class AnchorStore:
    """
    Stores the 40 anchor concepts, their sentences, and pre‑computed embeddings
    using a powerful bi‑encoder (e.g., all-mpnet-base-v2).
    """
    def __init__(self, model_name: str = None):
        # Use a stronger bi‑encoder for retrieval
        self.model_name = model_name or "all-mpnet-base-v2"
        self.model = SentenceTransformer(self.model_name)
        
        # Load anchor definitions from settings
        self.concept_names = list(settings.anchor_definitions.keys())
        self.anchor_sentences = list(settings.anchor_definitions.values())
        
        # Pre‑compute embeddings for all anchors
        self.anchor_embeddings = self.model.encode(self.anchor_sentences)
        print(f"AnchorStore: {len(self.concept_names)} anchors, embeddings shape {self.anchor_embeddings.shape}")
    
    def get_embeddings(self) -> np.ndarray:
        return self.anchor_embeddings
    
    def get_sentences(self) -> list:
        return self.anchor_sentences
    
    def get_names(self) -> list:
        return self.concept_names