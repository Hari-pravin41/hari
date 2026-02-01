from typing import List, Optional
import os

# Conditional import to avoid crashing if dependencies aren't installed yet
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

class EmbeddingService:
    """
    Manages local embedding generation using SentenceTransformers (HuggingFace).
    Runs completely offline on CPU/GPU.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None

    def load_model(self):
        """
        Loads the model into memory. 
        Note: This will download the model (~80MB) from HF Hub on first run.
        """
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers not installed. Run `pip install -r requirements.txt`")
        
        print(f"Loading embedding model: {self.model_name} on {self.device}...")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        print("Model loaded successfully.")

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generates a vector embedding for the given text.
        """
        if not self.model:
            self.load_model()
            
        if not text:
            return None
            
        # Generate embedding
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

if __name__ == "__main__":
    # Test stub
    service = EmbeddingService()
    # service.load_model() # Uncomment to test (requires install)
    # print(service.generate_embedding("Test sentence"))
