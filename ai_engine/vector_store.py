import numpy as np
import os
try:
    import faiss
except ImportError:
    faiss = None

class VectorMemory:
    """
    In-memory vector store using FAISS for fast similarity search.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = None
        self.documents = [] # Maps index ID to actual text/metadata

    def initialize_index(self):
        """
        Initializes a flat L2 index.
        """
        if faiss is None:
            raise ImportError("faiss-cpu not installed. Run `pip install -r requirements.txt`")
            
        self.index = faiss.IndexFlatL2(self.dimension)
        print("FAISS index initialized.")

    def add_memory(self, text: str, vector: list):
        """
        Adds a text and its embedding vector to the store.
        """
        if self.index is None:
            self.initialize_index()

        vector_np = np.array([vector], dtype='float32')
        self.index.add(vector_np)
        self.documents.append(text)
        
    def search(self, query_vector: list, k: int = 3):
        """
        Searches for the k nearest neighbors to the query vector.
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        query_np = np.array([query_vector], dtype='float32')
        distances, indices = self.index.search(query_np, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.documents):
                results.append({
                    "text": self.documents[idx],
                    "score": float(distances[0][i])
                })
        
        return results

if __name__ == "__main__":
    # Test stub
    pass
