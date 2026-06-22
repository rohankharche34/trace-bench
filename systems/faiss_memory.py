from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from systems.base import BaseMemory

class FaissMemory(BaseMemory):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.memories = []
        self.index = None
        self.dimension = None
        self._raw_embeddings = None

    def store(self, memories: list[str]):
        """Encodes raw text memories and loads them into a FAISS index."""
        if not memories:
            return
        
        unique_memories = [m for m in memories if m not in self.memories]
        if not unique_memories:
            return
            
        self.memories.extend(unique_memories)
        
        embeddings = self.model.encode(unique_memories, convert_to_numpy=True).astype('float32')

        if self._raw_embeddings is None:
            self._raw_embeddings = embeddings
        else:
            self._raw_embeddings = np.vstack([self._raw_embeddings, embeddings])
        
        faiss.normalize_L2(embeddings)
        
        if self.index is None:
            self.dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(self.dimension)
            
        self.index.add(embeddings)

    def retrieve(self, query: str) -> str | None:
        """Finds the single most semantically similar memory using FAISS index execution."""
        if not self.memories or self.index is None:
            return None
            
        query_embedding = self.model.encode([query], convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        D, I = self.index.search(query_embedding, k=1)
        
        top_1_idx = I[0][0]
        
        if top_1_idx == -1:
            return None
            
        return self.memories[top_1_idx]

    def get_all_embeddings(self) -> np.ndarray:
        return self._raw_embeddings
