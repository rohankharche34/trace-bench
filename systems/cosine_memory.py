from sentence_transformers import SentenceTransformer
import numpy as np
import csv

class CosineMemory:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.memories = []
        self.embeddings = None

    def store(self, memories: list[str]):
        if not memories:
            return

        unique_memories = [m for m in memories if m not in self.memories]
        if not unique_memories:
            return

        self.memories.extend(unique_memories)
        new_embeddings = self.model.encode(unique_memories, convert_to_numpy = True)

        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

    def retrieve(self, query: str) -> str | None:
        if not self.memories or self.embeddings is None:
            return None

        query_embeddings = self.model.encode([query], convert_to_numpy = True)
        similarity_scores = self.model.similarity(query_embeddings, self.embeddings)
        scores = similarity_scores.numpy()[0]

        top_1_index = np.argmax(scores)
        return self.memories[top_1_index]
