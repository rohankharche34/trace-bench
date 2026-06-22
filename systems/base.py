from abc import ABC, abstractmethod
import numpy as np

class BaseMemory(ABC):
    
    @abstractmethod
    def store(self, memories: list[str]) -> None:
        pass

    @abstractmethod
    def retrieve(self, query: str) -> str | None:
        pass

    @abstractmethod
    def get_all_embeddings(self) -> np.ndarray:
        pass
