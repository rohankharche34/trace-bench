import time
from systems.base import BaseMemory 
from benchmarks.noise import BenchmarkCorruptor

class EvaluationRunner:
    def __init__(self, dataset: list[dict]):
        self.dataset = dataset
        self.corruptor = BenchmarkCorruptor()
        self.all_memories = list({row["memory"] for row in self.dataset})

    def evaluate_system(self, memory_system: BaseMemory) -> dict:
        total_samples = len(self.dataset)

        start_store = time.perf_counter()
        memory_system.store(self.all_memories)
        store_time = time.perf_counter() - start_store

        metrics = {"clean": 0, "easy": 0, "medium": 0, "hard": 0}

        start_retrieve = time.perf_counter()
        for item in self.dataset:
            expected = item["memory"]
            clean_q = item["query"]

            queries = {
                    "clean": clean_q,
                    "easy": self.corruptor.corrupt(clean_q, "easy"),
                    "medium": self.corruptor.corrupt(clean_q, "medium"),
                    "hard": self.corruptor.corrupt(clean_q, "hard")
                    }

            for tier, variant in queries.items():
                if memory_system.retrieve(variant) == expected:
                    metrics[tier] += 1

        retrieve_time = time.perf_counter() - start_retrieve

        return {
                "clean_acc": (metrics["clean"] / total_samples) * 100,
                "easy_acc": (metrics["easy"] / total_samples) * 100,
                "med_acc": (metrics["medium"] / total_samples) * 100,
                "hard_acc": (metrics["hard"] / total_samples) * 100,
                "robustness": (metrics["hard"] / metrics["clean"] * 100) if metrics["clean"] > 0 else 0.0,
                "index_time": store_time,
                "search_time": retrieve_time
                }
