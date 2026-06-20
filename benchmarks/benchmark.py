import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import time
from systems.cosine_memory import CosineMemory
from systems.faiss_memory import FaissMemory

def load_csv_dataset(filepath: str) -> list[dict]:
    dataset = []
    with open(filepath, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            dataset.append(row)
    return dataset

def benchmark_system(system_instance, dataset: list[dict], system_name: str):
    all_memories = list({row["memory"] for row in dataset})
    
    start_store = time.perf_counter()
    system_instance.store(all_memories)
    store_time = time.perf_counter() - start_store
    
    correct_count = 0
    start_retrieve = time.perf_counter()
    
    for item in dataset:
        query = item["query"]
        expected = item["memory"]
        
        retrieved = system_instance.retrieve(query)
        if retrieved == expected:
            correct_count += 1
            
    retrieve_time = time.perf_counter() - start_retrieve
    accuracy = (correct_count / len(dataset)) * 100

    return {
        "name": system_name,
        "accuracy": f"{accuracy:.2f}%",
        "score": f"{correct_count}/{len(dataset)}",
        "store_time": f"{store_time:.4f}s",
        "retrieve_time": f"{retrieve_time:.4f}s"
    }

def main():
    csv_path = "../datasets/synthetic_data.csv" 
    dataset = load_csv_dataset(csv_path)
    
    print(f"Initializing Memory Benchmark on {len(dataset)} items...\n")
    
    cosine_results = benchmark_system(CosineMemory(), dataset, "Pure Cosine (NumPy)")
    faiss_results = benchmark_system(FaissMemory(), dataset, "FAISS (IndexFlatIP)")
    
    print(f"{'System Type':<22} | {'Accuracy':<10} | {'Raw Score':<10} | {'Index Time':<12} | {'Search Time':<12}")
    print("-" * 75)
    for res in [cosine_results, faiss_results]:
        print(f"{res['name']:<22} | {res['accuracy']:<10} | {res['score']:<10} | {res['store_time']:<12} | {res['retrieve_time']:<12}")

if __name__ == "__main__":
    main()
