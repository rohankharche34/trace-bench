import csv
import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from benchmarks.runner import EvaluationRunner
from systems.cosine_memory import CosineMemory
from systems.faiss_memory import FaissMemory
from metrics.anisotropy import calculate_geometry_metrics
from metrics.hubness import calculate_hubness_metrics


def load_dataset(filepath: str) -> list[dict]:
    with open(filepath, mode="r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def main():
    dataset = load_dataset("datasets/synthetic_data.csv")

    registry = [
        {"name": "Pure Cosine (NumPy)", "instance": CosineMemory()},
        {"name": "FAISS (IndexFlatIP)", "instance": FaissMemory()},
    ]
    
    print(f"OpenMemoryBench: Commencing evaluation sweep on {len(dataset)} pairs...\n")
    
    evaluator = EvaluationRunner(dataset)
    results = []
    
    for engine in registry:
        print(f"Running metrics sweep for {engine['name']}...")
        metrics = evaluator.evaluate_system(engine["instance"])
        metrics["name"] = engine["name"]
        results.append(metrics)
        
    print("\n" + "=" * 90)
    print(f"{'Engine Architecture':<22} | {'Clean':<7} | {'Easy':<7} | {'Medium':<7} | {'Hard':<7} | {'Robustness':<10} | {'Search Time':<11}")
    print("-" * 90)
    for r in results:
        print(f"{r['name']:<22} | {r['clean_acc']:>5.1f}% | {r['easy_acc']:>5.1f}% | {r['med_acc']:>5.1f}% | {r['hard_acc']:>5.1f}% | {r['robustness']:>9.1f}% | {r['search_time']:>10.3f}s")
    print("=" * 90 + "\n")

    raw_vectors = registry[1]["instance"].get_all_embeddings() 

    geom = calculate_geometry_metrics(raw_vectors)
    hubby = calculate_hubness_metrics(raw_vectors, k=3)
    
    print("\n" + "=" * 55)
    print("        VECTOR SPACE GEOMETRY & HEALTH REPORT        ")
    print("=" * 55)
    print(f"Space Shape Status (Anisotropy):")
    print(f"   Dominant Axis Magnitude:    {geom['largest_eigenvalue']:.4f}")
    print(f"   Anisotropy Space Ratio:     {geom['anisotropy_ratio']:.2f}x")
    print(f"   Matrix Condition Number:    {geom['condition_number']:.2f}")
    print("-" * 55)
    print(f"Nearest-Neighbor Hubness Status:")
    if "error" in hubby:
        print(f"   {hubby['error']}")
    else:
        print(f"   Max Appearance of One Hub:  {hubby['max_neighbor_appearances']} times")
        print(f"   Hubness Skewness Profile:   {hubby['hubness_skewness']:.3f}")
    print("=" * 55 + "\n")

if __name__ == "__main__":
    main()
