import numpy as np
from scipy.stats import skew 

def calculate_hubness_metrics(embeddings: np.ndarray, k: int = 3) -> dict:
    n_samples = len(embeddings)
    if n_samples <= k:
        return {"error": "Dataset size must be greater than k to calculate the hubness."}

    norms = np.linalg.norm(embeddings, axis = 1, keepdims = True)
    norm_embeddings = embeddings / norms 

    similarity_matrix = np.dot(norm_embeddings, norm_embeddings.T)


    np.fill_diagonal(similarity_matrix, -1.0)

    k_counts = np.zeros(n_samples)

    for i in range(n_samples):
        top_k_indices = np.argsort(similarity_matrix[i])[::-1][:k]
        for idx in top_k_indices:
            k_counts[idx] += 1

    max_hits = int(np.max(k_counts))
    hubness_skew = skew(k_counts)

    return {
            "raw_counts": k_counts.tolist(),
            "max_neighbor_appearances": max_hits,
            "hubness_skewness": hubness_skew
    }
