import numpy as np

def calculate_geometry_metrics(embeddings: np.ndarray) -> dict:
    if embeddings is None or len(embeddings) < 2:
        return {"error": "Insufficient embeddings to calculate metrics."}

    covariance_matrix = np.cov(embeddings, rowvar=False)

    eigenvalues = np.linalg.eigh(covariance_matrix)[0]

    eigenvalues = np.sort(eigenvalues)[::-1]

    eigenvalues = np.clip(eigenvalues, a_min=1e-9, a_max=None)

    largest_ev = eigenvalues[0]
    smallest_ev = eigenvalues[-1]
    average_ev = np.mean(eigenvalues)

    anisotropy_ratio = largest_ev / average_ev 
    condition_number = largest_ev / smallest_ev

    return {
        "largest_eigenvalue": largest_ev,
        "average_eigenvalue": average_ev,
        "anisotropy_ratio": anisotropy_ratio,
        "condition_number": condition_number
    }
