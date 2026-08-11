import numpy as np


def add_query_noise(query_embeddings: np.ndarray, level: float, seed: int = 42) -> np.ndarray:
    # gaussian noise on query side only
    rng = np.random.RandomState(seed)
    noise = rng.normal(loc=0.0, scale=level, size=query_embeddings.shape).astype("float32")
    noisy = query_embeddings + noise
    norms = np.linalg.norm(noisy, axis=1, keepdims=True)
    norms[norms == 0] = 1.0   # avoid divide by zero
    return noisy / norms
