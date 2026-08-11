import numpy as np


def suppress_near_duplicates(candidates: list, doc_embeddings: dict, k: int, similarity_threshold: float):
    # greedily keeps top candidates, skips near-duplicates
    kept, kept_embeddings = [], []

    for candidate in candidates:
        emb = doc_embeddings[candidate["doc_id"]]
        if kept_embeddings:
            similarities = np.dot(np.stack(kept_embeddings), emb)
            if similarities.max() >= similarity_threshold:
                continue
        kept.append(candidate)
        kept_embeddings.append(emb)
        if len(kept) == k:
            break

    return kept
