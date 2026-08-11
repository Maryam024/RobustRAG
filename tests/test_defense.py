import numpy as np

from src.defense.filter import suppress_near_duplicates


def test_suppresses_near_identical_embeddings():
    # candidates 0 and 1 are near-duplicates (same vector), 2 is distinct
    v = np.array([1.0, 0.0], dtype="float32")
    orthogonal = np.array([0.0, 1.0], dtype="float32")
    doc_embeddings = {0: v, 1: v.copy(), 2: orthogonal}

    candidates = [
        {"doc_id": 0, "score": 0.9},
        {"doc_id": 1, "score": 0.89},
        {"doc_id": 2, "score": 0.5},
    ]

    kept = suppress_near_duplicates(candidates, doc_embeddings, k=2, similarity_threshold=0.97)
    kept_ids = [c["doc_id"] for c in kept]
    assert kept_ids == [0, 2]   # doc 1 suppressed as a duplicate of doc 0, doc 2 backfilled in


def test_keeps_all_when_no_duplicates_present():
    doc_embeddings = {
        0: np.array([1.0, 0.0], dtype="float32"),
        1: np.array([0.0, 1.0], dtype="float32"),
        2: np.array([-1.0, 0.0], dtype="float32"),
    }
    candidates = [{"doc_id": i, "score": 1.0 - i * 0.1} for i in range(3)]

    kept = suppress_near_duplicates(candidates, doc_embeddings, k=3, similarity_threshold=0.97)
    assert [c["doc_id"] for c in kept] == [0, 1, 2]
