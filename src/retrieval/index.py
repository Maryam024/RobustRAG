import faiss
import numpy as np


class FaissIndex:
    # exact flat search, corpus is small enough

    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)
        self.doc_ids = []

    def build(self, embeddings: np.ndarray, doc_ids: list):
        self.index.add(embeddings)
        self.doc_ids = list(doc_ids)

    def search(self, query_embeddings: np.ndarray, k: int):
        scores, positions = self.index.search(query_embeddings, k)
        # map FAISS row positions back to doc_ids
        doc_id_matrix = [[self.doc_ids[pos] for pos in row] for row in positions]
        return doc_id_matrix, scores

    def size(self) -> int:
        return self.index.ntotal
