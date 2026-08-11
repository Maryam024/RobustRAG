import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    # BGE needs a query prefix for asymmetric retrieval
    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str, batch_size: int = 64, device: str = "cpu"):
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size

    def encode_passages(self, texts: list) -> np.ndarray:
        return self._encode(texts)

    def encode_queries(self, texts: list) -> np.ndarray:
        prefixed = [self.QUERY_PREFIX + t for t in texts]
        return self._encode(prefixed)

    def _encode(self, texts: list) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,   # so FAISS inner product = cosine sim
            show_progress_bar=len(texts) > 500,
        )
        return np.asarray(embeddings, dtype="float32")
