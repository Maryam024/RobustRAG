from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.index import FaissIndex


class Retriever:
    def __init__(self, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model
        self.index = None
        self.passage_lookup = {}    # doc_id -> Passage
        self.doc_embeddings = {}    # doc_id -> embedding, used by defense

    def build_index(self, passages: list):
        self.passage_lookup = {p.doc_id: p for p in passages}
        embeddings = self.embedding_model.encode_passages([p.text for p in passages])
        self.doc_embeddings = {p.doc_id: emb for p, emb in zip(passages, embeddings)}
        self.index = FaissIndex(dim=embeddings.shape[1])
        self.index.build(embeddings, doc_ids=[p.doc_id for p in passages])
        return embeddings   # let callers reuse without re-encoding

    def retrieve(self, questions: list, k: int):
        query_embeddings = self.embedding_model.encode_queries(questions)
        return self.retrieve_from_embeddings(query_embeddings, k)

    def retrieve_from_embeddings(self, query_embeddings, k: int):
        doc_id_matrix, scores = self.index.search(query_embeddings, k)
        results = []
        for doc_ids, row_scores in zip(doc_id_matrix, scores):
            results.append([
                {"doc_id": did, "text": self.passage_lookup[did].text, "score": float(s)}
                for did, s in zip(doc_ids, row_scores)
            ])
        return results
