from src.evaluation import metrics as m


def evaluate_retrieval(queries: list, retrieved_results: list, k: int) -> dict:
    # retrieval_accuracy is an alias for mean recall@k
    recalls, precisions, rr = [], [], []
    for q, results in zip(queries, retrieved_results):
        doc_ids = [r["doc_id"] for r in results]
        recalls.append(m.recall_at_k(q.gold_doc_id, doc_ids))
        precisions.append(m.precision_at_k(q.gold_doc_id, doc_ids, k))
        rr.append(m.reciprocal_rank(q.gold_doc_id, doc_ids))

    n = len(queries)
    return {
        "recall_at_k": sum(recalls) / n,
        "precision_at_k": sum(precisions) / n,
        "mrr": sum(rr) / n,
        "retrieval_accuracy": sum(recalls) / n,
    }


def evaluate_answers(queries: list, retrieved_results: list, reader) -> dict:
    # uses only the top-1 passage as context
    ems, bleus, rouges = [], [], []
    for q, results in zip(queries, retrieved_results):
        top_context = results[0]["text"] if results else ""
        prediction = reader.answer(q.question, top_context)
        ems.append(m.exact_match(prediction, q.answers))
        bleus.append(m.bleu(prediction, q.answers))
        rouges.append(m.rouge_l(prediction, q.answers))

    n = len(queries)
    return {
        "exact_match": sum(ems) / n,
        "bleu": sum(bleus) / n,
        "rouge_l": sum(rouges) / n,
    }
