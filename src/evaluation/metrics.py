import re
import string
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
_smoothing = SmoothingFunction().method1


def _normalize(text: str) -> str:
    # standard SQuAD-style normalization
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


def recall_at_k(gold_doc_id: int, retrieved_doc_ids: list) -> float:
    return 1.0 if gold_doc_id in retrieved_doc_ids else 0.0


def precision_at_k(gold_doc_id: int, retrieved_doc_ids: list, k: int) -> float:
    return (1.0 / k) if gold_doc_id in retrieved_doc_ids else 0.0


def reciprocal_rank(gold_doc_id: int, retrieved_doc_ids: list) -> float:
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id == gold_doc_id:
            return 1.0 / rank
    return 0.0


def exact_match(prediction: str, gold_answers: list) -> float:
    norm_pred = _normalize(prediction)
    return 1.0 if any(norm_pred == _normalize(g) for g in gold_answers) else 0.0


def bleu(prediction: str, gold_answers: list) -> float:
    pred_tokens = _normalize(prediction).split()
    if not pred_tokens:
        return 0.0
    references = [_normalize(g).split() for g in gold_answers]
    return sentence_bleu(references, pred_tokens, smoothing_function=_smoothing)


def rouge_l(prediction: str, gold_answers: list) -> float:
    if not prediction.strip():
        return 0.0
    # best score across gold answers
    return max(_rouge.score(g, prediction)["rougeL"].fmeasure for g in gold_answers)


def defense_recovery_rate(clean_score: float, poisoned_score: float, defended_score: float):
    # fraction of the poisoning gap recovered
    gap = clean_score - poisoned_score
    if abs(gap) < 1e-9:
        return None
    return (defended_score - poisoned_score) / gap
