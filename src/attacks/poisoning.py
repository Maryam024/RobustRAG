import random
from src.data.loader import Passage


def _build_answer_pool(queries) -> list:
    # distinct answers, used for fact substitution
    pool = []
    seen = set()
    for q in queries:
        for a in q.answers:
            if a not in seen:
                seen.add(a)
                pool.append(a)
    return pool


class PoisoningEngine:
    # builds poisoned passages, appends to corpus

    def __init__(self, strategy: str, rate: float, seed: int = 42):
        if strategy not in ("near_duplicate", "contradictory", "irrelevant"):
            raise ValueError(f"Unknown poisoning strategy: {strategy}")
        self.strategy = strategy
        self.rate = rate
        self.rng = random.Random(seed)

    def apply(self, corpus: list, queries: list):
        num_poisoned = max(1, int(len(corpus) * self.rate))
        next_id = max(p.doc_id for p in corpus) + 1

        if self.strategy == "near_duplicate":
            poisoned = self._near_duplicates(corpus, num_poisoned, next_id)
        elif self.strategy == "contradictory":
            poisoned = self._contradictory(corpus, queries, num_poisoned, next_id)
        else:
            poisoned = self._irrelevant(corpus, num_poisoned, next_id)

        poisoned_ids = {p.doc_id for p in poisoned}
        return corpus + poisoned, poisoned_ids

    def _near_duplicates(self, corpus, num_poisoned, next_id):
        sources = self.rng.choices(corpus, k=num_poisoned)
        return [
            Passage(doc_id=next_id + i, text=src.text, title=src.title)
            for i, src in enumerate(sources)
        ]

    def _contradictory(self, corpus, queries, num_poisoned, next_id):
        # swaps the true answer for a wrong one
        answer_pool = _build_answer_pool(queries)
        doc_to_answers = {}
        for q in queries:
            doc_to_answers.setdefault(q.gold_doc_id, []).extend(q.answers)

        eligible = [p for p in corpus if doc_to_answers.get(p.doc_id)]
        if not eligible:
            return self._near_duplicates(corpus, num_poisoned, next_id)

        poisoned = []
        sources = self.rng.choices(eligible, k=num_poisoned)
        for i, src in enumerate(sources):
            true_answer = self.rng.choice(doc_to_answers[src.doc_id])
            wrong_candidates = [a for a in answer_pool if a != true_answer]
            if not wrong_candidates or true_answer not in src.text:
                text = src.text   # fall back to plain duplicate
            else:
                wrong_answer = self.rng.choice(wrong_candidates)
                text = src.text.replace(true_answer, wrong_answer)
            poisoned.append(Passage(doc_id=next_id + i, text=text, title=src.title))
        return poisoned

    def _irrelevant(self, corpus, num_poisoned, next_id):
        # word-shuffled, embedding-close but incoherent
        poisoned = []
        sources = self.rng.choices(corpus, k=num_poisoned)
        for i, src in enumerate(sources):
            words = src.text.split()
            self.rng.shuffle(words)
            poisoned.append(Passage(doc_id=next_id + i, text=" ".join(words), title=src.title))
        return poisoned
