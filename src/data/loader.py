import json
import random
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Passage:
    doc_id: int
    text: str
    title: str


@dataclass
class QueryExample:
    question: str
    answers: list = field(default_factory=list)
    gold_doc_id: int = -1


def _parse_squad(raw: dict):
    # flatten title/paragraph/qas into passages + queries
    passages, queries = [], []
    seen_contexts = {}

    for article in raw["data"]:
        title = article.get("title", "")
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            if context not in seen_contexts:
                doc_id = len(passages)
                seen_contexts[context] = doc_id
                passages.append(Passage(doc_id=doc_id, text=context, title=title))
            doc_id = seen_contexts[context]

            for qa in paragraph["qas"]:
                if qa.get("is_impossible", False):
                    continue
                answer_texts = list({a["text"] for a in qa["answers"]})
                if not answer_texts:
                    continue
                queries.append(QueryExample(
                    question=qa["question"],
                    answers=answer_texts,
                    gold_doc_id=doc_id,
                ))

    return passages, queries


def load_squad_corpus(path: str, num_passages: int, seed: int = 42):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"SQuAD file not found at {path}. Download it from "
            "https://rajpurkar.github.io/SQuAD-explorer/ and place it there."
        )

    with open(path, "r") as f:
        raw = json.load(f)

    passages, queries = _parse_squad(raw)

    rng = random.Random(seed)
    if num_passages < len(passages):
        sampled = rng.sample(passages, num_passages)
    else:
        sampled = passages

    kept_ids = {p.doc_id for p in sampled}
    # reindex doc_ids to 0..N-1
    remap = {old_id: new_id for new_id, old_id in enumerate(sorted(kept_ids))}
    corpus = [Passage(doc_id=remap[p.doc_id], text=p.text, title=p.title) for p in sampled]

    filtered_queries = [
        QueryExample(q.question, q.answers, remap[q.gold_doc_id])
        for q in queries if q.gold_doc_id in kept_ids
    ]

    return corpus, filtered_queries
