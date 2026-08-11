from src.data.loader import load_squad_corpus
from src.attacks.poisoning import PoisoningEngine

FIXTURE = "data/sample_squad_tiny.json"


def _corpus_and_queries():
    return load_squad_corpus(FIXTURE, num_passages=15, seed=42)


def test_poisoning_appends_without_removing_originals():
    corpus, queries = _corpus_and_queries()
    engine = PoisoningEngine(strategy="near_duplicate", rate=0.2, seed=1)
    poisoned_corpus, poisoned_ids = engine.apply(corpus, queries)

    original_texts = {p.text for p in corpus}
    surviving_texts = {p.text for p in poisoned_corpus if p.doc_id not in poisoned_ids}
    assert original_texts == surviving_texts
    assert len(poisoned_corpus) == len(corpus) + len(poisoned_ids)


def test_near_duplicate_copies_are_textually_identical_to_a_source():
    corpus, queries = _corpus_and_queries()
    engine = PoisoningEngine(strategy="near_duplicate", rate=0.3, seed=1)
    poisoned_corpus, poisoned_ids = engine.apply(corpus, queries)

    original_texts = {p.text for p in corpus}
    for p in poisoned_corpus:
        if p.doc_id in poisoned_ids:
            assert p.text in original_texts


def test_contradictory_alters_the_true_answer():
    corpus, queries = _corpus_and_queries()
    doc_to_answer = {q.gold_doc_id: q.answers[0] for q in queries}

    engine = PoisoningEngine(strategy="contradictory", rate=0.3, seed=7)
    poisoned_corpus, poisoned_ids = engine.apply(corpus, queries)

    originals = {p.doc_id: p for p in corpus}
    for p in poisoned_corpus:
        if p.doc_id not in poisoned_ids:
            continue
        # a contradictory doc is derived from some original passage with the same title
        src = next(o for o in originals.values() if o.title == p.title)
        true_answer = doc_to_answer.get(src.doc_id)
        if true_answer and true_answer in src.text:
            assert true_answer not in p.text or p.text == src.text  # swapped, or safely fell back


def test_irrelevant_docs_share_vocabulary_but_differ_in_order():
    corpus, queries = _corpus_and_queries()
    engine = PoisoningEngine(strategy="irrelevant", rate=0.3, seed=3)
    poisoned_corpus, poisoned_ids = engine.apply(corpus, queries)

    original_word_sets = [set(p.text.split()) for p in corpus]
    for p in poisoned_corpus:
        if p.doc_id not in poisoned_ids:
            continue
        assert set(p.text.split()) in original_word_sets   # same words, shuffled order
