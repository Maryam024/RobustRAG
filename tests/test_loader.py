from src.data.loader import load_squad_corpus

FIXTURE = "data/sample_squad_tiny.json"


def test_loads_expected_counts():
    corpus, queries = load_squad_corpus(FIXTURE, num_passages=15, seed=42)
    assert len(corpus) == 15
    assert len(queries) == 15  # one answerable question per passage in this fixture


def test_doc_ids_are_contiguous_after_subsampling():
    corpus, _ = load_squad_corpus(FIXTURE, num_passages=10, seed=42)
    assert sorted(p.doc_id for p in corpus) == list(range(10))


def test_queries_reference_valid_gold_ids():
    corpus, queries = load_squad_corpus(FIXTURE, num_passages=10, seed=42)
    valid_ids = {p.doc_id for p in corpus}
    assert all(q.gold_doc_id in valid_ids for q in queries)


def test_subsampling_is_deterministic_given_seed():
    corpus_a, _ = load_squad_corpus(FIXTURE, num_passages=8, seed=7)
    corpus_b, _ = load_squad_corpus(FIXTURE, num_passages=8, seed=7)
    assert [p.text for p in corpus_a] == [p.text for p in corpus_b]
