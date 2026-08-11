from src.evaluation import metrics as m


def test_recall_at_k_hit_and_miss():
    assert m.recall_at_k(5, [3, 5, 7]) == 1.0
    assert m.recall_at_k(5, [3, 9, 7]) == 0.0


def test_reciprocal_rank():
    assert m.reciprocal_rank(5, [3, 5, 7]) == 0.5   # rank 2
    assert m.reciprocal_rank(5, [5, 3, 7]) == 1.0    # rank 1
    assert m.reciprocal_rank(5, [3, 9, 7]) == 0.0    # not found


def test_exact_match_is_case_and_article_insensitive():
    assert m.exact_match("The Eiffel Tower", ["eiffel tower"]) == 1.0
    assert m.exact_match("London", ["Paris"]) == 0.0


def test_defense_recovery_rate_edge_cases():
    assert m.defense_recovery_rate(0.8, 0.4, 0.8) == 1.0    # full recovery
    assert m.defense_recovery_rate(0.8, 0.4, 0.4) == 0.0    # no recovery
    assert m.defense_recovery_rate(0.8, 0.8, 0.8) is None   # no gap to recover
