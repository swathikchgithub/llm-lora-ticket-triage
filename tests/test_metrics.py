import metrics


def test_accuracy_all_correct():
    assert metrics.accuracy(["a", "b", "c"], ["a", "b", "c"]) == 1.0


def test_accuracy_all_wrong():
    assert metrics.accuracy(["a", "a", "a"], ["b", "b", "b"]) == 0.0


def test_accuracy_partial():
    assert metrics.accuracy(["a", "b", "c", "d"], ["a", "x", "c", "y"]) == 0.5


def test_accuracy_empty_returns_zero():
    assert metrics.accuracy([], []) == 0.0


def test_per_category_accuracy():
    preds = ["Network", "Network", "Hardware", "Hardware"]
    truth = ["Network", "Hardware", "Hardware", "Hardware"]
    result = metrics.per_category_accuracy(preds, truth)
    assert result["Network"] == 1.0  # 1/1 correct
    assert result["Hardware"] == 2 / 3


def test_confusion_matrix_diagonal_is_correct_predictions():
    labels = ["A", "B"]
    preds = ["A", "A", "B", "B"]
    truth = ["A", "B", "B", "A"]
    cm = metrics.confusion_matrix(preds, truth, labels)
    assert cm["A"]["A"] == 1
    assert cm["A"]["B"] == 1  # true A predicted B
    assert cm["B"]["B"] == 1
    assert cm["B"]["A"] == 1


def test_top_confusions_excludes_correct_predictions():
    preds = ["A", "A", "B"]
    truth = ["A", "B", "B"]
    confusions = metrics.top_confusions(preds, truth)
    assert confusions == [(("B", "A"), 1)]


def test_top_confusions_respects_top_n():
    preds = ["A", "B", "C"]
    truth = ["X", "Y", "Z"]
    confusions = metrics.top_confusions(preds, truth, top_n=2)
    assert len(confusions) == 2


def test_consistency_rate_all_agree():
    repeated = {"t1": ["Network", "Network", "Network"], "t2": ["Hardware", "Hardware"]}
    assert metrics.consistency_rate(repeated) == 1.0


def test_consistency_rate_none_agree():
    repeated = {"t1": ["Network", "Hardware", "Software"]}
    assert metrics.consistency_rate(repeated) == 0.0


def test_consistency_rate_partial():
    repeated = {
        "t1": ["Network", "Network", "Network"],  # consistent
        "t2": ["Hardware", "Software", "Hardware"],  # inconsistent
    }
    assert metrics.consistency_rate(repeated) == 0.5


def test_consistency_rate_empty():
    assert metrics.consistency_rate({}) == 0.0


def test_json_parse_success_rate():
    outputs = ["{}", "not json", "{}", "{}"]
    flags = [True, False, True, True]
    assert metrics.json_parse_success_rate(outputs, flags) == 0.75


def test_latency_stats_basic():
    stats = metrics.latency_stats([100, 200, 300, 400, 500])
    assert stats["mean_ms"] == 300
    assert stats["p50_ms"] == 300


def test_latency_stats_empty():
    stats = metrics.latency_stats([])
    assert stats == {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}


def test_token_cost_stats():
    stats = metrics.token_cost_stats([100, 200], [10, 20])
    assert stats["avg_prompt_tokens"] == 150
    assert stats["avg_completion_tokens"] == 15
    assert stats["avg_total_tokens"] == 165


def test_token_cost_stats_empty():
    stats = metrics.token_cost_stats([], [])
    assert stats["avg_prompt_tokens"] == 0


def test_failure_examples_only_includes_mismatches():
    preds = ["A", "B", "C"]
    truth = ["A", "X", "C"]
    ids = ["t1", "t2", "t3"]
    failures = metrics.failure_examples(preds, truth, ids)
    assert len(failures) == 1
    assert failures[0] == {"ticket_id": "t2", "predicted": "B", "actual": "X"}


def test_failure_examples_respects_limit():
    preds = ["A", "B", "C", "D"]
    truth = ["X", "Y", "Z", "W"]
    ids = ["t1", "t2", "t3", "t4"]
    failures = metrics.failure_examples(preds, truth, ids, limit=2)
    assert len(failures) == 2
