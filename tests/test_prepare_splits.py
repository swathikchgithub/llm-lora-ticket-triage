import json

from generate_dataset import generate
from prepare_splits import stratified_split, to_chat_example


def _make_records(n=300, seed=42):
    return generate(n=n, seed=seed)


def test_split_sizes_match_ratios_within_rounding():
    records = _make_records(500)
    train, val, test = stratified_split(records, (0.8, 0.1, 0.1), seed=0)
    assert len(train) + len(val) + len(test) == len(records)
    assert abs(len(train) - 400) <= 10
    assert abs(len(val) - 50) <= 10
    assert abs(len(test) - 50) <= 10


def test_no_overlap_between_splits():
    records = _make_records(300)
    train, val, test = stratified_split(records, (0.8, 0.1, 0.1), seed=0)
    train_ids = {r["ticket_id"] for r in train}
    val_ids = {r["ticket_id"] for r in val}
    test_ids = {r["ticket_id"] for r in test}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)


def test_stratification_keeps_every_category_in_every_split():
    records = _make_records(500)
    train, val, test = stratified_split(records, (0.8, 0.1, 0.1), seed=0)
    categories = {r["label"]["category"] for r in records}
    for split in (train, val, test):
        split_categories = {r["label"]["category"] for r in split}
        assert split_categories == categories


def test_split_is_deterministic_given_seed():
    records = _make_records(300)
    a = stratified_split(records, (0.8, 0.1, 0.1), seed=5)
    b = stratified_split(records, (0.8, 0.1, 0.1), seed=5)
    assert [r["ticket_id"] for r in a[0]] == [r["ticket_id"] for r in b[0]]


def test_to_chat_example_has_valid_json_assistant_content():
    record = _make_records(1)[0]
    chat = to_chat_example(record)
    assert len(chat["messages"]) == 3
    assert chat["messages"][0]["role"] == "system"
    assert chat["messages"][1]["role"] == "user"
    assert chat["messages"][2]["role"] == "assistant"
    parsed = json.loads(chat["messages"][2]["content"])
    assert parsed == record["label"]
    assert chat["category"] == record["label"]["category"]
    assert chat["priority"] == record["label"]["priority"]
