from collections import Counter

from generate_dataset import CATEGORIES, PRIORITIES, generate


def test_generate_returns_requested_count():
    records = generate(n=200, seed=1)
    assert len(records) == 200


def test_generate_is_deterministic_given_seed():
    a = generate(n=100, seed=7)
    b = generate(n=100, seed=7)
    assert a == b


def test_generate_different_seeds_produce_different_data():
    a = generate(n=100, seed=1)
    b = generate(n=100, seed=2)
    assert a != b


def test_all_categories_represented():
    records = generate(n=300, seed=42)
    seen = {r["label"]["category"] for r in records}
    assert seen == set(CATEGORIES)


def test_all_priorities_are_valid():
    records = generate(n=300, seed=42)
    for r in records:
        assert r["label"]["priority"] in PRIORITIES


def test_ticket_ids_are_unique():
    records = generate(n=300, seed=42)
    ids = [r["ticket_id"] for r in records]
    assert len(ids) == len(set(ids))


def test_every_record_has_required_fields():
    records = generate(n=50, seed=3)
    for r in records:
        assert "ticket_id" in r
        assert "subject" in r and r["subject"]
        assert "description" in r and r["description"]
        assert set(r["label"].keys()) == {"category", "priority", "severity_justification"}
        assert r["label"]["severity_justification"]


def test_category_distribution_is_roughly_balanced():
    records = generate(n=900, seed=42)
    counts = Counter(r["label"]["category"] for r in records)
    # regular (non-ambiguous) tickets are split evenly across 5 categories;
    # ambiguous tickets (~12%) can skew this, so allow a wide-ish band.
    for cat in CATEGORIES:
        assert counts[cat] > 900 / len(CATEGORIES) * 0.5
