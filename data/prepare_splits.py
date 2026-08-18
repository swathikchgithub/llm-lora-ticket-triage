"""Split the raw ticket dataset into train/val/test and format for SFT.

Stratifies by category so each split has the same ~class balance (critical
for a 5-way classification task — an unstratified split can starve val/test
of a whole category by chance). Output is in HF chat-format ("messages")
matching the Qwen2.5-Instruct chat template, ready for TRL's SFTTrainer.

Usage:
    python data/prepare_splits.py --in data/tickets_all.jsonl --outdir data/
"""

import argparse
import json
import random
from collections import defaultdict

SYSTEM_PROMPT = (
    "You are an IT support ticket triage assistant. Given a ticket subject "
    "and description, classify it and respond with ONLY a JSON object with "
    "exactly these keys: \"category\" (one of: Network, Access/Password, "
    "Hardware, Software, Billing), \"priority\" (one of: P1-Critical, "
    "P2-High, P3-Medium, P4-Low), and \"severity_justification\" (one "
    "sentence explaining the priority based on business impact). "
    "No extra text, no markdown, just the JSON object."
)


def to_chat_example(record):
    user_content = f"Subject: {record['subject']}\n\nDescription: {record['description']}"
    assistant_content = json.dumps(record["label"], ensure_ascii=False)
    return {
        "ticket_id": record["ticket_id"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        # kept flat for the eval harness, which needs the ground truth
        # without re-parsing the assistant JSON string every time.
        "category": record["label"]["category"],
        "priority": record["label"]["priority"],
    }


def stratified_split(records, ratios, seed):
    """Time: O(n log n) for the shuffle; Space: O(n).

    Splits by *scenario group* (records sharing the same category + subject
    text) rather than by individual record. The generator draws from a
    small, fixed pool of scenario templates per category and only
    randomizes names/departments/timing around them, so a row-level random
    split lets near-identical phrasing leak into both train and test - the
    model then "generalizes" by memorizing template wording instead of
    learning the task. Grouping by subject keeps each scenario wholly on
    one side of the split, so test measures unseen phrasing.
    """
    assert abs(sum(ratios) - 1.0) < 1e-6
    rng = random.Random(seed)

    groups_by_category = defaultdict(lambda: defaultdict(list))
    for r in records:
        groups_by_category[r["label"]["category"]][r["subject"]].append(r)

    train, val, test = [], [], []
    for category, groups in groups_by_category.items():
        group_keys = list(groups.keys())
        rng.shuffle(group_keys)
        n_groups = len(group_keys)

        # Small group counts need floors, not pure ratio rounding, so val
        # and test each keep at least one held-out scenario per category.
        n_test = max(1, round(n_groups * ratios[2]))
        n_val = max(1, round(n_groups * ratios[1]))
        n_test, n_val = min(n_test, n_groups - 1), min(n_val, max(0, n_groups - n_test - 1))
        n_train = n_groups - n_val - n_test

        test_keys = group_keys[:n_test]
        val_keys = group_keys[n_test:n_test + n_val]
        train_keys = group_keys[n_test + n_val:n_test + n_val + n_train]

        for k in train_keys:
            train += groups[k]
        for k in val_keys:
            val += groups[k]
        for k in test_keys:
            test += groups[k]

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="infile", default="data/tickets_all.jsonl")
    parser.add_argument("--outdir", default="data/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    args = parser.parse_args()

    with open(args.infile) as f:
        records = [json.loads(line) for line in f]

    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    train, val, test = stratified_split(records, ratios, args.seed)

    for name, split in [("train", train), ("val", val), ("test", test)]:
        path = f"{args.outdir.rstrip('/')}/{name}.jsonl"
        with open(path, "w") as f:
            for r in split:
                f.write(json.dumps(to_chat_example(r), ensure_ascii=False) + "\n")
        print(f"{name}: {len(split)} examples -> {path}")

    def dist(split):
        c = defaultdict(int)
        for r in split:
            c[r["label"]["category"]] += 1
        return dict(c)

    print("\nCategory balance check:")
    print("  train:", dist(train))
    print("  val:  ", dist(val))
    print("  test: ", dist(test))


if __name__ == "__main__":
    main()
