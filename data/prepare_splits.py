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
    """Time: O(n log n) for the shuffle; Space: O(n)."""
    assert abs(sum(ratios) - 1.0) < 1e-6
    rng = random.Random(seed)
    by_category = defaultdict(list)
    for r in records:
        by_category[r["label"]["category"]].append(r)

    train, val, test = [], [], []
    for category, items in by_category.items():
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        train += items[:n_train]
        val += items[n_train:n_train + n_val]
        test += items[n_train + n_val:]

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
