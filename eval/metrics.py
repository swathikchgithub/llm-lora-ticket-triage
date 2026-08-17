"""Pure metric functions for the base-vs-fine-tuned comparison.

No GPU, no model loading here on purpose — this module is the part of the
eval harness that can be unit tested deterministically (see tests/).
"""

from collections import Counter, defaultdict


def accuracy(predictions, ground_truths):
    """Overall exact-match accuracy. Time: O(n), Space: O(1)."""
    if not predictions:
        return 0.0
    correct = sum(p == g for p, g in zip(predictions, ground_truths))
    return correct / len(predictions)


def per_category_accuracy(predictions, ground_truths):
    """Accuracy broken out by the TRUE category (recall per class).

    Time: O(n), Space: O(k) for k classes.
    """
    correct = Counter()
    total = Counter()
    for pred, truth in zip(predictions, ground_truths):
        total[truth] += 1
        if pred == truth:
            correct[truth] += 1
    return {cat: correct[cat] / total[cat] for cat in total}


def confusion_matrix(predictions, ground_truths, labels):
    """Return {true_label: {pred_label: count}}. Time: O(n), Space: O(k^2)."""
    matrix = {t: {p: 0 for p in labels} for t in labels}
    for pred, truth in zip(predictions, ground_truths):
        if truth in matrix and pred in matrix[truth]:
            matrix[truth][pred] += 1
    return matrix


def top_confusions(predictions, ground_truths, top_n=5):
    """Most common (true, predicted) mismatch pairs, most frequent first.

    Time: O(n + m log m) where m = number of distinct mismatch pairs.
    """
    pairs = Counter()
    for pred, truth in zip(predictions, ground_truths):
        if pred != truth:
            pairs[(truth, pred)] += 1
    return pairs.most_common(top_n)


def consistency_rate(repeated_predictions):
    """Self-consistency: fraction of inputs where all repeated runs agree.

    `repeated_predictions` is {ticket_id: [pred_run1, pred_run2, pred_run3]}.
    A model that reduces variance under repeated sampling is more reliable
    in production even at equal average accuracy — this is what fine-tuning
    should improve versus a prompted base model, since the base model's
    output is more sensitive to sampling noise around the instruction.

    Time: O(n * r) for n inputs and r repeats, Space: O(1) per input.
    """
    if not repeated_predictions:
        return 0.0
    consistent = sum(
        1 for preds in repeated_predictions.values() if len(set(preds)) == 1
    )
    return consistent / len(repeated_predictions)


def json_parse_success_rate(raw_outputs, parsed_flags):
    """Fraction of raw generations that were valid, schema-conforming JSON.

    A prompted base model without fine-tuning is more prone to adding
    preamble text, markdown fences, or malformed JSON than a model
    fine-tuned specifically to emit this exact format — this metric makes
    that failure mode visible and countable rather than anecdotal.
    """
    if not raw_outputs:
        return 0.0
    return sum(parsed_flags) / len(raw_outputs)


def latency_stats(latencies_ms):
    """Basic latency distribution stats. Time: O(n log n) for the sort."""
    if not latencies_ms:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
    s = sorted(latencies_ms)
    n = len(s)
    return {
        "mean_ms": sum(s) / n,
        "p50_ms": s[n // 2],
        "p95_ms": s[min(n - 1, int(n * 0.95))],
    }


def token_cost_stats(prompt_tokens, completion_tokens):
    """Average prompt/completion token counts — the direct driver of
    per-inference cost. A fine-tuned model needs no few-shot examples or
    long instructions at inference time, so prompt_tokens should be much
    lower than the prompted-base-model baseline.
    """
    n = len(prompt_tokens)
    if n == 0:
        return {"avg_prompt_tokens": 0, "avg_completion_tokens": 0, "avg_total_tokens": 0}
    return {
        "avg_prompt_tokens": sum(prompt_tokens) / n,
        "avg_completion_tokens": sum(completion_tokens) / n,
        "avg_total_tokens": (sum(prompt_tokens) + sum(completion_tokens)) / n,
    }


def failure_examples(predictions, ground_truths, ticket_ids, limit=20):
    """Collect misclassified ticket ids for qualitative failure review."""
    out = []
    for pred, truth, tid in zip(predictions, ground_truths, ticket_ids):
        if pred != truth:
            out.append({"ticket_id": tid, "predicted": pred, "actual": truth})
        if len(out) >= limit:
            break
    return out
