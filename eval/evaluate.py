"""Base-vs-fine-tuned comparison harness. Run on a GPU after training.

For each model config (prompted base model, and the LoRA fine-tuned
model), this script:
  1. Runs greedy generation over the held-out test set for accuracy,
     confusion matrix, JSON-validity rate, latency, and token cost.
  2. Runs N sampled generations per ticket (temperature=0.7) to measure
     self-consistency — does fine-tuning reduce output variance?
  3. Writes raw predictions (for eval/llm_judge.py) and a comparison report.

Usage:
    python eval/evaluate.py \
        --base-model Qwen/Qwen2.5-1.5B-Instruct \
        --adapter-path checkpoints/qwen2.5-1.5b-ticket-lora \
        --test-file data/test.jsonl \
        --out results/ \
        --consistency-runs 3
"""

import argparse
import json
import os
from collections import defaultdict

import torch

import metrics
import model_runner

CATEGORIES = ["Network", "Access/Password", "Hardware", "Software", "Billing"]


def run_config(model, tokenizer, test_examples, use_few_shot, consistency_runs, max_new_tokens):
    """Evaluate one model config against the full test set.

    Time: O(n * (1 + r)) generations for n tickets and r consistency runs
    each — dominated by model.generate() calls, not by anything in this
    function's own control flow.
    """
    predictions, raw_outputs = [], []
    predicted_categories, predicted_priorities = [], []
    true_categories, true_priorities, ticket_ids = [], [], []
    parse_flags, latencies, prompt_toks, completion_toks = [], [], [], []
    repeated_category_preds = defaultdict(list)

    for ex in test_examples:
        subject = ex["messages"][1]["content"].split("Description:")[0].replace("Subject:", "").strip()
        description = ex["messages"][1]["content"].split("Description:", 1)[1].strip()
        messages = model_runner.build_prompt_messages(subject, description, use_few_shot)

        raw, latency_ms, p_tok, c_tok = model_runner.generate(
            model, tokenizer, messages, do_sample=False, max_new_tokens=max_new_tokens,
        )
        parsed, ok = model_runner.parse_prediction(raw)

        ticket_ids.append(ex["ticket_id"])
        true_categories.append(ex["category"])
        true_priorities.append(ex["priority"])
        parse_flags.append(ok)
        latencies.append(latency_ms)
        prompt_toks.append(p_tok)
        completion_toks.append(c_tok)
        raw_outputs.append(raw)

        pred_cat = parsed.get("category") if ok else None
        pred_prio = parsed.get("priority") if ok else None
        pred_just = parsed.get("severity_justification") if ok else None
        predicted_categories.append(pred_cat)
        predicted_priorities.append(pred_prio)
        predictions.append({
            "ticket_id": ex["ticket_id"],
            "true_category": ex["category"],
            "true_priority": ex["priority"],
            "pred_category": pred_cat,
            "pred_priority": pred_prio,
            "pred_justification": pred_just,
            "true_justification": ex["messages"][2]["content"],
            "raw_output": raw,
            "json_valid": ok,
            "latency_ms": latency_ms,
            "prompt_tokens": p_tok,
            "completion_tokens": c_tok,
        })

        for _ in range(consistency_runs):
            raw_r, _, _, _ = model_runner.generate(
                model, tokenizer, messages, do_sample=True, temperature=0.7,
                max_new_tokens=max_new_tokens,
            )
            parsed_r, ok_r = model_runner.parse_prediction(raw_r)
            repeated_category_preds[ex["ticket_id"]].append(
                parsed_r.get("category") if ok_r else "PARSE_ERROR"
            )

    report = {
        "n_examples": len(test_examples),
        "category_accuracy": metrics.accuracy(predicted_categories, true_categories),
        "priority_accuracy": metrics.accuracy(predicted_priorities, true_priorities),
        "per_category_accuracy": metrics.per_category_accuracy(predicted_categories, true_categories),
        "confusion_matrix": metrics.confusion_matrix(predicted_categories, true_categories, CATEGORIES),
        "top_confusions": metrics.top_confusions(predicted_categories, true_categories),
        "json_parse_success_rate": metrics.json_parse_success_rate(raw_outputs, parse_flags),
        "consistency_rate": metrics.consistency_rate(repeated_category_preds),
        "latency": metrics.latency_stats(latencies),
        "token_cost": metrics.token_cost_stats(prompt_toks, completion_toks),
        "failure_examples": metrics.failure_examples(predicted_categories, true_categories, ticket_ids),
    }
    return report, predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter-path", default="checkpoints/qwen2.5-1.5b-ticket-lora")
    parser.add_argument("--test-file", default="data/test.jsonl")
    parser.add_argument("--out", default="results/")
    parser.add_argument("--consistency-runs", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    args = parser.parse_args()

    with open(args.test_file) as f:
        test_examples = [json.loads(line) for line in f]

    os.makedirs(f"{args.out}/base_model", exist_ok=True)
    os.makedirs(f"{args.out}/finetuned_model", exist_ok=True)

    print(f"Evaluating PROMPTED BASE MODEL ({args.base_model}, 3-shot)...")
    model, tokenizer = model_runner.load_base_model(args.base_model)
    base_report, base_preds = run_config(
        model, tokenizer, test_examples, use_few_shot=True,
        consistency_runs=args.consistency_runs, max_new_tokens=args.max_new_tokens,
    )
    del model
    torch.cuda.empty_cache()

    print(f"Evaluating FINE-TUNED MODEL ({args.adapter_path}, 0-shot)...")
    model, tokenizer = model_runner.load_finetuned_model(args.base_model, args.adapter_path)
    ft_report, ft_preds = run_config(
        model, tokenizer, test_examples, use_few_shot=False,
        consistency_runs=args.consistency_runs, max_new_tokens=args.max_new_tokens,
    )
    del model
    torch.cuda.empty_cache()

    with open(f"{args.out}/base_model/predictions.jsonl", "w") as f:
        for p in base_preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(f"{args.out}/finetuned_model/predictions.jsonl", "w") as f:
        for p in ft_preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    comparison = {"base_model_prompted": base_report, "finetuned_model": ft_report}
    with open(f"{args.out}/comparison_report.json", "w") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    write_markdown_summary(comparison, f"{args.out}/comparison_report.md")
    print(f"\nDone. Reports written to {args.out}/comparison_report.{{json,md}}")


def write_markdown_summary(comparison, path):
    b, f_ = comparison["base_model_prompted"], comparison["finetuned_model"]
    lines = [
        "# Base vs. Fine-Tuned: Evaluation Report\n",
        "| Metric | Prompted Base (3-shot) | Fine-Tuned (0-shot) |",
        "|---|---|---|",
        f"| Category accuracy | {b['category_accuracy']:.1%} | {f_['category_accuracy']:.1%} |",
        f"| Priority accuracy | {b['priority_accuracy']:.1%} | {f_['priority_accuracy']:.1%} |",
        f"| JSON parse success rate | {b['json_parse_success_rate']:.1%} | {f_['json_parse_success_rate']:.1%} |",
        f"| Consistency rate (repeat agreement) | {b['consistency_rate']:.1%} | {f_['consistency_rate']:.1%} |",
        f"| Avg prompt tokens | {b['token_cost']['avg_prompt_tokens']:.0f} | {f_['token_cost']['avg_prompt_tokens']:.0f} |",
        f"| Avg total tokens/inference | {b['token_cost']['avg_total_tokens']:.0f} | {f_['token_cost']['avg_total_tokens']:.0f} |",
        f"| p50 latency (ms) | {b['latency']['p50_ms']:.0f} | {f_['latency']['p50_ms']:.0f} |",
        f"| p95 latency (ms) | {b['latency']['p95_ms']:.0f} | {f_['latency']['p95_ms']:.0f} |",
        "\n## Per-category accuracy\n",
        "| Category | Prompted Base | Fine-Tuned |",
        "|---|---|---|",
    ]
    for cat in CATEGORIES:
        lines.append(
            f"| {cat} | {b['per_category_accuracy'].get(cat, 0):.1%} | "
            f"{f_['per_category_accuracy'].get(cat, 0):.1%} |"
        )
    lines.append("\n## Top confusions (true -> predicted)\n")
    lines.append("**Base model:** " + ", ".join(f"{t}->{p} ({c}x)" for (t, p), c in b["top_confusions"]))
    lines.append("\n**Fine-tuned model:** " + ", ".join(f"{t}->{p} ({c}x)" for (t, p), c in f_["top_confusions"]))
    with open(path, "w") as fp:
        fp.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
