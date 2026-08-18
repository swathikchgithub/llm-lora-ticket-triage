"""Latency-only benchmark for the LoRA adapter merged into the base model.

The main eval harness (evaluate.py) benchmarks the fine-tuned model with
the adapter applied but NOT merged - every forward pass pays the extra
cost of the LoRA low-rank matrix multiplications on top of the base
model's own computation. Production deployments normally call
merge_and_unload() to fold the adapter into the base weights once,
offline, so serving pays zero LoRA overhead. This script isolates that
effect: merging is mathematically a no-op on the model's outputs (same
weights, just added together ahead of time instead of at each forward
pass), so we only need to re-measure latency, not accuracy.

Merging is done in bf16, not 4-bit: PEFT's merge support for 4-bit
(bitsandbytes NF4) models requires dequantizing each adapted layer and is
precision-lossy / version-fragile, so a bf16 merge is the standard,
reliable way to do this - which also matches how a production pipeline
would typically merge before any optional re-quantization step.

Usage:
    python eval/measure_merged_latency.py
"""

import argparse
import json

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import metrics
import model_runner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter-path", default="checkpoints/qwen2.5-1.5b-ticket-lora")
    parser.add_argument("--test-file", default="data/test.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--out", default="results/finetuned_model/merged_latency.json")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model in bf16 (unquantized) and merging adapter...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model = PeftModel.from_pretrained(base, args.adapter_path)
    model = model.merge_and_unload()
    model.eval()

    with open(args.test_file) as f:
        test_examples = [json.loads(line) for line in f]

    latencies, prompt_toks, completion_toks = [], [], []
    for ex in test_examples:
        subject = ex["messages"][1]["content"].split("Description:")[0].replace("Subject:", "").strip()
        description = ex["messages"][1]["content"].split("Description:", 1)[1].strip()
        messages = model_runner.build_prompt_messages(subject, description, use_few_shot=False)
        _, latency_ms, p_tok, c_tok = model_runner.generate(
            model, tokenizer, messages, do_sample=False, max_new_tokens=args.max_new_tokens,
        )
        latencies.append(latency_ms)
        prompt_toks.append(p_tok)
        completion_toks.append(c_tok)

    stats = {
        "note": "merged adapter, bf16 (not 4-bit) - see module docstring for why",
        "latency": metrics.latency_stats(latencies),
        "token_cost": metrics.token_cost_stats(prompt_toks, completion_toks),
    }
    print(json.dumps(stats, indent=2))
    with open(args.out, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
