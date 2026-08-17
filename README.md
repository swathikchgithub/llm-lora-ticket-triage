# IT Ticket Triage: LoRA Fine-Tuning vs. Prompted Base Model

A small, complete, rigorously-evaluated LoRA fine-tuning project: fine-tune
**Qwen2.5-1.5B-Instruct** to classify IT support tickets (category +
priority + severity justification) and **prove**, not assume, that
fine-tuning beats a well-prompted base model — across accuracy,
consistency, cost, and qualitative failure modes.

> **Status:** dataset, training script, and evaluation harness are built
> and tested. The actual GPU training run happens on a rented RunPod pod
> (see [docs/RUNBOOK_RUNPOD.md](docs/RUNBOOK_RUNPOD.md)) — the **Results**
> section below is a template to be filled in with real numbers after that
> run, not fabricated numbers. Every script referenced here has been
> written, syntax-checked, and (where GPU-independent) unit tested.

## Why this project

I have production experience in LLM **evaluation** — I built AutoEval (an
LLM-as-Judge framework) and Evalify (an open-source multi-model eval
platform) — but no hands-on fine-tuning experience. This project pairs a
new skill (LoRA fine-tuning) with my strongest existing skill (rigorous
evaluation methodology), on a task that maps directly to my ITSM/ServiceNow
background: triaging support tickets into category and priority.

## The task

Classify an IT support ticket (subject + description) into:
- **category**: `Network`, `Access/Password`, `Hardware`, `Software`, `Billing`
- **priority**: `P1-Critical`, `P2-High`, `P3-Medium`, `P4-Low`
- **severity_justification**: one sentence explaining the priority, grounded
  in the ticket's actual business impact

The model must output a single JSON object matching this schema — a
realistic enterprise "structured extraction + classification" task, not a
toy.

## Why LoRA, not full fine-tuning

| | Full fine-tuning | LoRA |
|---|---|---|
| Trainable params | 100% of the model (1.5B) | ~0.5-2% (small adapter matrices) |
| GPU memory | Needs to hold full optimizer state for every weight (3-4x model size in VRAM) | Frozen base + tiny adapter; fits on a single consumer GPU |
| Training cost/time | Hours to days, expensive multi-GPU setups | Minutes on one GPU |
| Risk of catastrophic forgetting | Higher — every weight can drift | Lower — base weights frozen, only a narrow update path |
| Artifact to ship/version | A full new multi-GB model checkpoint per task | A ~10-30MB adapter; swap adapters per task on one base model |
| Serving | One model per fine-tune | One base model + many hot-swappable adapters |

**This is also, in practice, what most companies mean when they say "we
fine-tuned our LLM."** Full fine-tuning of even a 1-8B model is rarely worth
the infra cost when the task is narrow (format compliance, classification,
tone) rather than teaching genuinely new knowledge or reasoning ability —
which is exactly the ticket-triage task here. QLoRA (LoRA + 4-bit
quantization of the frozen base, via bitsandbytes) pushes this further: it's
the difference between needing an A100 and needing a $0.30/hr RTX 4090.

## Model choice

**Qwen2.5-1.5B-Instruct** — chosen over Llama 3.2 1B/3B and Phi-3-mini for
this task because: strong instruction-following at 1.5B (competitive with
larger models on structured-output tasks), first-class `transformers` +
`peft` + `bitsandbytes` support, no gated-license friction on HF Hub
(unlike Llama), and a training/eval loop fast and cheap enough to iterate
on in a single sitting.

## Dataset

`data/generate_dataset.py` produces **900 synthetic tickets** (I don't have
external access to real ServiceNow data), stratified across the 5
categories with a realistic priority skew (few P1s, mostly P2-P4). ~12% are
**deliberately ambiguous/cross-category** (e.g. "VPN login rejects my
password" — Network or Access?) so the task has genuine failure modes for
the eval harness to surface, rather than being trivially separable.

Split (stratified by category, `data/prepare_splits.py`):
- `train.jsonl`: 718 examples
- `val.jsonl`: 88 examples
- `test.jsonl`: 94 examples (held out, never seen during training)

Formatted as HF chat-format (`messages: [system, user, assistant]`) matching
the Qwen2.5-Instruct chat template, with the assistant turn as the target
JSON.

## Training setup

`scripts/train_lora.py` — QLoRA fine-tuning:

| Hyperparameter | Value | Why |
|---|---|---|
| `lora_r` (rank) | 16 | Task is narrow (format + classification, not new knowledge); r=16 gives enough capacity without inviting overfitting on 718 examples |
| `lora_alpha` | 32 | Standard `alpha = 2*r` rule of thumb (Hu et al. 2021) — keeps update magnitude sane at this rank |
| `target_modules` | all attention + MLP projections | Lets the adapter adjust both how the model reads ticket text and how it maps that to the output format |
| `learning_rate` | 2e-4 | LoRA adapters start near-zero and need a higher LR than full fine-tuning (1e-5–5e-5) to move meaningfully in 3 epochs |
| `epochs` | 3 | Small, narrow dataset converges fast; watched via `eval_loss` to catch overfitting on synthetic template phrasing |
| `effective batch size` | 16 (4 × grad-accum 4) | Fits comfortably in a single 24GB GPU with 4-bit base weights |
| quantization | NF4 4-bit (bitsandbytes) | Frozen base in 4-bit, LoRA compute in bf16 — the standard QLoRA recipe |
| loss masking | prompt-masked (`DataCollatorForCompletionOnlyLM`) | Only trains on the assistant's JSON completion, not on re-predicting the ticket text |

Run on RunPod (RTX 4090, ~$0.30-0.50/hr) — see
[docs/RUNBOOK_RUNPOD.md](docs/RUNBOOK_RUNPOD.md) for the exact commands.
Monitored via TensorBoard (`train/loss`, `eval/loss`).

## Evaluation methodology — the differentiating part

`eval/evaluate.py` compares **the fine-tuned model (0-shot)** against
**the base model with a strong 3-shot prompt** (a realistic baseline, not a
strawman) on the held-out test set, across multiple dimensions:

1. **Classification accuracy**, overall and **per-category**, plus a
   confusion matrix and top confusion pairs (`eval/metrics.py`) — where
   does each model actually fail, and on which categories?
2. **Consistency**: each ticket is re-generated 3x at temperature 0.7;
   `consistency_rate` measures how often all 3 runs agree. A fine-tuned
   model should be less sensitive to sampling noise than a prompted one.
3. **JSON validity rate**: fraction of generations that parse as
   schema-conforming JSON — the prompted base model is more prone to
   preamble/markdown-fence chatter than a model tuned specifically to emit
   this format.
4. **Latency & token cost**: prompt tokens, completion tokens, p50/p95
   latency. The fine-tuned model needs no few-shot examples at inference
   time, which should show up directly as lower prompt-token cost.
5. **LLM-as-Judge on the free-text justification** (`eval/llm_judge.py`):
   exact-match accuracy doesn't cover free text, so the justification is
   graded by Claude (a separate judge model, avoiding self-preference bias)
   against a grounded 3-dimension rubric (relevance, consistency,
   specificity) — the same methodology as AutoEval. **Critically, the judge
   is calibrated first**: `data/human_calibration_set.jsonl` has 20
   hand-labeled examples; `eval/llm_judge.py calibrate` checks judge/human
   agreement *before* any full-set judge scores are trusted. Skipping this
   step is the most common mistake in LLM-as-judge setups.
6. **Qualitative failure mode analysis**: the harness dumps
   misclassified examples and top confusion pairs per model so failures can
   be inspected, not just counted.

## Results

*(Fill in after running `scripts/train_lora.py` then `eval/evaluate.py` on
RunPod — see the runbook. Numbers below are placeholders showing the report
shape, not real measurements.)*

| Metric | Prompted Base (3-shot) | Fine-Tuned (0-shot) |
|---|---|---|
| Category accuracy | TBD | TBD |
| Priority accuracy | TBD | TBD |
| JSON parse success rate | TBD | TBD |
| Consistency rate | TBD | TBD |
| Avg prompt tokens | TBD | TBD |
| p50 latency (ms) | TBD | TBD |
| LLM-judge avg justification score | TBD | TBD |

Full report (per-category breakdown, confusion matrix, failure examples)
lands in `results/comparison_report.md` after running `eval/evaluate.py`.

## Repo structure

```
data/
  generate_dataset.py       synthetic ticket generator
  prepare_splits.py         stratified train/val/test split + chat formatting
  human_calibration_set.jsonl  hand-labeled examples for judge calibration
  train.jsonl / val.jsonl / test.jsonl
scripts/
  train_lora.py              QLoRA fine-tuning script
eval/
  metrics.py                 pure metric functions (unit tested)
  model_runner.py             model loading + generation (GPU required)
  evaluate.py                 base-vs-fine-tuned comparison harness
  llm_judge.py                LLM-as-judge scoring + calibration
docs/
  RUNBOOK_RUNPOD.md           exact steps to train on a rented GPU
tests/
  test_generate_dataset.py, test_prepare_splits.py, test_metrics.py
results/                      populated by eval/evaluate.py
```

## Running it

```bash
pip install -r requirements.txt

# 1. Generate + split data (no GPU needed, already done in this repo)
python data/generate_dataset.py --n 900 --seed 42 --out data/tickets_all.jsonl
python data/prepare_splits.py --in data/tickets_all.jsonl --outdir data/

# 2. Train (GPU required — see docs/RUNBOOK_RUNPOD.md)
python scripts/train_lora.py

# 3. Evaluate (GPU required)
python eval/evaluate.py

# 4. Calibrate + run the LLM judge (needs ANTHROPIC_API_KEY, no GPU)
python eval/llm_judge.py calibrate
python eval/llm_judge.py score \
    --predictions-file results/finetuned_model/predictions.jsonl \
    --out results/finetuned_model/judge_scores.jsonl

# 5. Run the test suite (no GPU needed)
pytest tests/ -v
```

## How this differs from production-scale fine-tuning (e.g., at PayPal)

Being upfront about the gap between this demo and real production
fine-tuning work:

- **Data scale & provenance**: 900 synthetic examples here vs. millions of
  real, PII-scrubbed, legally-reviewed production tickets. Production data
  pipelines need de-duplication, PII redaction, label QA at scale, and
  ongoing drift monitoring — none of which a synthetic generator faces.
- **Distributed training**: a 1.5B model on one GPU here; production
  fine-tunes of larger models use multi-GPU/multi-node training (FSDP,
  DeepSpeed ZeRO), which introduces its own engineering surface
  (sharding, communication overhead, checkpoint resharding).
- **Evaluation rigor at scale**: this project's held-out test set is 94
  examples. Production evaluation needs much larger, continuously-refreshed
  golden sets, stratified by real traffic distribution, plus offline eval
  gates *and* online A/B testing before a fine-tuned model ever serves
  live traffic.
- **Model lifecycle**: no versioning, rollback, or canary deployment system
  here — just a saved adapter. Production needs a model registry, shadow
  traffic comparison, automated rollback on regression, and monitoring for
  live drift between the eval-time distribution and production traffic.
- **Human labeling scale**: the judge-calibration set here is 20 examples I
  hand-labeled myself. Production LLM-judge calibration typically uses
  larger, multi-annotator human-labeled sets with inter-annotator agreement
  measurement, to catch judge bias that a single labeler could miss.
- **Security/compliance**: no real PII, access control, or audit-logging
  concerns here since the data is synthetic. Production fine-tuning on real
  ticket data would need data governance sign-off, redaction pipelines, and
  audit trails on who accessed training data.

## Interview summary (spoken, 3-4 sentences)

"I built an end-to-end LoRA fine-tuning project on a small open-source
model — Qwen2.5-1.5B — to classify IT support tickets into category and
priority, going from raw synthetic data through QLoRA training on a rented
GPU to a full evaluation harness. The part I actually care most about is
the evaluation: I didn't just report accuracy, I compared the fine-tuned
model against a strongly-prompted base model on accuracy per category,
output consistency under repeated sampling, token cost, and JSON-validity
rate, plus an LLM-as-judge score for the free-text justifications —
calibrated against a hand-labeled set first, the same way I've done judge
calibration in production eval work. The goal was to prove fine-tuning
helped, with numbers, not just assume it — and to be explicit about
exactly where this differs from production-scale fine-tuning, like dataset
scale, distributed training, and the online A/B testing you'd need before
shipping a fine-tuned model to real traffic."

## License

MIT (or your preference — update before publishing).
