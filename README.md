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

Trained on an RTX A4000 (RunPod), 3 epochs, ~7 min wall-clock, well under
$1. Numbers below are real, from `results/comparison_report.md` and
`results/finetuned_model/merged_latency.json`.

**Two methodology bugs caught and fixed along the way — this is the part
that matters more than any single number below:**

1. The first version of this dataset's train/test split was stratified by
   category only, not by the underlying scenario template. Since the
   generator draws from a small, fixed pool of ~40 phrasings per category,
   a row-level random split let near-duplicate phrasing (same template,
   different name/dept) leak into both train and test — inflating the
   first fine-tuned run to a suspicious 100% category accuracy. Caught by
   checking subject-line overlap between splits (89% of test examples
   shared verbatim phrasing with training examples); fixed by holding out
   entire scenario templates instead (`data/prepare_splits.py`).
2. Fixing the split alone wasn't enough — the already-trained checkpoint
   had been trained on the *old* split, which could still contain sibling
   examples from templates the new split now holds out for test. Fixing
   the data doesn't retroactively un-leak a model that already trained on
   the leaky version. Caught this before publishing results, not after,
   and retrained from scratch on the corrected split.

Numbers below are from that retrained checkpoint evaluated on the
leakage-free test set — the first evaluation run where both the data
split and the model itself are clean.

| Metric | Prompted Base (3-shot) | Fine-Tuned (0-shot, unmerged) | Fine-Tuned (0-shot, **merged**) |
|---|---|---|---|
| Category accuracy | 49.5% | 84.2% | *(same weights — unaffected by merge)* |
| Priority accuracy | 21.1% | 24.2% | *(same weights — unaffected by merge)* |
| JSON parse success rate | 100.0% | 100.0% | — |
| Consistency rate (repeat agreement) | 83.2% | 71.6% | — |
| Avg total tokens/inference | 561 | 221 | 219 |
| p50 latency (ms) | 2990 | 3910 | **1086** |
| p95 latency (ms) | 3672 | 4655 | **1246** |

### What this actually shows

- **Category classification: a real, generalizing win.** 49.5% → 84.2% on
  genuinely unseen ticket phrasing (not memorized templates). Lower than
  the leaky run's 91.6%, which is exactly the point — this number is
  trustworthy in a way the earlier one wasn't. This is the headline
  result.
- **Priority classification: fine-tuning didn't help.** 21.1% → 24.2% —
  both sit right at the 25% random-chance baseline for 4 classes. My
  read: priority requires implicit reasoning about business impact from
  the narrative, not surface lexical cues category classification can
  lean on. With only ~40 scenario templates, 3 epochs, and rank-16 LoRA,
  the model most likely memorized priority-per-template during training
  rather than learning the underlying judgment, so it doesn't transfer to
  unseen phrasing. This is a real limitation, not a rounding error — see
  [Failure modes](#failure-modes-worth-naming) below.
- **Consistency: a possible real drop, not fully confirmed.** Fine-tuned
  came in 11.6 points below base (71.6% vs. 83.2%). Repeated runs of this
  benchmark on identical settings have shown ~7-8 points of pure sampling
  noise at this sample size (95 tickets × 3 repeats), so this gap is
  larger than typical noise but not far enough outside it to claim
  confidently without another repeated trial. Stated honestly rather than
  rounded into a clean story either direction.
- **Latency/cost: a real win, but only once served correctly.** The naive
  comparison (unmerged LoRA adapter, base still in 4-bit) made the
  fine-tuned model look *slower* than base despite using 61% fewer
  tokens — an unmerged adapter pays extra matmul cost on every forward
  pass. Merging the adapter (`merge_and_unload()`, done in bf16 since
  PEFT's 4-bit merge is precision-lossy/fragile) removes that overhead
  entirely: merged p50 latency is **2.75x faster than the base model**,
  on top of the token savings. Bonus lesson: 4-bit quantization on a
  model this small was actively counterproductive — a 1.5B model fits
  comfortably in bf16 on a 16GB GPU, so the dequant overhead wasn't
  buying anything. Quantization pays off when VRAM is the actual
  constraint; here it wasn't.
- **Overfits fast.** Training's `eval_loss` hit its best value at epoch
  0.56 and got worse every eval step afterward while train loss kept
  falling toward zero — classic overfitting on a small (711-example)
  training set. `load_best_model_at_end=True` meant the saved adapter is
  correctly the early best checkpoint, not the overfit final-epoch one,
  but running the full 3 epochs was more than this dataset size needed.

### Failure modes worth naming

- **Base model has a strong "Software" bias for two categories.** It
  scored a flat 0% on both Network and Access/Password, with
  `Network->Software` (26x) and `Access/Password->Software` (21x) as the
  dominant confusions — it's not randomly wrong, it's systematically
  defaulting to one label.
- **Fine-tuned model's main weak spot: Network, confused with Hardware.**
  Network accuracy is only 42.3% — 15 of 26 Network test tickets were
  misclassified as Hardware (`Network->Hardware`, 15x), the only
  confusion the fine-tuned model has left. It's a defensible mistake, not
  a random one: network jack/cabling/router issues genuinely share
  vocabulary with hardware tickets. Every other category (Access/Password,
  Hardware, Software, Billing) hit 100%. This is the clearest concrete
  next step for this project: Network needs more diverse training
  phrasing to separate it from Hardware.
- **Priority judgment doesn't generalize from this dataset's diversity.**
  The clearest actionable next step for this project, not a footnote:
  either hand-write a larger, more diverse set of priority-labeled
  examples per template, or add reasoning traces (chain-of-thought before
  the final JSON) so the model has to articulate the impact reasoning
  during training instead of pattern-matching a template to a label.

Full report (per-category breakdown, confusion matrix, raw predictions
with per-repeat outputs for consistency debugging) is in
`results/comparison_report.md` and `results/*/predictions.jsonl`.

**LLM-as-judge scoring of the free-text severity justifications is not
yet run** — needs `eval/llm_judge.py` with an `ANTHROPIC_API_KEY`, which
wasn't set up on the training pod. See [Running it](#running-it) below.

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

## License

MIT (or your preference — update before publishing).
