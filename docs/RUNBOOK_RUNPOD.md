# Runbook: Training on RunPod

This is the exact sequence to go from zero to a trained LoRA adapter on a
rented RunPod GPU. Total cost for this dataset size (718 train examples, 3
epochs, Qwen2.5-1.5B) should land around **$0.50-$1.50** and take
**15-30 minutes** of actual training time on a single RTX 4090.

## 1. Create the pod

1. Go to [runpod.io](https://runpod.io) → Console → Pods → Deploy.
2. GPU: **RTX 4090 (24GB)** — cheapest GPU that comfortably fits a 1.5B
   model in 4-bit plus activations. (An RTX 3090 or A4000 also work and are
   often cheaper; anything with ≥16GB VRAM is fine for this model size.)
3. Template: **RunPod PyTorch 2.x** (comes with CUDA + PyTorch preinstalled).
4. Storage: 20GB container disk is plenty (base model is ~3GB in 4-bit).
5. Deploy, then open a **Web Terminal** or copy the **SSH command** from the
   pod's "Connect" menu.

## 2. Environment setup (on the pod)

```bash
cd /workspace
git clone <your-repo-url> llm-finetuning   # or scp/rsync the project up
cd llm-finetuning
pip install -r requirements.txt
```

If you didn't push to GitHub yet, the fastest path is `scp` from your
laptop:

```bash
# run this LOCALLY, not on the pod
rsync -avz --exclude 'checkpoints' --exclude '.git' \
    /Volumes/LaCie/llm-finetuning/ root@<pod-ip>:/workspace/llm-finetuning/
```

## 3. (If using a gated model) accept the license + login

Qwen2.5 is not gated, so this step is normally skippable. If you switch to
a gated model (e.g. Llama 3.2), you'd need:

```bash
huggingface-cli login   # paste a token with read access
```

## 4. Kick off training

```bash
python scripts/train_lora.py \
    --base-model Qwen/Qwen2.5-1.5B-Instruct \
    --train-file data/train.jsonl \
    --val-file data/val.jsonl \
    --output-dir checkpoints/qwen2.5-1.5b-ticket-lora
```

## 5. Monitor

In a second terminal on the pod (or via RunPod's exposed port):

```bash
tensorboard --logdir checkpoints/qwen2.5-1.5b-ticket-lora --port 6006 --host 0.0.0.0
```

Expose port 6006 in the RunPod dashboard (Pods → your pod → "Connect" →
HTTP port) and watch `train/loss` and `eval/loss`. What to expect:
- `train/loss` should drop sharply in the first ~20 steps then plateau.
- `eval/loss` should track train loss closely (small gap = not overfitting).
  If eval loss starts rising while train loss keeps dropping, the run is
  overfitting the synthetic templates — stop early and lower `--epochs`.

## 6. Pull the adapter back down

The adapter is small (LoRA weights only, ~10-30MB, not the full 1.5B
model), so this is fast:

```bash
# run this LOCALLY
rsync -avz root@<pod-ip>:/workspace/llm-finetuning/checkpoints/ \
    /Volumes/LaCie/llm-finetuning/checkpoints/
```

## 7. Terminate the pod

**Don't forget this step** — RunPod bills by the minute while the pod is
running, even if idle. Console → Pods → Stop/Terminate.

## 8. Run evaluation

Evaluation (`eval/evaluate.py`) needs a GPU too (it runs both the base
model and the fine-tuned model), so either:
- run it on the same pod before terminating (add ~10 min / ~$0.10), or
- spin up a fresh pod later with the adapter already pulled down.

```bash
python eval/evaluate.py \
    --base-model Qwen/Qwen2.5-1.5B-Instruct \
    --adapter-path checkpoints/qwen2.5-1.5b-ticket-lora \
    --test-file data/test.jsonl \
    --out results/
```

See [README.md](../README.md) for what the eval harness reports and how to
read the results.
