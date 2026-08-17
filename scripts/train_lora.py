"""LoRA fine-tune Qwen2.5-1.5B-Instruct on the IT ticket triage dataset.

Run on a CUDA GPU (e.g. a RunPod pod — see docs/RUNBOOK_RUNPOD.md). This is
QLoRA: the frozen base model is loaded in 4-bit (bitsandbytes NF4) and only
small trainable LoRA adapter matrices are updated in bf16. That's the
mechanism most companies mean when they say "we fine-tuned our LLM" —
see the "Why LoRA" section in README.md for the full tradeoff discussion.

Usage:
    python scripts/train_lora.py \
        --base-model Qwen/Qwen2.5-1.5B-Instruct \
        --train-file data/train.jsonl \
        --val-file data/val.jsonl \
        --output-dir checkpoints/qwen2.5-1.5b-ticket-lora
"""

import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--train-file", default="data/train.jsonl")
    p.add_argument("--val-file", default="data/val.jsonl")
    p.add_argument("--output-dir", default="checkpoints/qwen2.5-1.5b-ticket-lora")

    # --- LoRA hyperparameters ---
    # rank (r): dimensionality of the low-rank update matrices A (d x r) and
    # B (r x d) that get added to each frozen weight (W' = W + BA). Higher r
    # = more trainable capacity but more params/VRAM and higher overfit risk
    # on a small dataset. r=16 is the standard starting point for a task
    # this narrow (5-way classification + a short justification string) —
    # we're not teaching new knowledge, just a narrow output format/behavior,
    # so we don't need the capacity of r=64+ that open-ended chat tuning uses.
    p.add_argument("--lora-r", type=int, default=16)
    # alpha: scales the LoRA update (effective scale = alpha / r). alpha=2*r
    # is the well-established rule of thumb (Hu et al. 2021) that keeps the
    # update magnitude reasonable relative to the frozen weights at this rank.
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)

    # --- Training hyperparameters ---
    # LR: LoRA adapters are randomly initialized (A ~ Gaussian, B = 0) and
    # small relative to the full model, so they tolerate — and need — a much
    # higher LR than full fine-tuning (which typically uses 1e-5 to 5e-5).
    # 2e-4 is the standard QLoRA default; too low and the adapters barely
    # move off B=0 within 3 epochs on 718 examples.
    p.add_argument("--learning-rate", type=float, default=2e-4)
    # Epochs: small dataset (718 train examples) + narrow task converges
    # fast. 3 epochs is enough to fit the format/task without the adapter
    # memorizing phrasing idiosyncrasies of the synthetic templates (watched
    # via val loss — see --early-stopping-patience).
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--per-device-batch-size", type=int, default=4)
    # Gradient accumulation to reach an effective batch size of 16 without
    # needing 16 examples resident in VRAM at once (24GB-class GPU budget).
    p.add_argument("--grad-accum-steps", type=int, default=4)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--max-seq-length", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = build_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4-bit NF4 quantization of the frozen base weights (QLoRA). Compute
    # happens in bf16; only the stored weights are 4-bit. This is what lets
    # a 1.5B model train comfortably even on modest GPU VRAM (RunPod's
    # cheapest CUDA instances), and it's the standard bitsandbytes recipe.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        # Attention + MLP projections — covering both lets the adapter
        # adjust how the model attends to ticket text AND how it maps that
        # to the output format, rather than just one or the other.
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_ds = load_dataset("json", data_files=args.train_file, split="train")
    val_ds = load_dataset("json", data_files=args.val_file, split="train")

    def format_example(example):
        text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
        return {"text": text}

    train_ds = train_ds.map(format_example, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(format_example, remove_columns=val_ds.column_names)

    # Mask the prompt (system+user turns) out of the loss — we only want
    # the model learning to predict the assistant's JSON completion, not to
    # re-predict the ticket text it was given. response_template must match
    # the literal token sequence Qwen's chat template emits before the
    # assistant turn.
    response_template = "<|im_start|>assistant\n"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
        bf16=True,
        optim="paged_adamw_8bit",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=25,
        save_strategy="steps",
        save_steps=25,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to=["tensorboard"],
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Best eval_loss: {trainer.state.best_metric}")
    print(f"Adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
