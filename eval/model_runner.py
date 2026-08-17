"""Model loading + generation for the base-vs-fine-tuned comparison.

Requires a GPU and the training stack (transformers/peft/bitsandbytes) —
run this on the RunPod pod (or any CUDA box) after training, not on a
laptop. Kept separate from eval/metrics.py so the scoring logic can be
unit tested without touching a model.
"""

import json
import re
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SYSTEM_PROMPT = (
    "You are an IT support ticket triage assistant. Given a ticket subject "
    "and description, classify it and respond with ONLY a JSON object with "
    "exactly these keys: \"category\" (one of: Network, Access/Password, "
    "Hardware, Software, Billing), \"priority\" (one of: P1-Critical, "
    "P2-High, P3-Medium, P4-Low), and \"severity_justification\" (one "
    "sentence explaining the priority based on business impact). "
    "No extra text, no markdown, just the JSON object."
)

# Few-shot examples for the PROMPTED BASE MODEL baseline. Pulled from
# data/train.jsonl (never from val/test — that would leak eval data into
# the baseline's context). This is deliberately a strong, realistic
# baseline: 3-shot with the exact schema, not a strawman zero-shot prompt.
# A fine-tuned model needs none of this at inference time, which is exactly
# the "fewer tokens needed" advantage measured in latency/cost stats.
FEW_SHOT_EXAMPLES = [
    {
        "subject": "VPN connection keeps dropping every few minutes",
        "description": "Hi IT, I'm James Ortega from Sales. VPN connection keeps dropping every few minutes. This started roughly this morning. I've already tried reconnecting the VPN client but the issue persists.",
        "output": {"category": "Network", "priority": "P2-High",
                    "severity_justification": "Priya Nair in Finance cannot maintain a stable VPN session, interrupting remote work. Time-sensitive but limited to one person or one system."},
    },
    {
        "subject": "Password reset email never arrived",
        "description": "Hello, this is Wei Chen (HR). Password reset email never arrived, related to Workday. I need this resolved so I can process payroll.",
        "output": {"category": "Access/Password", "priority": "P3-Medium",
                    "severity_justification": "User can still access most tools; only this one reset is pending. Should be fixed this week; not stopping any deadline-critical work."},
    },
    {
        "subject": "Dell 27\" monitor won't power on at all",
        "description": "Hi, Daniel Kim here from Engineering. Dell 27\" monitor won't power on at all. This is affecting a client deadline.",
        "output": {"category": "Hardware", "priority": "P2-High",
                    "severity_justification": "Daniel Kim has no working machine and cannot do any work until replaced or repaired. Single user is fully blocked from a core work task, but a workaround may exist."},
    },
]


def load_base_model(model_name, dtype=torch.bfloat16):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, device_map="auto",
    )
    model.eval()
    return model, tokenizer


def load_finetuned_model(base_model_name, adapter_path, dtype=torch.bfloat16):
    model, tokenizer = load_base_model(base_model_name, dtype=dtype)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def build_prompt_messages(subject, description, use_few_shot):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if use_few_shot:
        for ex in FEW_SHOT_EXAMPLES:
            messages.append({
                "role": "user",
                "content": f"Subject: {ex['subject']}\n\nDescription: {ex['description']}",
            })
            messages.append({
                "role": "assistant",
                "content": json.dumps(ex["output"], ensure_ascii=False),
            })
    messages.append({
        "role": "user",
        "content": f"Subject: {subject}\n\nDescription: {description}",
    })
    return messages


def parse_prediction(raw_text):
    """Extract the schema JSON from a raw generation. Tolerant of markdown
    fences and leading/trailing chatter, since the prompted base model
    (unlike the fine-tuned one) is prone to adding both.

    Returns (parsed_dict_or_none, success_bool).
    """
    text = raw_text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None, False
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None, False
    required = {"category", "priority", "severity_justification"}
    if not required.issubset(parsed.keys()):
        return None, False
    return parsed, True


@torch.inference_mode()
def generate(model, tokenizer, messages, do_sample=False, temperature=0.7, max_new_tokens=120):
    """Single generation, returns (raw_text, latency_ms, prompt_tokens, completion_tokens)."""
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_tokens = inputs["input_ids"].shape[1]

    start = time.perf_counter()
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        pad_token_id=tokenizer.pad_token_id,
    )
    latency_ms = (time.perf_counter() - start) * 1000

    completion_ids = output_ids[0][prompt_tokens:]
    raw_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
    return raw_text, latency_ms, prompt_tokens, len(completion_ids)
