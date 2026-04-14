"""Model loading and inference utilities for diagnostic tests (vLLM backend)."""

import os
import re

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

SYSTEM_PROMPT = "You are a chess expert. Answer the question precisely and concisely."

_ENABLE_THINKING = os.environ.get("DIAG_THINKING", "1") == "1"
_MAX_TOKENS = int(os.environ.get("DIAG_MAX_TOKENS", "8192"))
_GPU_MEM = float(os.environ.get("DIAG_GPU_MEM", "0.80"))


def load_model(model_name: str):
    """Load vLLM engine + tokenizer."""
    print(f"Loading {model_name} via vLLM (thinking={_ENABLE_THINKING}, max_tokens={_MAX_TOKENS})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = LLM(
        model=model_name, tensor_parallel_size=1, dtype="bfloat16",
        gpu_memory_utilization=_GPU_MEM, max_model_len=_MAX_TOKENS + 2048,
    )
    print("Model loaded")
    return model, tokenizer


def generate_answer(model, tokenizer, question: str, max_new_tokens: int = 512) -> str:
    return generate_answers_batch(model, tokenizer, [question], batch_size=1, max_new_tokens=max_new_tokens)[0]


def generate_answers_batch(
    model, tokenizer, questions, batch_size: int = 8, max_new_tokens: int = 512,
):
    """Generate answers for a list of questions using vLLM batched decoding."""
    prompts = []
    for q in questions:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
        ]
        prompts.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=_ENABLE_THINKING,
        ))
    # Use env-configured max if thinking (need lots of headroom), else the small default
    actual_max = _MAX_TOKENS if _ENABLE_THINKING else max_new_tokens
    params = SamplingParams(temperature=0.0, max_tokens=actual_max)
    outs = model.generate(prompts, params)
    return [o.outputs[0].text for o in outs]


def extract_short_answer(raw: str) -> str:
    """Extract the concise answer from model output, handling <think> tags."""
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", raw, re.DOTALL)
    if m:
        return m.group(1).strip()

    if "</think>" in raw:
        after = raw.split("</think>")[-1].strip()
        if after:
            for line in after.split("\n"):
                line = line.strip()
                if line:
                    return line

    lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]
    if lines:
        return lines[-1]
    return raw.strip()
