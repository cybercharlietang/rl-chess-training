"""Model loading and inference utilities for diagnostic tests."""

import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = "You are a chess expert. Answer the question precisely and concisely."


def load_model(model_name: str):
    """Load model and tokenizer in bf16."""
    print(f"Loading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Model loaded on {model.device}")
    return model, tokenizer


def generate_answer(model, tokenizer, question: str, max_new_tokens: int = 512) -> str:
    """Generate a single answer using greedy decoding."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, temperature=None, top_p=None,
        )
    prompt_len = inputs["input_ids"].shape[1]
    generated = output[0][prompt_len:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def generate_answers_batch(
    model, tokenizer, questions: list[str], batch_size: int = 8, max_new_tokens: int = 512,
) -> list[str]:
    """Generate answers for a list of questions using batched greedy decoding."""
    all_answers = []
    tokenizer.padding_side = "left"

    for i in range(0, len(questions), batch_size):
        batch_qs = questions[i : i + batch_size]
        prompts = []
        for q in batch_qs:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q},
            ]
            prompts.append(tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            ))

        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=False, temperature=None, top_p=None,
            )

        for j, output in enumerate(outputs):
            prompt_len = inputs["input_ids"][j].shape[0]
            generated = output[prompt_len:]
            answer = tokenizer.decode(generated, skip_special_tokens=True)
            all_answers.append(answer)

    return all_answers


def extract_short_answer(raw: str) -> str:
    """Extract the concise answer from model output, handling <think> tags."""
    # If there's an <answer> tag, use that
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", raw, re.DOTALL)
    if m:
        return m.group(1).strip()

    # If there's </think>, take everything after it
    if "</think>" in raw:
        after = raw.split("</think>")[-1].strip()
        if after:
            # Take the first non-empty line
            for line in after.split("\n"):
                line = line.strip()
                if line:
                    return line

    # Otherwise just take the last non-empty line (models often reason then answer)
    lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]
    if lines:
        return lines[-1]
    return raw.strip()
