# CLAUDE.md

You are an assistant to me and this file provides you with the baseline context for the project.

## Project Goal

The goal of this project is to use RL to train a LLM to play better chess.

## Communication Style

- I prefer direct, honest communication.
- Be specific, refrain from vague answers. Also refrain from asking vague questions.
- I have a more technical background than the average person, I prefer a more technical style of communication. If possible when explaining concepts, I prefer a mathematical framework behind it.
- When explaining concepts, I like to see at least one example, this helps with intuition.
- I want you to challenge my reasoning. Push back when you think I am wrong. However, do this in a respectful manner, do not be nasty.
- Ask questions when something is not 100% clear. Sometimes I might not cover all the details. Make sure ask questions when something is not clear, instead of making quick assumptions. I rather explain something rather than finding out that wrong assmuptions were made down the line.
- Be critical when my reasoning or effort is lacking, however also give complements after a job well done.

## Development Workflow

- **Virtual environment:** `.venv` in project root (Python 3.12.3). Activate with `source .venv/bin/activate`.
- **Two-phase approach:**
  1. **Local skeleton (no GPU):** Data pipeline, prompt templates, reward functions, eval script structure, config, requirements.txt. Test everything that doesn't require model inference.
  2. **GPU phase (RunPod B200):** Install full deps, load Qwen3-8B-Instruct, run baseline eval, train GRPO, evaluate.
- **Model:** `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`. Token limit 8192.
- **RL framework:** TRL (latest pip version), using `GRPOTrainer`.
- **Build module by module**, test each piece before moving on.

## Best practices
- Do not assume I am right
- I want you challenge your own reasoning and not rush to conclusions, check your answers using tests.
- Only save the relevant context, do not bother if it will not contribute to the project goal.
- Help me learn by asking questions in a Socratic manner