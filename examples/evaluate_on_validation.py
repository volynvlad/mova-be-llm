#!/usr/bin/env python3
"""
Example: Evaluate the fine-tuned model on the validation dataset.

Loads random samples from the validation set and compares the model's
generated responses to the reference answers. Useful for qualitative
assessment of model quality after fine-tuning.
"""

import argparse
import json
import random
from pathlib import Path

from transformers import AutoTokenizer
from unsloth import FastLanguageModel


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        do_sample=False,
    )
    return tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)


def truncate(text: str, max_len: int = 300) -> str:
    return text[:max_len] + "..." if len(text) > max_len else text


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model on validation set")
    parser.add_argument("--model-path", default="./results/qwen2.5_1.5b_be", help="Path to the fine-tuned model")
    parser.add_argument("--val-path", default="data/processed/validation_dataset.jsonl", help="Path to validation JSONL")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples to evaluate")
    parser.add_argument("--max-length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum new tokens to generate")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    val_path = Path(args.val_path)
    if not val_path.exists():
        raise FileNotFoundError(f"Validation file not found: {val_path}")

    records = load_jsonl(val_path)
    print(f"Loaded {len(records)} validation records from {val_path}")

    rng = random.Random(args.seed)
    samples = rng.sample(records, min(args.num_samples, len(records)))

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    print(f"Loading model from: {model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_path),
        max_seq_length=args.max_length,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    print(f"Model loaded. Vocabulary size: {len(tokenizer)}\n")

    print("=" * 60)
    print(f"  Evaluating {len(samples)} validation samples")
    print("=" * 60)

    for i, record in enumerate(samples, 1):
        instruction = record["conversations"][0]["value"]
        reference = record["conversations"][1]["value"]

        print(f"\n--- Sample {i}/{len(samples)} ---")
        print(f"[Instruction]: {truncate(instruction)}")
        print(f"[Reference]:   {truncate(reference)}")
        print("-" * 40)

        generated = generate_response(model, tokenizer, instruction, args.max_new_tokens)
        print(f"[Generated]:   {truncate(generated)}")
        print()

    print("Done!")


if __name__ == "__main__":
    main()
