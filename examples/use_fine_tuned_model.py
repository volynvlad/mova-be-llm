#!/usr/bin/env python3
"""
Example: Load and use a fine-tuned Belarusian Qwen model.

This script demonstrates how to load a previously fine-tuned model and tokenizer
from the fine_tune_belarusian_qwen.py training script, and generate text using it.
"""

import argparse
from pathlib import Path
from loguru import logger
from transformers import AutoTokenizer
from unsloth import FastLanguageModel


def generate_response(model, tokenizer, prompt: str, temperature: float = 0.7, max_new_tokens: int = 512) -> str:
    """Generate a response from the model for a given prompt."""
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
        temperature=temperature,
        do_sample=True,
    )

    return tokenizer.decode(outputs[0][len(inputs[0]) :], skip_special_tokens=True)


def run_test_prompts(model, tokenizer, temperature: float = 0.7, max_new_tokens: int = 512):
    """Run a set of test prompts through the model."""
    test_prompts = [
        "Беларуская мова - гэта",
        "Мінск з'яўляецца",
        "Вясной у лесе",
    ]

    print("\n" + "=" * 60)
    print("Testing fine-tuned Belarusian Qwen model")
    print("=" * 60 + "\n")

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n--- Test {i}/{len(test_prompts)} ---")
        print(f"Prompt: {prompt}")
        print("-" * 40)

        response = generate_response(model, tokenizer, prompt, temperature, max_new_tokens)
        print(f"Generated: {response}")
        print()


def load_fine_tuned_model(model_path: Path, max_length: int = 1024):
    """Load a fine-tuned model and tokenizer from disk."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    print(f"Loading model from: {model_path}")

    # Load the model (this loads the LoRA adapter on top of the base model)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_path),
        max_seq_length=max_length,
        dtype=None,  # Auto-detect
        load_in_4bit=True,
    )

    # Enable 2x faster inference
    FastLanguageModel.for_inference(model)

    print(f"Model loaded successfully!")
    print(f"Tokenizer vocabulary size: {len(tokenizer)}")

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Load and test a fine-tuned Belarusian Qwen model")
    parser.add_argument(
        "--model-path",
        default="./results/qwen2.5_1.5b_be",
        help="Path to the fine-tuned model directory (LoRA adapter)",
    )
    parser.add_argument("--max-length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature for generation")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum number of new tokens to generate")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode (type your own prompts)")

    args = parser.parse_args()
    model, tokenizer = load_fine_tuned_model(Path(args.model_path), args.max_length)

    if args.interactive:
        print("\n" + "=" * 60)
        print("Interactive mode (type 'quit' to exit)")
        print("=" * 60 + "\n")

        while True:
            try:
                prompt = input("\nPrompt: ").strip()
                if prompt.lower() in ("quit", "exit", "q"):
                    break
                if not prompt:
                    continue

                print("-" * 40)
                response = generate_response(model, tokenizer, prompt, args.temperature, args.max_new_tokens)
                print(f"Generated: {response}")

            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
    else:
        run_test_prompts(model, tokenizer, args.temperature, args.max_new_tokens)

    print("\nDone!")


if __name__ == "__main__":
    main()
