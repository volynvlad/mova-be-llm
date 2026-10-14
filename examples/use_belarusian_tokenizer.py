"""
Example: Using Belarusian-enhanced Qwen Tokenizer

This example demonstrates how to use the merged Belarusian tokenizer with Qwen models
for better Belarusian text processing and generation.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_belarusian_qwen_model(
    tokenizer_path: str = "./results/qwen2.5_7b_be_fixed_bytes",
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
):
    """
    Load Qwen model with Belarusian-enhanced tokenizer.

    Args:
        tokenizer_path: Path to the merged tokenizer
        base_model: HuggingFace model name for the base model

    Returns:
        tuple: (model, tokenizer)
    """
    print(f"Loading base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )

    print(f"Loading Belarusian tokenizer: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Resizing model embeddings...")
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    print(f"Model vocabulary resized to {len(tokenizer):,} tokens")

    return model, tokenizer


def compare_tokenization(tokenizer_original, tokenizer_be, texts: list[str]):
    """Compare original vs Belarusian-enhanced tokenizer performance."""
    print("\n=== Tokenization Comparison ===")

    for text in texts:
        print(f"\nText: '{text}'")

        orig_ids = tokenizer_original.encode(text, add_special_tokens=False)
        orig_tokens = [tokenizer_original.decode([tid]) for tid in orig_ids]

        be_ids = tokenizer_be.encode(text, add_special_tokens=False)
        be_tokens = [tokenizer_be.decode([tid]) for tid in be_ids]

        print(f"Original:  {orig_tokens} ({len(orig_ids)} tokens)")
        print(f"Belarusian: {be_tokens} ({len(be_ids)} tokens)")
        print(
            f"Improvement: {len(orig_ids) - len(be_ids)} tokens reduction ({(1 - len(be_ids) / len(orig_ids)) * 100:.1f}%)"
        )


def generate_text(model, tokenizer, prompt: str, max_length: int = 100):
    """Generate text using the model with Belarusian tokenizer."""
    print(f"\n=== Generation ===")
    print(f"Prompt: '{prompt}'")

    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer.encode(formatted_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs, max_new_tokens=max_length, temperature=0.7, do_sample=True, pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0][len(inputs[0]) :], skip_special_tokens=True)
    print(f"Generated: '{response}'")

    return response


def main():
    """Main example demonstrating Belarusian tokenizer usage."""
    # tokenizer_be = AutoTokenizer.from_pretrained("./results/qwen2.5_7b_be_fixed_bytes", trust_remote_code=True)
    model, tokenizer_be = load_belarusian_qwen_model()

    print(f"Belarusian vocab size: {len(tokenizer_be.get_vocab()):,}")

    generation_prompts = [
        "Напішы кароткае апавяданне пра беларускую прыроду.",
        "Перакладзі на беларускую мову: 'The weather is nice today.'",
        "Што такае нацыянальная ідэнтычнасць?",
    ]

    print("\n=== Generation Results ===")
    for i, prompt in enumerate(generation_prompts, 1):
        print(f"\n--- Prompt {i} ---")
        try:
            generate_text(model, tokenizer_be, prompt, max_length=80)
        except Exception as e:
            print(f"Error generating: {e}")


if __name__ == "__main__":
    main()
