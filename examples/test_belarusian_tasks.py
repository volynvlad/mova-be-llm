#!/usr/bin/env python3
"""
Example: Test the fine-tuned Belarusian model on specific task types.

Runs the model through the same task categories used during training:
text continuation, grammar correction, and QA/comprehension.
"""

import argparse
from pathlib import Path

from transformers import AutoTokenizer
from unsloth import FastLanguageModel


def generate_response(model, tokenizer, prompt: str, temperature: float = 0.0, max_new_tokens: int = 512) -> str:
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    gen_kwargs = {
        "input_ids": inputs,
        "max_new_tokens": max_new_tokens,
        "use_cache": True,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["do_sample"] = True
    else:
        gen_kwargs["do_sample"] = False

    outputs = model.generate(**gen_kwargs)
    return tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Task-specific prompts matching the training data format
# ---------------------------------------------------------------------------

CONTINUATION_PROMPTS = [
    "Працягні тэкст:\nВясной прырода абуджаецца ад зімовага сну. Першыя кветкі з'яўляюцца на палянах,",
    "Напішы працяг гэтага тэксту:\nБеларусь — краіна з багатай гісторыяй і культурай.",
    "Прадоўжы наступны ўрывак:\nУ старадаўнія часы на беларускіх землях жылі плямёны крывічоў, дрыгавічоў і радзімічаў.",
]

GRAMMAR_PROMPTS = [
    "Выправі памылкі ў тэксце:\nУ беларускай мове ёсць шмат цікавых слоу і выразау, якія не маюць аналагау у іншых мовах.",
    "Знайдзі і выправі русізмы ў тэксце:\nДеці пайшлі у школу вучыцца чытаць книгі.",
    "Перапішы тэкст без памылак:\nБеларуская мова — гэта адна з самых прыгожых славянскіх моу.",
    "Устаў прапушчаныя словы:\nБеларусь ____ ў цэнтры Еўропы. Яе ____ — Мінск.",
]

QA_PROMPTS = [
    "Пра што гэты тэкст?\nМінск — сталіца Рэспублікі Беларусь, найбуйнейшы горад краіны. Горад размешчаны на рацэ Свіслач. Першае пісьмовае ўпамінанне пра Мінск адносіцца да 1067 года.",
    "Ці праўда наступнае сцвярджэнне?\nМінск з'яўляецца сталіцай Украіны.",
    "Якая тэма гэтага ўрыўку?\nФранцыск Скарына — беларускі першадрукар, які ў 1517 годзе выдаў першую друкаваную кнігу на старабеларускай мове.",
]


def run_task_tests(model, tokenizer, temperature: float, max_new_tokens: int):
    task_groups = [
        ("Text Continuation", CONTINUATION_PROMPTS),
        ("Grammar Correction", GRAMMAR_PROMPTS),
        ("QA & Comprehension", QA_PROMPTS),
    ]

    for group_name, prompts in task_groups:
        print(f"\n{'=' * 60}")
        print(f"  {group_name}")
        print(f"{'=' * 60}")

        for i, prompt in enumerate(prompts, 1):
            print(f"\n--- Task {i}/{len(prompts)} ---")
            print(f"[Prompt]:\n{prompt}")
            print("-" * 40)

            response = generate_response(model, tokenizer, prompt, temperature, max_new_tokens)
            print(f"[Response]:\n{response}\n")


def main():
    parser = argparse.ArgumentParser(description="Test fine-tuned Belarusian model on specific tasks")
    parser.add_argument("--model-path", default="./results/qwen2.5_1.5b_be", help="Path to the fine-tuned model")
    parser.add_argument("--max-length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 = greedy)")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum new tokens to generate")
    args = parser.parse_args()

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
    print(f"Model loaded. Vocabulary size: {len(tokenizer)}")

    run_task_tests(model, tokenizer, args.temperature, args.max_new_tokens)
    print("\nDone!")


if __name__ == "__main__":
    main()
