#!/usr/bin/env python3
"""
Example: Inspect the generated Belarusian training datasets.

Shows sample entries from each dataset type (continuation, grammar, QA)
and prints statistics about the combined training/validation split.
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def show_samples(records: list[dict], title: str, n: int = 3, seed: int = 42):
    rng = random.Random(seed)
    samples = rng.sample(records, min(n, len(records)))

    print(f"\n{'=' * 60}")
    print(f"  {title} ({len(records)} records)")
    print(f"{'=' * 60}")

    for i, record in enumerate(samples, 1):
        human = record["conversations"][0]["value"]
        gpt = record["conversations"][1]["value"]

        # Truncate for display
        human_display = human[:200] + "..." if len(human) > 200 else human
        gpt_display = gpt[:200] + "..." if len(gpt) > 200 else gpt

        print(f"\n--- Sample {i} ---")
        print(f"[Human]: {human_display}")
        print(f"[GPT]:   {gpt_display}")


def print_stats(records: list[dict], title: str):
    human_lens = [len(r["conversations"][0]["value"]) for r in records]
    gpt_lens = [len(r["conversations"][1]["value"]) for r in records]
    total_lens = [h + g for h, g in zip(human_lens, gpt_lens)]

    print(f"\n{title}:")
    print(f"  Records:         {len(records)}")
    print(f"  Avg human len:   {sum(human_lens) / len(human_lens):.0f} chars")
    print(f"  Avg gpt len:     {sum(gpt_lens) / len(gpt_lens):.0f} chars")
    print(f"  Avg total len:   {sum(total_lens) / len(total_lens):.0f} chars")
    print(f"  Min total len:   {min(total_lens)} chars")
    print(f"  Max total len:   {max(total_lens)} chars")


def detect_task_type(record: dict) -> str:
    human = record["conversations"][0]["value"]
    if any(t in human for t in ["Працягні", "Прадоўжы", "Дапішы", "Напішы далей", "Напішы працяг", "Закончы"]):
        return "continuation"
    if any(t in human for t in ["Выправі", "Знайдзі і выправі", "Перапішы", "Напішы правільна"]):
        return "grammar_correction"
    if any(t in human for t in ["Устаў прапушчаныя", "Запоўні пропускі", "Дапішы прапушчаныя"]):
        return "cloze"
    if any(t in human for t in ["правільна напісаць", "правільная", "правільны варыянт"]):
        return "spelling"
    if any(t in human for t in ["Пра што", "Якая тэма", "Аб чым", "Назаві тэму", "галоўная думка"]):
        return "qa_topic"
    if any(t in human for t in ["Ці праўда"]):
        return "qa_true_false"
    if human.endswith("?"):
        return "qa_factual"
    return "other"


def main():
    parser = argparse.ArgumentParser(description="Inspect Belarusian training datasets")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--samples", type=int, default=3, help="Number of samples to show per dataset")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Show samples from individual datasets
    dataset_files = [
        ("continuation_dataset.jsonl", "Text Continuation"),
        ("grammar_dataset.jsonl", "Grammar & Vocabulary"),
        ("qa_dataset.jsonl", "QA & Comprehension"),
    ]

    for filename, title in dataset_files:
        path = data_dir / filename
        if path.exists():
            records = load_jsonl(path)
            show_samples(records, title, args.samples, args.seed)
            print_stats(records, f"  Stats for {filename}")
        else:
            print(f"\n[!] Not found: {path}")

    # Combined dataset stats
    print(f"\n{'=' * 60}")
    print("  Combined Training Dataset")
    print(f"{'=' * 60}")

    train_path = data_dir / "training_dataset.jsonl"
    val_path = data_dir / "validation_dataset.jsonl"

    if train_path.exists():
        train = load_jsonl(train_path)
        print_stats(train, "  Training set")

        task_counts = Counter(detect_task_type(r) for r in train)
        print("\n  Task type distribution:")
        for task, count in task_counts.most_common():
            pct = count / len(train) * 100
            print(f"    {task:25s} {count:6d}  ({pct:.1f}%)")

    if val_path.exists():
        val = load_jsonl(val_path)
        print_stats(val, "  Validation set")

    if train_path.exists() and val_path.exists():
        print(f"\n  Total: {len(train) + len(val)} records "
              f"(train: {len(train)}, val: {len(val)}, "
              f"split: {len(val) / (len(train) + len(val)) * 100:.1f}%)")


if __name__ == "__main__":
    main()
