import argparse
import random

from loguru import logger

from mova_be_llm.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from mova_be_llm.dataset_utils import (
    load_all_paragraphs,
    load_paragraphs_by_file,
    make_conversation,
    split_into_sentences,
    write_jsonl,
)

CONTINUATION_TEMPLATES = [
    "Працягні тэкст:",
    "Напішы працяг гэтага тэксту:",
    "Што будзе далей?",
    "Прадоўжы наступны ўрывак:",
    "Дапішы тэкст:",
    "Працягні апавяданне:",
    "Напішы, што было далей:",
    "Як працягнецца гэты тэкст?",
    "Прадоўжы думку:",
    "Дадай працяг да тэксту:",
    "Закончы тэкст:",
    "Напішы далей:",
]


def find_split_point(text: str, ratio: float) -> int | None:
    sentences = split_into_sentences(text)
    if len(sentences) < 2:
        return None

    target_pos = int(len(text) * ratio)
    best_pos = None
    best_distance = len(text)

    current_pos = 0
    for i, sentence in enumerate(sentences[:-1]):
        current_pos += len(sentence) + 1
        distance = abs(current_pos - target_pos)
        if distance < best_distance:
            best_distance = distance
            best_pos = current_pos
            best_sent_idx = i

    return best_pos


def create_single_paragraph_samples(
    paragraphs: list[tuple[str, str]],
    split_ratio: float,
    rng: random.Random,
) -> list[dict]:
    records = []
    for _filename, para in paragraphs:
        if len(para) < 200 or len(para) > 2000:
            continue

        split_pos = find_split_point(para, split_ratio)
        if split_pos is None:
            continue

        prefix = para[:split_pos].strip()
        continuation = para[split_pos:].strip()

        if len(prefix) < 50 or len(continuation) < 50:
            continue
        if len(prefix.split()) < 8 or len(continuation.split()) < 8:
            continue

        template = rng.choice(CONTINUATION_TEMPLATES)
        instruction = f"{template}\n\n{prefix}"
        records.append(make_conversation(instruction, continuation))

    return records


def create_multi_paragraph_samples(
    data_dir: str,
    rng: random.Random,
    max_samples: int,
) -> list[dict]:
    records = []

    for _filename, file_paragraphs in load_paragraphs_by_file(data_dir):
        if len(file_paragraphs) < 2:
            continue

        for i in range(len(file_paragraphs) - 1):
            num_paras = rng.randint(2, min(3, len(file_paragraphs) - i))
            combined = " ".join(file_paragraphs[i : i + num_paras])

            if len(combined) < 200 or len(combined) > 3000:
                continue

            split_pos = find_split_point(combined, 0.4)
            if split_pos is None:
                continue

            prefix = combined[:split_pos].strip()
            continuation = combined[split_pos:].strip()

            if len(prefix) < 50 or len(continuation) < 50:
                continue
            if len(prefix.split()) < 8 or len(continuation.split()) < 8:
                continue

            template = rng.choice(CONTINUATION_TEMPLATES)
            instruction = f"{template}\n\n{prefix}"
            records.append(make_conversation(instruction, continuation))

            if len(records) >= max_samples:
                return records

    return records


def main():
    parser = argparse.ArgumentParser(description="Create text continuation dataset")
    parser.add_argument("--input-dir", type=str, default=str(INTERIM_DATA_DIR))
    parser.add_argument("--output-path", type=str, default=str(PROCESSED_DATA_DIR / "continuation_dataset.jsonl"))
    parser.add_argument("--max-samples", type=int, default=50000)
    parser.add_argument("--split-ratio", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    logger.info("Loading paragraphs...")
    paragraphs = list(load_all_paragraphs(args.input_dir))
    logger.info(f"Loaded {len(paragraphs)} paragraphs")

    logger.info("Creating single-paragraph continuation samples...")
    single_records = create_single_paragraph_samples(paragraphs, args.split_ratio, rng)
    logger.info(f"Created {len(single_records)} single-paragraph samples")

    multi_budget = max(0, args.max_samples - len(single_records))
    logger.info(f"Creating multi-paragraph continuation samples (budget: {multi_budget})...")
    multi_records = create_multi_paragraph_samples(args.input_dir, rng, multi_budget)
    logger.info(f"Created {len(multi_records)} multi-paragraph samples")

    all_records = single_records + multi_records
    rng.shuffle(all_records)

    if len(all_records) > args.max_samples:
        all_records = all_records[: args.max_samples]

    logger.info(f"Total continuation samples: {len(all_records)}")
    write_jsonl(all_records, args.output_path)


if __name__ == "__main__":
    main()
