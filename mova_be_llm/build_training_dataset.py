import argparse
import hashlib
import json
import random

from loguru import logger

from mova_be_llm.config import PROCESSED_DATA_DIR
from mova_be_llm.dataset_utils import is_belarusian_text, write_jsonl

DEFAULT_INPUT_FILES = [
    str(PROCESSED_DATA_DIR / "continuation_dataset.jsonl"),
    str(PROCESSED_DATA_DIR / "grammar_dataset.jsonl"),
    str(PROCESSED_DATA_DIR / "qa_dataset.jsonl"),
]


def load_jsonl_records(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError:
                logger.warning(f"Skipping invalid JSON at {path}:{line_num}")
    return records


def validate_record(record: dict) -> bool:
    if "conversations" not in record:
        return False

    convs = record["conversations"]
    if not isinstance(convs, list) or len(convs) != 2:
        return False

    if convs[0].get("from") != "human" or convs[1].get("from") != "gpt":
        return False

    human_text = convs[0].get("value", "")
    gpt_text = convs[1].get("value", "")

    if not human_text.strip() or not gpt_text.strip():
        return False

    if not is_belarusian_text(gpt_text):
        return False

    return True


def record_hash(record: dict) -> str:
    instruction = record["conversations"][0]["value"]
    response = record["conversations"][1]["value"]
    content = f"{instruction}|||{response}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Combine and split training datasets")
    parser.add_argument(
        "--input-files",
        nargs="+",
        type=str,
        default=DEFAULT_INPUT_FILES,
    )
    parser.add_argument("--output-path", type=str, default=str(PROCESSED_DATA_DIR / "training_dataset.jsonl"))
    parser.add_argument("--val-split", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    all_records = []
    source_counts = {}

    for input_file in args.input_files:
        source_name = input_file.rsplit("/", 1)[-1].replace(".jsonl", "")
        logger.info(f"Loading {input_file}...")

        try:
            records = load_jsonl_records(input_file)
        except FileNotFoundError:
            logger.error(f"File not found: {input_file}")
            continue

        valid_records = [r for r in records if validate_record(r)]
        invalid_count = len(records) - len(valid_records)
        if invalid_count > 0:
            logger.warning(f"Filtered {invalid_count} invalid records from {source_name}")

        source_counts[source_name] = len(valid_records)
        all_records.extend(valid_records)
        logger.info(f"Loaded {len(valid_records)} valid records from {source_name}")

    logger.info(f"Total records before dedup: {len(all_records)}")
    seen_hashes = set()
    unique_records = []
    for record in all_records:
        h = record_hash(record)
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_records.append(record)

    dedup_removed = len(all_records) - len(unique_records)
    if dedup_removed > 0:
        logger.info(f"Removed {dedup_removed} duplicates")

    rng.shuffle(unique_records)

    val_size = int(len(unique_records) * args.val_split)
    train_records = unique_records[val_size:]
    val_records = unique_records[:val_size]

    train_path = args.output_path
    val_path = args.output_path.replace("training_dataset", "validation_dataset")

    write_jsonl(train_records, train_path)
    write_jsonl(val_records, val_path)

    train_lengths = []
    val_lengths = []
    for r in train_records:
        total_len = len(r["conversations"][0]["value"]) + len(r["conversations"][1]["value"])
        train_lengths.append(total_len)
    for r in val_records:
        total_len = len(r["conversations"][0]["value"]) + len(r["conversations"][1]["value"])
        val_lengths.append(total_len)

    logger.info("=== Dataset Statistics ===")
    logger.info(f"Total unique records: {len(unique_records)}")
    logger.info(f"Training set: {len(train_records)}")
    logger.info(f"Validation set: {len(val_records)}")
    logger.info("Per-source counts:")
    for source, count in sorted(source_counts.items()):
        logger.info(f"  {source}: {count}")
    if train_lengths:
        avg_train = sum(train_lengths) / len(train_lengths)
        logger.info(f"Avg train conversation length: {avg_train:.0f} chars")
    if val_lengths:
        avg_val = sum(val_lengths) / len(val_lengths)
        logger.info(f"Avg validation conversation length: {avg_val:.0f} chars")


if __name__ == "__main__":
    main()
