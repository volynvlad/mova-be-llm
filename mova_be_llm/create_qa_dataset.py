import argparse
import random
import re

from loguru import logger

from mova_be_llm.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from mova_be_llm.dataset_utils import (
    load_all_paragraphs,
    make_conversation,
    split_into_sentences,
    write_jsonl,
)

TOPIC_TEMPLATES = [
    "Пра што гэты тэкст?",
    "Якая тэма гэтага ўрыўку?",
    "Пра што апавядаецца ў тэксце?",
    "Аб чым ідзе гаворка ў гэтым урыўку?",
    "Назаві тэму гэтага тэксту.",
    "Якая галоўная думка тэксту?",
]

FACTUAL_PATTERNS = [
    (
        re.compile(r"(.{3,50})\s+з'яўляецца\s+(.{3,80})", re.UNICODE),
        lambda m: f"Чым з'яўляецца {m.group(1).strip()}?",
        lambda m: f"{m.group(1).strip()} з'яўляецца {m.group(2).strip()}",
    ),
    (
        re.compile(r"(.{3,50})\s+знаходзіцца\s+(?:ў|на)\s+(.{3,80})", re.UNICODE),
        lambda m: f"Дзе знаходзіцца {m.group(1).strip()}?",
        lambda m: f"{m.group(1).strip()} знаходзіцца {m.group(2).strip()}",
    ),
    (
        re.compile(r"[Уу]\s+(\d{3,4})\s+годзе\s+(.{5,100})", re.UNICODE),
        lambda m: f"Што адбылося ў {m.group(1)} годзе?",
        lambda m: f"У {m.group(1)} годзе {m.group(2).strip()}",
    ),
    (
        re.compile(r"(.{3,50})\s+называецца\s+(.{3,80})", re.UNICODE),
        lambda m: f"Як называецца {m.group(1).strip()}?",
        lambda m: f"{m.group(1).strip()} называецца {m.group(2).strip()}",
    ),
    (
        re.compile(r"(.{3,40})\s+(?:напісаў|стварыў|напісала|стварыла)\s+(.{3,80})", re.UNICODE),
        lambda m: f"Што напісаў {m.group(1).strip()}?",
        lambda m: m.group(0).strip(),
    ),
    (
        re.compile(r"(.{3,50})\s+—\s+гэта\s+(.{5,100})", re.UNICODE),
        lambda m: f"Што такое {m.group(1).strip()}?",
        lambda m: f"{m.group(1).strip()} — гэта {m.group(2).strip()}",
    ),
]

TRUE_FALSE_TEMPLATE = "Ці праўда наступнае сцвярджэнне?"

_NUMBER_PATTERN = re.compile(r"\b(\d+)\b")


def create_topic_samples(
    paragraphs: list[tuple[str, str]],
    max_samples: int,
    rng: random.Random,
) -> list[dict]:
    records = []

    for _filename, para in paragraphs:
        if len(para) < 200:
            continue

        sentences = split_into_sentences(para)
        if not sentences:
            continue

        topic_sentence = sentences[0]
        if len(topic_sentence) < 20:
            continue

        template = rng.choice(TOPIC_TEMPLATES)
        instruction = f"{template}\n\n{para}"
        records.append(make_conversation(instruction, topic_sentence))

        if len(records) >= max_samples:
            break

    return records


def create_factual_samples(
    paragraphs: list[tuple[str, str]],
    max_samples: int,
    rng: random.Random,
) -> list[dict]:
    records = []

    for _filename, para in paragraphs:
        sentences = split_into_sentences(para)

        for sentence in sentences:
            if len(sentence) < 30 or len(sentence) > 500:
                continue

            for pattern, question_fn, answer_fn in FACTUAL_PATTERNS:
                match = pattern.search(sentence)
                if match:
                    try:
                        question = question_fn(match)
                        answer = answer_fn(match)
                    except (IndexError, AttributeError):
                        continue

                    if len(question) < 10 or len(answer) < 10:
                        continue

                    records.append(make_conversation(question, answer))

                    if len(records) >= max_samples:
                        return records
                    break

    return records


def negate_sentence(sentence: str) -> str | None:
    negation_targets = [
        (re.compile(r"\b(быў|была|было|былі)\b"), r"не \1"),
        (re.compile(r"\b(з'яўляецца)\b"), r"не \1"),
        (re.compile(r"\b(мае|маюць)\b"), r"не \1"),
        (re.compile(r"\b(стаў|стала|сталі)\b"), r"не \1"),
        (re.compile(r"\b(ёсць)\b"), r"не \1"),
    ]

    for pattern, replacement in negation_targets:
        if pattern.search(sentence):
            return pattern.sub(replacement, sentence, count=1)

    return None


def modify_number(sentence: str, rng: random.Random) -> str | None:
    matches = list(_NUMBER_PATTERN.finditer(sentence))
    if not matches:
        return None

    match = rng.choice(matches)
    original_num = int(match.group(1))
    offset = rng.choice([10, 50, 100, -10, -50, -100])
    new_num = max(1, original_num + offset)

    if new_num == original_num:
        return None

    return sentence[:match.start()] + str(new_num) + sentence[match.end():]


def create_true_false_samples(
    paragraphs: list[tuple[str, str]],
    max_samples: int,
    rng: random.Random,
) -> list[dict]:
    records = []

    factual_sentences = []
    for _filename, para in paragraphs:
        for sentence in split_into_sentences(para):
            if 30 < len(sentence) < 300 and _NUMBER_PATTERN.search(sentence):
                factual_sentences.append(sentence)
            elif 30 < len(sentence) < 300:
                for pattern, _, _ in FACTUAL_PATTERNS:
                    if pattern.search(sentence):
                        factual_sentences.append(sentence)
                        break

    rng.shuffle(factual_sentences)

    for sentence in factual_sentences:
        if len(records) >= max_samples:
            break

        if rng.random() < 0.5:
            instruction = f"{TRUE_FALSE_TEMPLATE}\n\n{sentence}"
            response = f"Так, гэта праўда. {sentence}"
            records.append(make_conversation(instruction, response))
        else:
            modified = modify_number(sentence, rng)
            if modified is None:
                modified = negate_sentence(sentence)
            if modified is None:
                continue

            instruction = f"{TRUE_FALSE_TEMPLATE}\n\n{modified}"
            response = f"Не, гэта няпраўда. Насамрэч: {sentence}"
            records.append(make_conversation(instruction, response))

    return records


def main():
    parser = argparse.ArgumentParser(description="Create QA & comprehension dataset")
    parser.add_argument("--input-dir", type=str, default=str(INTERIM_DATA_DIR))
    parser.add_argument("--output-path", type=str, default=str(PROCESSED_DATA_DIR / "qa_dataset.jsonl"))
    parser.add_argument("--max-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    logger.info("Loading paragraphs...")
    paragraphs = list(load_all_paragraphs(args.input_dir))
    logger.info(f"Loaded {len(paragraphs)} paragraphs")
    rng.shuffle(paragraphs)

    topic_budget = int(args.max_samples * 0.35)
    factual_budget = int(args.max_samples * 0.35)
    tf_budget = int(args.max_samples * 0.30)

    logger.info(f"Creating topic/summary samples (budget: {topic_budget})...")
    topic_records = create_topic_samples(paragraphs, topic_budget, rng)
    logger.info(f"Created {len(topic_records)} topic samples")

    logger.info(f"Creating factual extraction samples (budget: {factual_budget})...")
    factual_records = create_factual_samples(paragraphs, factual_budget, rng)
    logger.info(f"Created {len(factual_records)} factual samples")

    logger.info(f"Creating true/false samples (budget: {tf_budget})...")
    tf_records = create_true_false_samples(paragraphs, tf_budget, rng)
    logger.info(f"Created {len(tf_records)} true/false samples")

    all_records = topic_records + factual_records + tf_records
    rng.shuffle(all_records)

    if len(all_records) > args.max_samples:
        all_records = all_records[: args.max_samples]

    logger.info(f"Total QA samples: {len(all_records)}")
    write_jsonl(all_records, args.output_path)


if __name__ == "__main__":
    main()
