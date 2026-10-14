import argparse
import random
import re

from loguru import logger

from mova_be_llm.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from mova_be_llm.dataset_utils import (
    BELARUSIAN_STOP_WORDS,
    load_all_paragraphs,
    make_conversation,
    split_into_sentences,
    write_jsonl,
)

CORRECTION_TEMPLATES = [
    "Выправі памылкі ў тэксце:",
    "Знайдзі і выправі русізмы:",
    "Перапішы тэкст без памылак:",
    "Выправі граматычныя памылкі:",
    "Напішы правільна:",
    "Знайдзі і выправі памылкі:",
]

CLOZE_TEMPLATES = [
    "Устаў прапушчаныя словы:",
    "Запоўні пропускі ў тэксце:",
    "Дапішы прапушчаныя словы:",
    "Устаў патрэбныя словы замест пропускаў:",
]

SPELLING_TEMPLATES = [
    "Як правільна напісаць: '{wrong}' ці '{right}'?",
    "Якая форма правільная: '{wrong}' ці '{right}'?",
    "Выберы правільны варыянт: '{wrong}' ці '{right}'?",
]

RUSSIFICATION_RULES = [
    (re.compile(r"ў"), "у", "ў → у"),
    (re.compile(r"Ў"), "У", "Ў → У"),
    (re.compile(r"дзе"), "де", "дзеканне"),
    (re.compile(r"Дзе"), "Де", "дзеканне"),
    (re.compile(r"дзі"), "ді", "дзеканне"),
    (re.compile(r"Дзі"), "Ді", "дзеканне"),
    (re.compile(r"дзя"), "дя", "дзеканне"),
    (re.compile(r"дзю"), "дю", "дзеканне"),
    (re.compile(r"дзё"), "дё", "дзеканне"),
    (re.compile(r"це(?=[^н])"), "те", "цеканне"),
    (re.compile(r"ці"), "ті", "цеканне"),
    (re.compile(r"ця"), "тя", "цеканне"),
    (re.compile(r"цю"), "тю", "цеканне"),
    (re.compile(r"цё"), "тё", "цеканне"),
    (re.compile(r"(?<=[а-яёіў])і(?=[а-яёіў])"), "и", "і → и"),
    (re.compile(r"шч"), "щ", "шч → щ"),
    (re.compile(r"Шч"), "Щ", "шч → щ"),
    (re.compile(r"(?<=\w)'(?=\w)"), "", "apostrophe removal"),
    (re.compile(r"(?<=\w)'(?=\w)"), "", "apostrophe removal"),
    (re.compile(r"тэ"), "те", "э → е"),
    (re.compile(r"вя"), "ве", "яканне"),
]


def introduce_errors(text: str, error_rate: float, rng: random.Random) -> tuple[str, int]:
    result = text
    error_count = 0

    rules = list(RUSSIFICATION_RULES)
    rng.shuffle(rules)

    for pattern, replacement, _rule_name in rules:
        matches = list(pattern.finditer(result))
        if not matches:
            continue

        for match in reversed(matches):
            if rng.random() < error_rate:
                start, end = match.start(), match.end()
                result = result[:start] + replacement + result[end:]
                error_count += 1

    return result, error_count


def create_correction_samples(
    paragraphs: list[tuple[str, str]],
    error_rate: float,
    max_samples: int,
    rng: random.Random,
) -> list[dict]:
    records = []

    for _filename, para in paragraphs:
        sentences = split_into_sentences(para)
        for sentence in sentences:
            if len(sentence) < 30 or len(sentence) > 500:
                continue

            corrupted, error_count = introduce_errors(sentence, error_rate, rng)
            if error_count == 0 or corrupted == sentence:
                continue

            template = rng.choice(CORRECTION_TEMPLATES)
            instruction = f"{template}\n\n{corrupted}"
            records.append(make_conversation(instruction, sentence))

            if len(records) >= max_samples:
                return records

    return records


def create_cloze_samples(
    paragraphs: list[tuple[str, str]],
    max_samples: int,
    rng: random.Random,
) -> list[dict]:
    records = []

    for _filename, para in paragraphs:
        sentences = split_into_sentences(para)
        for sentence in sentences:
            words = sentence.split()
            if len(words) < 8 or len(sentence) > 500:
                continue

            content_words = [
                (i, w) for i, w in enumerate(words)
                if len(w) > 4
                and w.lower().strip(".,!?;:()\"'") not in BELARUSIAN_STOP_WORDS
                and w[0].islower()
            ]

            if len(content_words) < 2:
                continue

            num_masks = rng.randint(1, min(3, len(content_words)))
            masked_indices = rng.sample(content_words, num_masks)

            masked_words = list(words)
            for idx, _word in masked_indices:
                masked_words[idx] = "____"

            masked_sentence = " ".join(masked_words)
            template = rng.choice(CLOZE_TEMPLATES)
            instruction = f"{template}\n\n{masked_sentence}"
            records.append(make_conversation(instruction, sentence))

            if len(records) >= max_samples:
                return records

    return records


SPELLING_PAIRS = [
    ("дзеці", "деці"),
    ("дзякуй", "дякуй"),
    ("цікавы", "тікавы"),
    ("цішыня", "тішыня"),
    ("шчасце", "щасце"),
    ("кніга", "книга"),
    ("тэкст", "текст"),
    ("вясна", "весна"),
    ("з'ява", "зява"),
    ("аб'ём", "абём"),
    ("дзяўчына", "дявчына"),
    ("цвёрды", "твёрды"),
    ("ідзе", "іде"),
    ("цёплы", "тёплы"),
    ("дзверы", "дверы"),
    ("цэгла", "тэгла"),
    ("шчодры", "щодры"),
    ("дзіўна", "діўна"),
    ("цягнік", "тягнік"),
    ("пішчаць", "піщаць"),
]


def create_spelling_samples(
    max_samples: int,
    rng: random.Random,
) -> list[dict]:
    records = []

    for correct, incorrect in SPELLING_PAIRS:
        for template_str in SPELLING_TEMPLATES:
            if rng.random() < 0.5:
                instruction = template_str.format(wrong=incorrect, right=correct)
            else:
                instruction = template_str.format(wrong=correct, right=incorrect)

            response = f"Правільна: '{correct}'. Гэта беларуская форма напісання."
            records.append(make_conversation(instruction, response))

            if len(records) >= max_samples:
                return records

    return records


def main():
    parser = argparse.ArgumentParser(description="Create grammar & vocabulary dataset")
    parser.add_argument("--input-dir", type=str, default=str(INTERIM_DATA_DIR))
    parser.add_argument("--output-path", type=str, default=str(PROCESSED_DATA_DIR / "grammar_dataset.jsonl"))
    parser.add_argument("--max-samples", type=int, default=30000)
    parser.add_argument("--error-rate", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    logger.info("Loading paragraphs...")
    paragraphs = list(load_all_paragraphs(args.input_dir))
    logger.info(f"Loaded {len(paragraphs)} paragraphs")
    rng.shuffle(paragraphs)

    correction_budget = int(args.max_samples * 0.60)
    cloze_budget = int(args.max_samples * 0.25)
    spelling_budget = int(args.max_samples * 0.15)

    logger.info(f"Creating correction samples (budget: {correction_budget})...")
    correction_records = create_correction_samples(paragraphs, args.error_rate, correction_budget, rng)
    logger.info(f"Created {len(correction_records)} correction samples")

    logger.info(f"Creating cloze samples (budget: {cloze_budget})...")
    cloze_records = create_cloze_samples(paragraphs, cloze_budget, rng)
    logger.info(f"Created {len(cloze_records)} cloze samples")

    logger.info(f"Creating spelling samples (budget: {spelling_budget})...")
    spelling_records = create_spelling_samples(spelling_budget, rng)
    logger.info(f"Created {len(spelling_records)} spelling samples")

    all_records = correction_records + cloze_records + spelling_records
    rng.shuffle(all_records)

    if len(all_records) > args.max_samples:
        all_records = all_records[: args.max_samples]

    logger.info(f"Total grammar samples: {len(all_records)}")
    write_jsonl(all_records, args.output_path)


if __name__ == "__main__":
    main()
