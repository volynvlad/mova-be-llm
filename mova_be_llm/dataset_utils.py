import json
import os
import re
import typing as t
from pathlib import Path

from loguru import logger

from mova_be_llm.clean_txt import clean_text
from mova_be_llm.config import INTERIM_DATA_DIR

BELARUSIAN_STOP_WORDS = {
    "і", "ў", "на", "да", "з", "у", "не", "як", "гэта", "яна", "ён", "што",
    "але", "бо", "ці", "або", "дзе", "калі", "яго", "яе", "іх", "мы", "вы",
    "яны", "сябе", "свой", "гэты", "той", "які", "усе", "ўсе", "ўжо", "ужо",
    "так", "тут", "там", "тады", "потым", "яшчэ", "вось", "пра", "пад", "над",
    "без", "для", "пры", "між", "праз", "перад", "ад", "па", "за", "са",
}

BELARUSIAN_SPECIFIC_CHARS = set("ўіЎІ")
CYRILLIC_RANGE = re.compile(r"[а-яёіўА-ЯЁІЎ]")


def is_belarusian_text(text: str, threshold: float = 0.5) -> bool:
    if not text or len(text.strip()) < 10:
        return False

    alpha_chars = [c for c in text if c.isalpha()]
    if not alpha_chars:
        return False

    cyrillic_count = sum(1 for c in alpha_chars if CYRILLIC_RANGE.match(c))
    cyrillic_ratio = cyrillic_count / len(alpha_chars)

    if cyrillic_ratio < threshold:
        return False

    has_belarusian_chars = any(c in BELARUSIAN_SPECIFIC_CHARS for c in text)
    words = text.lower().split()
    has_stop_words = sum(1 for w in words if w in BELARUSIAN_STOP_WORDS) >= min(3, len(words) // 5)

    return has_belarusian_chars or has_stop_words


_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_PAGE_NUMBER = re.compile(r"^\s*\d{1,4}\s*$")
_HEADER_LIKE = re.compile(r"^[A-ZА-ЯЁІЎ\s\d]{3,50}$")


def split_into_paragraphs(text: str) -> list[str]:
    raw_paragraphs = text.split("\n\n")
    result = []
    for para in raw_paragraphs:
        para = _HYPHEN_BREAK.sub(r"\1\2", para)
        para = para.replace("\n", " ")
        para = re.sub(r"\s+", " ", para).strip()

        if not para:
            continue
        if _PAGE_NUMBER.match(para):
            continue
        if _HEADER_LIKE.match(para) and len(para) < 50:
            continue
        if is_belarusian_text(para):
            result.append(para)
    return result


_SENTENCE_BOUNDARY = re.compile(
    r'(?<=[.!?])\s+(?=[A-ZА-ЯЁІЎ])'
)
_ABBREVIATIONS = re.compile(
    r'\b(г|гг|р|рр|ст|стст|т|тт|м|н|в|с|д|п|ім|напр|гл|руб|тыс|млн|млрд)\.\s+'
)


def split_into_sentences(text: str) -> list[str]:
    zwsp = "\u200B"
    protected = _ABBREVIATIONS.sub(rf"\1.{zwsp}", text)

    initials_pattern = re.compile(r'([А-ЯЁІЎA-Z])\.\s+')
    protected = initials_pattern.sub(rf"\1.{zwsp}", protected)

    parts = _SENTENCE_BOUNDARY.split(protected)

    sentences = []
    for part in parts:
        s = part.replace(zwsp, " ").strip()
        s = re.sub(r"\s+", " ", s)
        if s:
            sentences.append(s)
    return sentences


def load_all_paragraphs(
    data_dir: str | Path | None = None,
    min_length: int = 100,
    min_words: int = 10,
    max_non_alpha_ratio: float = 0.5,
) -> t.Iterator[tuple[str, str]]:
    if data_dir is None:
        data_dir = INTERIM_DATA_DIR
    data_dir = Path(data_dir)

    txt_files = sorted(data_dir.glob("*.txt"))
    logger.info(f"Found {len(txt_files)} text files in {data_dir}")

    for filepath in txt_files:
        if filepath.stat().st_size == 0:
            continue

        try:
            content = filepath.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            logger.warning(f"Skipping {filepath.name}: encoding error")
            continue

        paragraphs = split_into_paragraphs(content)

        for para in paragraphs:
            cleaned = clean_text(para)
            if len(cleaned) < min_length:
                continue

            words = cleaned.split()
            if len(words) < min_words:
                continue

            alpha_count = sum(1 for c in cleaned if c.isalpha())
            if alpha_count == 0 or (len(cleaned) - alpha_count) / len(cleaned) > max_non_alpha_ratio:
                continue

            yield filepath.name, cleaned


def load_paragraphs_by_file(
    data_dir: str | Path | None = None,
    min_length: int = 100,
    min_words: int = 10,
    max_non_alpha_ratio: float = 0.5,
) -> t.Iterator[tuple[str, list[str]]]:
    if data_dir is None:
        data_dir = INTERIM_DATA_DIR
    data_dir = Path(data_dir)

    txt_files = sorted(data_dir.glob("*.txt"))
    logger.info(f"Found {len(txt_files)} text files in {data_dir}")

    for filepath in txt_files:
        if filepath.stat().st_size == 0:
            continue

        try:
            content = filepath.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            logger.warning(f"Skipping {filepath.name}: encoding error")
            continue

        file_paragraphs = []
        for para in split_into_paragraphs(content):
            cleaned = clean_text(para)
            if len(cleaned) < min_length:
                continue
            words = cleaned.split()
            if len(words) < min_words:
                continue
            alpha_count = sum(1 for c in cleaned if c.isalpha())
            if alpha_count == 0 or (len(cleaned) - alpha_count) / len(cleaned) > max_non_alpha_ratio:
                continue
            file_paragraphs.append(cleaned)

        if file_paragraphs:
            yield filepath.name, file_paragraphs


def make_conversation(instruction: str, response: str) -> dict:
    return {
        "conversations": [
            {"from": "human", "value": instruction},
            {"from": "gpt", "value": response},
        ]
    }


def write_jsonl(records: list[dict], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(f"Wrote {len(records)} records to {output_path}")
