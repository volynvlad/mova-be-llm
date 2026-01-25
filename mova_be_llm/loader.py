import json
from glob import glob
import typing as t
from pathlib import Path
import logging

from datasets import load_dataset
from rich.progress import track

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_text_file(path: str) -> t.Iterator[str]:
    with open(path, "r", encoding="utf-8") as f:
        for line in track(f, description=f"Loading from text file: {path}"):
            line = line.strip()
            if line:
                yield line


def load_directory(path: str, pattern: str = "*.txt") -> t.Iterator[str]:
    for filepath in track(glob(str(Path(path) / pattern)), description=f"Loading directory from directory: {path}"):
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            for para in content.split("\n\n"):
                para = para.strip()
                if para:
                    yield para


def load_jsonl(path: str, text_field: str = "text") -> t.Iterator[str]:
    with open(path, "r", encoding="utf-8") as f:
        for line in track(f, description=f"Loading from jsonl file: {path}"):
            try:
                data = json.loads(line)
                if text_field in data:
                    yield data[text_field]
            except json.JSONDecodeError:
                continue


def load_oscar_dataset(lang: str = "be", split: str = "train", max_samples: int | None = None) -> t.Iterator[str]:
    dataset = load_dataset(
        "oscar-corpus/OSCAR-2301",
        language=lang,
        split=split,
        streaming=True,
    )

    count = 0
    for item in track(dataset, description=f"Loading OSCAR dataset for language: {lang}"):
        yield item["text"]
        count += 1
        if max_samples and count >= max_samples:
            break

    logger.info(f"Loaded {count} total samples from OSCAR")


# TODO change name to c4
def load_cc100_dataset(lang: str = "be", split: str = "train", max_samples: int | None = None) -> t.Iterator[str]:
    logger.info(f"Loading CC-100 dataset for language: {lang}")

    dataset = load_dataset(
        "allenai/c4",
        lang,
        split=split,
        streaming=True,
    )

    count = 0
    for item in track(dataset, description=f"Loading cc100 dataset for language: {lang}"):
        yield item["text"]
        count += 1
        if max_samples and count >= max_samples:
            break

    logger.info(f"Loaded {count} total samples from CC-100")
