#!/usr/bin/env python3
"""
Training Script for Belarusian Tokenizer

This script demonstrates:
1. Training a standalone Belarusian tokenizer
2. Merging with an existing LLM tokenizer (Qwen2.5, LLaMA, etc.)
3. Saving and loading the result

Usage:
    # Train standalone tokenizer
    python train_belarusian_tokenizer.py --corpus corpus.txt --output ./be_tokenizer

    # Merge with Qwen2.5
    python train_belarusian_tokenizer.py --corpus corpus.txt --merge Qwen/Qwen2.5-1.5B --output ./qwen_be

    # Download and use a Belarusian dataset
    python train_belarusian_tokenizer.py --dataset oscar --output ./be_tokenizer
"""

import argparse
import logging
from pathlib import Path
from typing import Iterator
import json
from glob import glob

from datasets import load_dataset
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from mova_be_llm.belarusian_tokenizer import BelarusianTokenizer, BPEConfig
from mova_be_llm.merge_tokenizers import TokenizerMerger, MergeConfig


def load_text_file(path: str) -> Iterator[str]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


def load_directory(path: str, pattern: str = "*.txt") -> Iterator[str]:
    for filepath in glob(str(Path(path) / pattern)):
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            for para in content.split("\n\n"):
                para = para.strip()
                if para:
                    yield para


def load_jsonl(path: str, text_field: str = "text") -> Iterator[str]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                if text_field in data:
                    yield data[text_field]
            except json.JSONDecodeError:
                continue


def load_oscar_dataset(
    lang: str = "be", split: str = "train", max_samples: int | None = None
) -> Iterator[str]:
    logger.info(f"Loading OSCAR dataset for language: {lang}")

    dataset = load_dataset(
        "oscar-corpus/OSCAR-2301",
        language=lang,
        split=split,
        streaming=True,
        trust_remote_code=True,
    )

    count = 0
    for item in dataset:
        yield item["text"]
        count += 1
        if max_samples and count >= max_samples:
            break
        if count % 10000 == 0:
            logger.info(f"Loaded {count} samples...")

    logger.info(f"Loaded {count} total samples from OSCAR")


def load_cc100_dataset(lang: str = "be", max_samples: int | None = None) -> Iterator[str]:
    logger.info(f"Loading CC-100 dataset for language: {lang}")

    dataset = load_dataset(
        "cc100",
        lang=lang,
        split="train",
        streaming=True,
    )

    count = 0
    for item in dataset:
        yield item["text"]
        count += 1
        if max_samples and count >= max_samples:
            break
        if count % 10000 == 0:
            logger.info(f"Loaded {count} samples...")

    logger.info(f"Loaded {count} total samples from CC-100")


def load_wikipedia_dataset(lang: str = "be", max_samples: int | None = None) -> Iterator[str]:
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets library required. Install with: pip install datasets")

    logger.info(f"Loading Wikipedia for language: {lang}")

    dataset = load_dataset(
        "wikipedia",
        language=lang,
        date="20231101",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    count = 0
    for item in dataset:
        yield item["text"]
        count += 1
        if max_samples and count >= max_samples:
            break
        if count % 1000 == 0:
            logger.info(f"Loaded {count} articles...")

    logger.info(f"Loaded {count} total articles from Wikipedia")


SAMPLE_BELARUSIAN_CORPUS = """
Беларуская мова — адна з усходнеславянскіх моў, нацыянальная мова беларусаў.
Яна з'яўляецца дзяржаўнай мовай Рэспублікі Беларусь разам з рускай мовай.
Беларуская мова мае багатую гісторыю, якая налічвае многія стагоддзі.

Беларуская літаратура пачала развівацца яшчэ ў часы Вялікага Княства Літоўскага.
Францыск Скарына быў першым беларускім і ўсходнеславянскім друкаром.
У 1517 годзе ён выдаў Біблію на старабеларускай мове.

Янка Купала і Якуб Колас — найвялікшыя беларускія паэты XX стагоддзя.
Іх творчасць аказала вялікі ўплыў на развіццё беларускай літаратуры.
Купала напісаў шмат вершаў аб каханні да роднай зямлі.

Мінск — сталіца і найбуйнейшы горад Рэспублікі Беларусь.
Горад размешчаны на рацэ Свіслач і мае насельніцтва каля двух мільёнаў чалавек.
Мінск з'яўляецца важным культурным, навуковым і эканамічным цэнтрам краіны.

Беларусь — краіна ў цэнтры Еўропы, якая мяжуе з Расіяй, Украінай, Польшчай, Літвой і Латвіяй.
Тэрыторыя Беларусі складае каля 207 тысяч квадратных кіламетраў.
Краіна вядомая сваімі лясамі, азёрамі і балотамі.

Белавежская пушча — адзін з найстарэйшых лясоў Еўропы.
Тут жывуць зубры — найбуйнейшыя еўрапейскія наземныя жывёлы.
Пушча з'яўляецца аб'ектам Сусветнай спадчыны ЮНЕСКА.

Традыцыйная беларуская кухня вельмі смачная і разнастайная.
Папулярнымі стравамі з'яўляюцца дранікі, мачанка, калдуны і верашчака.
Беларусы любяць ужываць бульбу, якую называюць другім хлебам.

Беларуская мова мае унікальную фанетыку, якая адрозніваецца ад рускай і ўкраінскай.
Характэрнымі асаблівасцямі з'яўляюцца дзеканне і цеканне, аканне і яканне.
У беларускай мове ёсць спецыфічныя літары: ў (у нескладовае) і і (і дзесяцінае).

Народная творчасць беларусаў вельмі багатая і разнастайная.
Беларускія народныя песні, казкі і легенды перадаюцца з пакалення ў пакаленне.
Традыцыйныя рамёствы ўключаюць ткацтва, ганчарства і разьбу па дрэве.

Сучасная Беларусь — гэта краіна з развітой прамысловасцю і сельскай гаспадаркай.
Беларускія трактары, грузавікі і аўтобусы вядомыя ва ўсім свеце.
IT-сектар краіны таксама актыўна развіваецца.

Адукацыя ў Беларусі з'яўляецца бясплатнай і абавязковай для дзяцей.
У краіне працуюць шматлікія ўніверсітэты і іншыя вышэйшыя навучальныя ўстановы.
Беларускі дзяржаўны ўніверсітэт — найстарэйшы і найбуйнейшы ВНУ краіны.

Спорт займае важнае месца ў жыцці беларусаў.
Краіна дасягнула значных поспехаў у біятлоне, лёгкай атлетыцы і хакеі.
Беларускія спартсмены рэгулярна ўдзельнічаюць у Алімпійскіх гульнях.

Прырода Беларусі надзвычай прыгожая і разнастайная.
У краіне налічваецца каля 10 тысяч азёраў і 20 тысяч рэк.
Найбуйнейшае возера — Нарач, а найдаўжэйшая рака — Дняпро.

Беларуская музыка мае глыбокія народныя карані.
Традыцыйнымі музычнымі інструментамі з'яўляюцца дуда, жалейка і цымбалы.
Сучасныя беларускія выканаўцы спалучаюць народныя матывы з сучаснымі стылямі.
""".strip().split("\n")


def train_standalone_tokenizer(
    corpus: Iterator[str],
    vocab_size: int = 32000,
    output_path: str | None = None,
    use_morphology: bool = True,
    algorithm: str = "bpe",
) -> BelarusianTokenizer:
    """
    Train a standalone Belarusian tokenizer.
    """
    logger.info(f"Training {algorithm.upper()} tokenizer with vocab_size={vocab_size}")

    tokenizer = BelarusianTokenizer(
        vocab_size=vocab_size,
        algorithm=algorithm,
        use_morphology=use_morphology,
    )

    def progress(current, target):
        if current % 1000 == 0:
            logger.info(f"Vocabulary: {current}/{target} ({100 * current / target:.1f}%)")

    corpus_list = list(corpus)
    logger.info(f"Training on {len(corpus_list)} text samples...")
    tokenizer.train(iter(corpus_list), progress_callback=progress)

    logger.info(f"Final vocabulary size: {tokenizer.vocab_size}")

    if output_path:
        tokenizer.save(output_path)
        logger.info(f"Saved tokenizer to {output_path}")

    test_texts = [
        "Беларуская мова вельмі прыгожая.",
        "Перакладчык працаваў над кнігай.",
        "Мінск — сталіца Беларусі.",
    ]

    logger.info("\nTokenization examples:")
    for text in test_texts:
        tokens = tokenizer.tokenize(text)
        logger.info(f"  '{text}'")
        logger.info(f"  -> {tokens}")

    return tokenizer


def train_and_merge_with_base(
    corpus: Iterator[str],
    base_model: str,
    output_path: str,
    max_new_tokens: int = 8000,
) -> None:
    """
    Train Belarusian tokens and merge with a base model tokenizer.
    """

    logger.info(f"Merging with base model: {base_model}")

    merger = TokenizerMerger(base_model)

    config = MergeConfig(
        max_new_tokens=max_new_tokens,
        min_frequency=5,
        prioritize_morphemes=True,
    )

    corpus_list = list(corpus)
    tokenizer = merger.merge_from_corpus(iter(corpus_list), output_path, config)

    logger.info("\nTokenization comparison:")
    test_text = "Беларуская мова вельмі прыгожая."

    original = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    logger.info(f"  Text: '{test_text}'")
    logger.info(
        f"  Original: {original.tokenize(test_text)} ({len(original.encode(test_text))} tokens)"
    )
    logger.info(
        f"  Merged:   {tokenizer.tokenize(test_text)} ({len(tokenizer.encode(test_text))} tokens)"
    )


def main():
    parser = argparse.ArgumentParser(description="Train Belarusian tokenizer")

    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--corpus", type=str, help="Path to corpus file (text or JSONL) or directory"
    )
    input_group.add_argument(
        "--dataset",
        type=str,
        choices=["oscar", "cc100", "wikipedia", "sample"],
        help="Use a predefined dataset",
    )

    parser.add_argument(
        "--output", "-o", type=str, required=True, help="Output path for the tokenizer"
    )

    parser.add_argument(
        "--vocab-size", type=int, default=32000, help="Vocabulary size for standalone training"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["bpe", "unigram"],
        default="bpe",
        help="Tokenization algorithm",
    )
    parser.add_argument(
        "--no-morphology", action="store_true", help="Disable morphological awareness"
    )

    parser.add_argument(
        "--merge", type=str, help="Base model to merge with (e.g., 'Qwen/Qwen2.5-1.5B')"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=8000, help="Maximum new tokens to add when merging"
    )

    parser.add_argument("--max-samples", type=int, help="Maximum samples to use from dataset")
    parser.add_argument(
        "--text-field", type=str, default="text", help="Field name for text in JSONL files"
    )

    args = parser.parse_args()

    if args.dataset:
        if args.dataset == "oscar":
            corpus = load_oscar_dataset(max_samples=args.max_samples)
        elif args.dataset == "cc100":
            corpus = load_cc100_dataset(max_samples=args.max_samples)
        elif args.dataset == "wikipedia":
            corpus = load_wikipedia_dataset(max_samples=args.max_samples)
        elif args.dataset == "sample":
            logger.info("Using built-in sample corpus (for testing)")
            corpus = iter(SAMPLE_BELARUSIAN_CORPUS * 100)
    elif args.corpus:
        path = Path(args.corpus)
        if path.is_dir():
            corpus = load_directory(args.corpus)
        elif path.suffix == ".jsonl":
            corpus = load_jsonl(args.corpus, args.text_field)
        else:
            corpus = load_text_file(args.corpus)
    else:
        logger.error("Please provide --corpus or --dataset")
        return

    if args.merge:
        train_and_merge_with_base(
            corpus,
            base_model=args.merge,
            output_path=args.output,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        train_standalone_tokenizer(
            corpus,
            vocab_size=args.vocab_size,
            output_path=args.output,
            use_morphology=not args.no_morphology,
            algorithm=args.algorithm,
        )


if __name__ == "__main__":
    main()
