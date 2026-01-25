import argparse
import logging
from pathlib import Path
import typing as t

from transformers import AutoTokenizer
from transformers.models.marian.tokenization_marian import load_json

from mova_be_llm.belarusian_tokenizer import BelarusianTokenizer
from mova_be_llm.examples import SAMPLE_BELARUSIAN_CORPUS
from mova_be_llm.loader import (
    load_cc100_dataset,
    load_directory,
    load_oscar_dataset,
    load_text_file,
)
from mova_be_llm.merge_tokenizers import TokenizerMerger
from mova_be_llm.types import MergeConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_VOCAB_SIZE = 32000
DEFAULT_MAX_NEW_TOKENS = 8000
DEFAULT_MIN_FREQUENCY = 5


def train_standalone_tokenizer(
    corpus: t.Iterator[str],
    vocab_size: int = DEFAULT_VOCAB_SIZE,
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
        tokenizer.save(Path(output_path))
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
    corpus: t.Iterator[str],
    base_model: str,
    output_path: str,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    min_frequency: int = DEFAULT_MIN_FREQUENCY,
) -> None:
    """
    Train Belarusian tokens and merge with a base model tokenizer.
    """

    logger.info(f"Merging with base model: {base_model}")

    merger = TokenizerMerger(base_model)
    config = MergeConfig(max_new_tokens=max_new_tokens, min_frequency=min_frequency, prioritize_morphemes=True)

    tokenizer = merger.merge_from_corpus(list(corpus), output_path, config)

    logger.info("\nTokenization comparison:")
    test_text = "Беларуская мова вельмі прыгожая."

    original = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    logger.info(f"  Text: '{test_text}'")
    logger.info(f"  Original: {original.tokenize(test_text)} ({len(original.encode(test_text)):,} tokens)")
    logger.info(f"  Merged:   {tokenizer.tokenize(test_text)} ({len(tokenizer.encode(test_text)):,} tokens)")


def main():
    parser = argparse.ArgumentParser(description="Train Belarusian tokenizer")

    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--corpus", type=str, help="Path to corpus file (text or JSONL) or directory")
    input_group.add_argument(
        "--dataset",
        type=str,
        choices=["oscar", "cc100", "sample"],
        help="Use a predefined dataset",
    )

    parser.add_argument("--output", "-o", type=str, required=True, help="Output path for the tokenizer")

    parser.add_argument(
        "--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE, help="Vocabulary size for standalone training"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["bpe", "unigram"],
        default="bpe",
        help="Tokenization algorithm",
    )
    parser.add_argument("--no-morphology", action="store_true", help="Disable morphological awareness")

    parser.add_argument("--merge", type=str, help="Base model to merge with (e.g., 'Qwen/Qwen2.5-1.5B')")
    parser.add_argument(
        "--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS, help="Maximum new tokens to add when merging"
    )

    parser.add_argument("--max-samples", type=int, help="Maximum samples to use from dataset")
    parser.add_argument("--text-field", type=str, default="text", help="Field name for text in JSONL files")

    args = parser.parse_args()

    if args.dataset:
        if args.dataset == "oscar":
            corpus = load_oscar_dataset(max_samples=args.max_samples)
        elif args.dataset == "cc100":
            corpus = load_cc100_dataset(max_samples=args.max_samples)
        elif args.dataset == "sample":
            logger.info("Using built-in sample corpus (for testing)")
            corpus = iter(SAMPLE_BELARUSIAN_CORPUS * 100)
        else:
            logger.error("Please provide --corpus or --dataset")
            return
    elif args.corpus:
        path = Path(args.corpus)
        if path.is_dir():
            corpus = load_directory(args.corpus)
        elif path.suffix == ".jsonl":
            corpus = load_json(args.corpus)
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
