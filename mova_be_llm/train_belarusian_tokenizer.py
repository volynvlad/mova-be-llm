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
from mova_be_llm.types import DEFAULT_MAX_NEW_TOKENS, DEFAULT_VOCAB_SIZE, MergeConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


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
    corpus: t.Iterator[str], output_path: str, base_model: str, config: MergeConfig, use_bytes: bool = False
) -> None:
    """
    Train Belarusian tokens and merge with a base model tokenizer.
    """

    logger.info(f"Merging with base model: {base_model} {'(byte-level)' if use_bytes else ''}")

    merger = TokenizerMerger(base_model, use_bytes=use_bytes)

    tokenizer = merger.merge_from_corpus(list(corpus), output_path, config)

    logger.info("\nTokenization comparison:")
    test_text = "Беларуская мова вельмі прыгожая."

    original = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    logger.info(f"  Text: '{test_text}'")

    # Get token IDs first, then decode for display
    original_ids = original.encode(test_text, add_special_tokens=False)
    merged_ids = tokenizer.encode(test_text, add_special_tokens=False)

    # For display, try to show readable tokens with fallback
    try:
        # Try to get readable tokens
        original_tokens = [original.decode([tid]) for tid in original_ids]
        merged_tokens = [tokenizer.decode([tid]) for tid in merged_ids]
    except Exception:
        # Fallback to convert_ids_to_tokens if decode fails
        original_tokens = original.convert_ids_to_tokens(original_ids)
        merged_tokens = tokenizer.convert_ids_to_tokens(merged_ids)

    logger.info(f"  Original: {original_tokens} ({len(original_ids):,} tokens)")
    logger.info(f"  Merged:   {merged_tokens} ({len(merged_ids):,} tokens)")


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
    parser.add_argument(
        "--use-bytes", action="store_true", help="Use byte-level tokenization for better Cyrillic capital handling"
    )

    parser.add_argument("--merge-model", type=str, help="Base model to merge with (e.g., 'Qwen/Qwen2.5-1.5B')")
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

    config = MergeConfig(max_new_tokens=args.max_new_tokens, prioritize_morphemes=True)
    if args.merge_model:
        train_and_merge_with_base(corpus, args.output, args.merge_model, config, use_bytes=args.use_bytes)
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
