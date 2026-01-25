"""
Merge Belarusian Tokenizer with Existing LLM Tokenizers (Qwen2.5, LLaMA, etc.)

This module provides utilities to extend existing tokenizers with Belarusian-specific
tokens while preserving the original vocabulary and token IDs.

Strategies:
1. Token Addition - Add new Belarusian tokens to existing vocab
2. Vocabulary Merging - Combine BPE merge rules
3. Embedding Extension - Extend model embeddings for new tokens
"""

import json
from pathlib import Path
import typing as t
from collections import Counter
import logging


import torch
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from mova_be_llm.belarusian_tokenizer import BelarusianPreTokenizer, BelarusianMorphology
from mova_be_llm.types import MergeConfig, TokenType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_SUBWORD_LENGTH = 15
MIN_FREQ = 5


class BelarusianTokenExtractor:
    def __init__(self, base_tokenizer=None):
        self.base_tokenizer = base_tokenizer
        self.pre_tokenizer = BelarusianPreTokenizer()
        self.morphology = BelarusianMorphology()

    def extract_candidate_tokens(
        self,
        texts: t.Iterator[str],
        max_subword_length: int = MAX_SUBWORD_LENGTH,
        min_freq: int = MIN_FREQ,
    ) -> dict[str, int]:
        word_freqs = Counter()
        token_freqs = Counter()

        for text in texts:
            for token, token_type in self.pre_tokenizer.pre_tokenize(text):
                if token_type == TokenType.word:
                    word = token.lower()
                    word_freqs[word] += 1

                    for i in range(len(word)):
                        for j in range(i + 1, min(i + max_subword_length + 1, len(word) + 1)):
                            substr = word[i:j]
                            if len(substr) >= 2:
                                token_freqs[substr] += 1

        for word, freq in word_freqs.items():
            if freq >= min_freq:
                token_freqs[word] = max(token_freqs.get(word, 0), freq)

        return {t: f for t, f in token_freqs.items() if f >= min_freq}

    def score_tokens(self, token_freqs: dict[str, int], prioritize_morphemes: bool = True) -> list[tuple[str, float]]:
        scored = []

        prefixes = set(self.morphology.PREFIXES)
        suffixes = set(self.morphology.SUFFIXES)
        endings = set(self.morphology.ENDINGS)

        for token, freq in token_freqs.items():
            if prioritize_morphemes:
                if token in prefixes:
                    freq *= 3.0
                elif token in suffixes:
                    freq *= 2.5
                elif token in endings:
                    freq *= 2.0
                if "ў" in token or "і" in token or "ё" in token:
                    freq *= 1.5
                if len(token) < 3 and token not in prefixes | suffixes | endings:
                    freq *= 0.5

            if self.base_tokenizer is not None:
                base_tokens = self.base_tokenizer.tokenize(token)
                if len(base_tokens) == 1 and base_tokens[0] == token:
                    freq *= 0.1
                elif len(base_tokens) > 3:
                    freq *= len(base_tokens) - 1

            scored.append((token, freq))

        scored.sort(key=lambda x: -x[1])
        return scored

    def get_fertility_improvement(self, texts: t.Iterator[str], new_tokens: set[str]) -> dict[str, float]:
        if self.base_tokenizer is None:
            return {}

        stats = {
            "original_tokens": 0,
            "original_words": 0,
            "improved_tokens": 0,
        }

        for text in texts:
            pre_tokens = self.pre_tokenizer.pre_tokenize(text)
            for token, token_type in pre_tokens:
                if token_type == TokenType.word:
                    stats["original_words"] += 1

                    base_tokens = self.base_tokenizer.tokenize(token)
                    stats["original_tokens"] += len(base_tokens)

                    word_lower = token.lower()
                    if word_lower in new_tokens:
                        stats["improved_tokens"] += 1
                    else:
                        stats["improved_tokens"] += len(base_tokens)

        if stats["original_words"] > 0:
            return {
                "original_fertility": stats["original_tokens"] / stats["original_words"],
                "estimated_fertility": stats["improved_tokens"] / stats["original_words"],
                "improvement": (stats["original_tokens"] - stats["improved_tokens"]) / stats["original_tokens"],
            }
        return {}


class TokenizerMerger:
    def __init__(self, base_tokenizer_name_or_path: str):
        """
        Args:
            base_tokenizer_name_or_path: HuggingFace model name or path
                e.g., "Qwen/Qwen2.5-7B", "meta-llama/Llama-2-7b-hf"
        """
        self.base_tokenizer = AutoTokenizer.from_pretrained(
            base_tokenizer_name_or_path,
            trust_remote_code=True,
        )
        self.base_vocab = set(self.base_tokenizer.get_vocab().keys())
        self.extractor = BelarusianTokenExtractor(self.base_tokenizer)

        logger.info(f"Loaded base tokenizer with {len(self.base_vocab)} tokens")

    def analyze_belarusian_coverage(self, texts: t.Iterator[str]) -> dict[str, int | list[str]]:
        """
        Analyze how well the base tokenizer handles Belarusian.
        """
        stats = {
            "total_words": 0,
            "total_tokens": 0,
            "unk_tokens": 0,
            "single_token_words": 0,
            "multi_token_words": 0,
            "avg_tokens_per_word": 0,
            "worst_words": [],  # Words with highest fragmentation
        }

        word_fragmentation = {}
        pre_tokenizer = BelarusianPreTokenizer()

        for text in texts:
            pre_tokens = pre_tokenizer.pre_tokenize(text)
            for token, token_type in pre_tokens:
                if token_type == TokenType.word:
                    stats["total_words"] += 1

                    encoded = self.base_tokenizer.encode(token, add_special_tokens=False)
                    num_tokens = len(encoded)
                    stats["total_tokens"] += num_tokens

                    unk_id = self.base_tokenizer.unk_token_id
                    if unk_id and unk_id in encoded:
                        stats["unk_tokens"] += 1

                    if num_tokens == 1:
                        stats["single_token_words"] += 1
                    else:
                        stats["multi_token_words"] += 1
                        word_fragmentation[token] = num_tokens

        if stats["total_words"] > 0:
            stats["avg_tokens_per_word"] = stats["total_tokens"] / stats["total_words"]

        worst = sorted(word_fragmentation.items(), key=lambda x: -x[1])[:20]
        stats["worst_words"] = [(w, n, self.base_tokenizer.tokenize(w)) for w, n in worst]

        return stats

    def select_tokens_to_add(
        self,
        texts: t.Iterator[str],
        config: MergeConfig | None = None,
    ) -> list[str]:
        """
        Select which Belarusian tokens to add to the vocabulary.

        Returns:
            List of tokens to add, in priority order
        """
        config = config or MergeConfig()

        logger.info("Extracting candidate tokens from corpus...")
        token_freqs = self.extractor.extract_candidate_tokens(
            texts,
            min_freq=config.min_frequency,
        )
        logger.info(f"Found {len(token_freqs)} candidate tokens")

        scored = self.extractor.score_tokens(
            token_freqs,
            prioritize_morphemes=config.prioritize_morphemes,
        )

        selected = []
        for token, _ in scored:
            if len(selected) >= config.max_new_tokens:
                break

            if config.skip_existing:
                forms_to_check = [
                    token,
                    f"▁{token}",  # SentencePiece style
                    f"Ġ{token}",  # GPT-2 style
                    f"##{token}",  # BERT style
                ]
                if any(form in self.base_vocab for form in forms_to_check):
                    continue

            selected.append(token)

        logger.info(f"Selected {len(selected)} new tokens to add")
        return selected

    def add_tokens(self, new_tokens: list[str], output_path: str | None = None) -> "PreTrainedTokenizerFast":
        """
        Add new tokens to the tokenizer.

        Args:
            new_tokens: List of tokens to add
            output_path: Where to save the merged tokenizer

        Returns:
            The extended tokenizer
        """
        num_added = self.base_tokenizer.add_tokens(new_tokens)
        logger.info(f"Added {num_added} new tokens to tokenizer")
        logger.info(f"New vocabulary size: {len(self.base_tokenizer)}")

        if output_path:
            self.base_tokenizer.save_pretrained(output_path)
            logger.info(f"Saved merged tokenizer to {output_path}")

            added_tokens_file = Path(output_path) / "added_belarusian_tokens.json"
            with open(added_tokens_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "num_added": num_added,
                        "tokens": new_tokens,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

        return self.base_tokenizer

    def merge_from_corpus(
        self,
        texts: list[str],
        output_path: str,
        config: MergeConfig | None = None,
    ) -> "PreTrainedTokenizerFast":
        config = config or MergeConfig()
        logger.info("Analyzing Belarusian coverage...")
        stats = self.analyze_belarusian_coverage(iter(texts))
        logger.info(f"Current avg tokens/word: {stats['avg_tokens_per_word']:.2f}")
        logger.info(f"Single-token words: {stats['single_token_words']:,}/{stats['total_words']:,}")

        tokens_to_add = self.select_tokens_to_add(iter(texts), config)
        return self.add_tokens(tokens_to_add, output_path)


def extend_model_embeddings(
    model,
    tokenizer,
    new_token_count: int,
    init_strategy: str = "mean",
):
    model.resize_token_embeddings(len(tokenizer))

    if hasattr(model, "get_input_embeddings"):
        embeddings = model.get_input_embeddings()
    else:
        raise ValueError("Model doesn't have get_input_embeddings method")

    with torch.no_grad():
        if init_strategy == "mean":
            old_embeddings = embeddings.weight[:-new_token_count]
            mean_embedding = old_embeddings.mean(dim=0)
            embeddings.weight[-new_token_count:] = mean_embedding

        elif init_strategy == "random":
            old_embeddings = embeddings.weight[:-new_token_count]
            std = old_embeddings.std()
            embeddings.weight[-new_token_count:] = torch.randn_like(embeddings.weight[-new_token_count:]) * std

        elif init_strategy == "copy_similar":
            logger.warning("copy_similar not fully implemented, falling back to mean")
            old_embeddings = embeddings.weight[:-new_token_count]
            mean_embedding = old_embeddings.mean(dim=0)
            embeddings.weight[-new_token_count:] = mean_embedding

    logger.info(f"Extended embeddings with {new_token_count} new tokens using '{init_strategy}' strategy")
    return model


class StandaloneMerger:
    def __init__(self, vocab_file: str, merges_file: str | None = None):
        with open(vocab_file, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)

        self.merges = []
        if merges_file and Path(merges_file).exists():
            with open(merges_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        parts = line.split()
                        if len(parts) >= 2:
                            self.merges.append((parts[0], parts[1]))

        logger.info(f"Loaded vocab with {len(self.vocab)} tokens")
        logger.info(f"Loaded {len(self.merges)} merge rules")

    def add_tokens(self, new_tokens: list[str]) -> dict[str, int]:
        """Add new tokens to vocabulary."""
        max_id = max(self.vocab.values())

        added = {}
        for token in new_tokens:
            if token not in self.vocab:
                max_id += 1
                self.vocab[token] = max_id
                added[token] = max_id

        logger.info(f"Added {len(added)} new tokens")
        return added

    def add_merges(self, new_merges: list[tuple[str, str]]):
        """Add new merge rules."""
        existing = set(self.merges)
        for merge in new_merges:
            if merge not in existing:
                self.merges.append(merge)

    def save(self, output_dir: Path):
        """Save merged tokenizer."""
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "vocab.json", "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

        with open(output_dir / "merges.txt", "w", encoding="utf-8") as f:
            for a, b in self.merges:
                f.write(f"{a} {b}\n")

        logger.info(f"Saved merged tokenizer to {output_dir}")


def merge_with_model(
    corpus_texts: t.Iterator[str],
    output_path: str,
    model_name: str,
    max_new_tokens: int = 8000,
) -> t.Optional["PreTrainedTokenizerFast"]:
    merger = TokenizerMerger(model_name)
    config = MergeConfig(max_new_tokens=max_new_tokens, prioritize_morphemes=True)
    return merger.merge_from_corpus(list(corpus_texts), output_path, config)


def main():
    print("=" * 60)
    print("Tokenizer Merging Example")
    print("=" * 60)

    sample_corpus = [
        "Беларуская мова — адна з усходнеславянскіх моў.",
        "Яна з'яўляецца дзяржаўнай мовай Рэспублікі Беларусь.",
        "Беларуская літаратура мае багатую гісторыю.",
        "Янка Купала і Якуб Колас — вялікія беларускія паэты.",
        "Мінск — сталіца і найбуйнейшы горад Беларусі.",
        "Беларусы — усходнеславянскі народ.",
        "Традыцыйная беларуская кухня вельмі смачная.",
        "Прырода Беларусі надзвычай прыгожая.",
        "Белавежская пушча — адзін з найстарэйшых лясоў Еўропы.",
        "Беларуская мова мае унікальную фанетыку і граматыку.",
    ] * 100

    print("\n1. Standalone Token Extraction (no transformers required):")
    print("-" * 40)

    extractor = BelarusianTokenExtractor()
    token_freqs = extractor.extract_candidate_tokens(iter(sample_corpus), min_freq=5)
    scored = extractor.score_tokens(token_freqs)

    print(f"Extracted {len(token_freqs)} candidate tokens")
    print("\nTop 20 tokens by score:")
    for token, score in scored[:20]:
        print(f"  {token:20s} score={score:.1f}")

    print("\n2. Full Merge with Qwen2.5 (requires transformers):")
    print("-" * 40)
    print("To merge with Qwen2.5:")
    print("""
    from merge_tokenizers import merge_with_model

    # Load your Belarusian corpus
    with open('corpus.txt', 'r') as f:
        texts = f.readlines()

    # Merge and save
    tokenizer = merge_with_model(
        iter(texts),
        model_name="Qwen/Qwen2.5-1.5B",
        output_path='./qwen2.5-belarusian',
        max_new_tokens=8000
    )

    # Use the merged tokenizer
    tokens = tokenizer.tokenize("Прывітанне, свет!")
        """)

    print("\nQuick analysis (loading small model)...")
    try:
        merger = TokenizerMerger("Qwen/Qwen2.5-0.5B")
        stats = merger.analyze_belarusian_coverage(iter(sample_corpus[:100]))
        print(f"  Avg tokens per Belarusian word: {stats['avg_tokens_per_word']:.2f}")
        print(
            f"  Single-token words: {stats['single_token_words']}/{stats['total_words']} ({100 * stats['single_token_words'] / max(1, stats['total_words']):.1f}%)"
        )
        print(f"\n  Most fragmented words:")
        for word, num_tokens, tokens in stats["worst_words"][:5]:
            print(f"    {word} -> {tokens} ({num_tokens} tokens)")
    except Exception as e:
        print(f"  Could not load model: {e}")
        print("  (This is expected if you don't have the model downloaded)")

    print("\n" + "=" * 60)
    print("Embedding Extension (after adding tokens to model):")
    print("=" * 60)
    print("""
    from merge_tokenizers import extend_model_embeddings
    from transformers import AutoModelForCausalLM

    # Load model and extended tokenizer
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B")
    tokenizer = AutoTokenizer.from_pretrained("./qwen2.5-belarusian")

    # Extend embeddings
    num_new_tokens = 8000  # Same as max_new_tokens used above
    model = extend_model_embeddings(
        model,
        tokenizer,
        new_token_count=num_new_tokens,
        init_strategy="mean"  # or "random"
    )

    # Now fine-tune the model on Belarusian data
    # The new embeddings will learn during training
    """)


if __name__ == "__main__":
    main()
