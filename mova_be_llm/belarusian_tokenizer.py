import math
import re
import json
from collections import Counter
from typing import List, Dict, Tuple, Optional, Set, Iterator
from dataclasses import dataclass, field
from pathlib import Path


class BelarusianMorphology:
    PREFIXES = [
        # Sorted by length (longest first) for greedy matching
        "недо",
        "пера",
        "пры",
        "раз",
        "без",
        "між",
        "над",
        "пад",
        "пра",
        "пре",
        "про",
        "суп",
        "уз",
        "аб",
        "ад",
        "ва",
        "вы",
        "да",
        "за",
        "із",
        "на",
        "па",
        "пе",
        "пі",
        "по",
        "са",
        "со",
        "су",
        "а",
        "в",
        "з",
        "с",
        "у",
        "о",
    ]

    SUFFIXES = [
        # Noun suffixes
        "ацыя",
        "енне",
        "анне",
        "ванн",
        "асць",
        "сць",
        "ства",
        "цтва",
        "шчык",
        "чык",
        "нік",
        "льн",
        "цель",
        "ель",
        "ар",
        "ер",
        "ір",
        "ізм",
        "іст",
        "ант",
        "ент",
        "ат",
        "ёт",
        # Diminutive suffixes
        "ачк",
        "ечк",
        "ічк",
        "очк",
        "ушк",
        "юшк",
        "еньк",
        "аньк",
        "оньк",
        "ік",
        "ок",
        "ек",
        "ак",
        "ка",
        # Augmentative
        "іск",
        "ышч",
        "ішч",
        # Adjective suffixes
        "ічн",
        "ацк",
        "енн",
        "анн",
        "оўн",
        "аўн",
        "льн",
        "ічн",
        "ав",
        "ов",
        "ев",
        "ёв",
        "ын",
        "ін",
        "ск",
        "цк",
        # Verb suffixes
        "ава",
        "ява",
        "ова",
        "ыва",
        "іва",
        "ну",
        "ва",
        # Abstract noun
        "аўк",
        "оўк",
        "еўк",
    ]

    ENDINGS = [
        # Noun endings (by case and number)
        "амі",
        "ямі",
        "аму",
        "яму",
        "ага",
        "яга",
        "ым",
        "ім",
        "ам",
        "ям",
        "ах",
        "ях",
        "оў",
        "ёў",
        "эў",
        "еў",
        "ай",
        "яй",
        "ой",
        "ей",
        "ою",
        "ею",
        "ую",
        "юю",
        "ае",
        "яе",
        "ое",
        "ее",
        "ыя",
        "ія",
        # Verb endings
        "юць",
        "у|ць",
        "яць",
        "аць",
        "іць",
        "ыць",
        "эць",
        "ець",
        "еш",
        "аш",
        "іш",
        "ыш",
        "юш",
        "ем",
        "ім",
        "ым",
        "ам",
        "лі",
        "ла",
        "ло",
        "лы",
        "ць",
        "чы",
        "шы",
        "це",
        "ше",
        # Short endings
        "а",
        "я",
        "о",
        "ё",
        "е",
        "э",
        "у",
        "ю",
        "ы",
        "і",
        "й",
        "ў",
    ]

    CYRILLIC_LOWER = "абвгдеёжзійклмнопрстуўфхцчшыьэюя"
    CYRILLIC_UPPER = "АБВГДЕЁЖЗІЙКЛМНОПРСТУЎФХЦЧШЫЬЭЮЯ"
    CYRILLIC_ALL = CYRILLIC_LOWER + CYRILLIC_UPPER

    VOWELS = set("аеёіоуыэюяАЕЁІОУЫЭЮЯ")
    CONSONANTS = set("бвгджзйклмнпрстўфхцчшьБВГДЖЗЙКЛМНПРСТЎФХЦЧШЬ")

    WORD_PATTERN = re.compile(
        r"^(?P<prefix>(?:недо|пера|пры|раз|без|між|над|пад|пра|пре|про|суп|"
        r"уз|аб|ад|ва|вы|да|за|із|на|па|пе|пі|по|са|со|су|а|в|з|с|у|о)?)"
        r"(?P<root>[а-яёіўА-ЯЁІЎ]{2,}?)"
        r"(?P<suffix>(?:ацыя|енне|анне|ванн|асць|сць|ства|цтва|шчык|чык|нік|"
        r"льн|цель|ель|ізм|іст|ант|ент|ачк|ечк|ічк|очк|ушк|юшк|еньк|аньк|оньк|"
        r"ік|ок|ек|ак|ка|іск|ышч|ішч|ічн|ацк|енн|анн|оўн|аўн|ава|ява|ова|ыва|"
        r"іва|ну|ва|аўк|оўк|еўк|ар|ер|ір|ат|ёт|ав|ов|ев|ёв|ын|ін|ск|цк)?)"
        r"(?P<ending>(?:амі|ямі|аму|яму|ага|яга|юць|у|ць|яць|аць|іць|ыць|эць|ець|"
        r"ам|ям|ах|ях|оў|ёў|эў|еў|ай|яй|ой|ей|ою|ею|ую|юю|ае|яе|ое|ее|ыя|ія|"
        r"еш|аш|іш|ыш|юш|ем|ім|ым|лі|ла|ло|лы|ць|чы|шы|це|ше|"
        r"а|я|о|ё|е|э|у|ю|ы|і|й|ў)?)$",
        re.IGNORECASE | re.UNICODE,
    )


class BelarusianPreTokenizer:
    SPLIT_PATTERN = re.compile(
        r"""
        # URLs
        (?P<url>https?://[^\s]+)|
        # Email addresses
        (?P<email>[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})|
        # Numbers with optional decimal/thousand separators
        (?P<number>-?[\d]+(?:[.,][\d]+)*)|
        # Belarusian words (including apostrophe for words like з'ява)
        (?P<word>[а-яёіўА-ЯЁІЎ]+(?:'[а-яёіўА-ЯЁІЎ]+)?)|
        # Latin words
        (?P<latin>[a-zA-Z]+)|
        # Punctuation (single or repeated)
        (?P<punct>[^\s\w]+)|
        # Whitespace (preserved for reconstruction)
        (?P<space>\s+)
        """,
        re.VERBOSE | re.UNICODE,
    )

    def __init__(self, lowercase: bool = False):
        self.lowercase = lowercase

    def pre_tokenize(self, text: str) -> List[Tuple[str, str]]:
        tokens = []
        for match in self.SPLIT_PATTERN.finditer(text):
            for name, value in match.groupdict().items():
                if value is not None:
                    if self.lowercase and name in ("word", "latin"):
                        value = value.lower()
                    tokens.append((value, name))
                    break
        return tokens

    def tokenize(self, text: str) -> List[str]:
        return [token for token, _ in self.pre_tokenize(text)]


class MorphologicalTokenizer:
    def __init__(self):
        self.morphology = BelarusianMorphology()
        # Sort by length (longest first) for greedy matching
        self.prefixes = sorted(self.morphology.PREFIXES, key=len, reverse=True)
        self.suffixes = sorted(self.morphology.SUFFIXES, key=len, reverse=True)
        self.endings = sorted(self.morphology.ENDINGS, key=len, reverse=True)

    def segment(self, word: str) -> Dict[str, str]:
        original = word
        word_lower = word.lower()

        result = {"prefix": "", "root": "", "suffix": "", "ending": ""}

        match = self.morphology.WORD_PATTERN.match(word_lower)
        if match and match.group("root"):
            groups = match.groupdict()
            start = len(groups["prefix"])
            end = start + len(groups["root"])
            result["prefix"] = original[:start]
            result["root"] = original[start:end]
            result["suffix"] = groups["suffix"]
            result["ending"] = groups["ending"]
            return result

        remaining = word_lower
        pos = 0

        for prefix in self.prefixes:
            if remaining.startswith(prefix) and len(remaining) > len(prefix) + 2:
                result["prefix"] = original[: len(prefix)]
                remaining = remaining[len(prefix) :]
                pos = len(prefix)
                break

        for ending in self.endings:
            if remaining.endswith(ending) and len(remaining) > len(ending) + 2:
                result["ending"] = ending
                remaining = remaining[: -len(ending)]
                break

        for suffix in self.suffixes:
            if remaining.endswith(suffix) and len(remaining) > len(suffix) + 1:
                result["suffix"] = suffix
                remaining = remaining[: -len(suffix)]
                break

        root_end = len(original) - len(result["ending"]) - len(result["suffix"])
        result["root"] = original[pos:root_end] if root_end > pos else original[pos:]
        return result

    def segment_to_tokens(self, word: str, mark_boundaries: bool = True) -> List[str]:
        segments = self.segment(word)
        tokens = []

        if mark_boundaries:
            if segments["prefix"]:
                tokens.append(f"▸{segments['prefix']}")
            if segments["root"]:
                tokens.append(segments["root"])
            if segments["suffix"]:
                tokens.append(f"◂{segments['suffix']}")
            if segments["ending"]:
                tokens.append(f"▪{segments['ending']}")
        else:
            tokens = [v for v in segments.values() if v]

        return tokens if tokens else [word]


@dataclass
class BPEConfig:
    vocab_size: int = 32000
    min_frequency: int = 2
    special_tokens: List[str] = field(
        default_factory=lambda: [
            "<pad>",
            "<unk>",
            "<bos>",
            "<eos>",
            "<mask>",
            "<sep>",
            "<cls>",
            "<nl>",
        ]
    )
    morpheme_boundary_bonus: float = 0.1
    use_morphology: bool = True
    lowercase: bool = False
    continuing_subword_prefix: str = "##"


class BelarusianBPETokenizer:
    def __init__(self, config: Optional[BPEConfig] = None):
        self.config = config or BPEConfig()
        self.pre_tokenizer = BelarusianPreTokenizer(lowercase=self.config.lowercase)
        self.morphological = MorphologicalTokenizer()

        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        self.merge_ranks: Dict[Tuple[str, str], int] = {}

        self._init_special_tokens()

        self._cache: Dict[str, List[str]] = {}
        self._cache_max_size = 100000

    def _init_special_tokens(self):
        for i, token in enumerate(self.config.special_tokens):
            self.token_to_id[token] = i
            self.id_to_token[i] = token

    def _get_base_vocab(self) -> Set[str]:
        vocab = set()

        for char in BelarusianMorphology.CYRILLIC_ALL:
            vocab.add(char)

        for char in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789":
            vocab.add(char)

        for char in '.,!?;:\'"()-–—«»„"…':
            vocab.add(char)

        if self.config.use_morphology:
            morph = BelarusianMorphology()
            for prefix in morph.PREFIXES[:15]:
                vocab.add(f"▸{prefix}")
            for suffix in morph.SUFFIXES[:20]:
                vocab.add(f"◂{suffix}")
            for ending in morph.ENDINGS[:25]:
                vocab.add(f"▪{ending}")
        return vocab

    def _word_to_chars(self, word: str) -> List[str]:
        if self.config.use_morphology:
            segments = self.morphological.segment(word)
            chars = []

            if segments["prefix"]:
                chars.append(f"▸{segments['prefix']}")

            if segments["root"]:
                root_chars = list(segments["root"])
                if root_chars:
                    chars.append(root_chars[0])
                    for c in root_chars[1:]:
                        chars.append(f"{self.config.continuing_subword_prefix}{c}")

            if segments["suffix"]:
                chars.append(f"◂{segments['suffix']}")

            if segments["ending"]:
                chars.append(f"▪{segments['ending']}")

            return chars if chars else list(word)

        if not word:
            return []
        chars = [word[0]]
        for c in word[1:]:
            chars.append(f"{self.config.continuing_subword_prefix}{c}")
        return chars

    def _get_pairs(self, tokens: List[str]) -> Counter:
        pairs = Counter()
        for i in range(len(tokens) - 1):
            pairs[(tokens[i], tokens[i + 1])] += 1
        return pairs

    def _merge_pair(self, tokens: List[str], pair: Tuple[str, str]) -> List[str]:
        merged = pair[0] + pair[1].lstrip(self.config.continuing_subword_prefix)
        result = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                result.append(merged)
                i += 2
            else:
                result.append(tokens[i])
                i += 1
        return result

    def train(self, texts: Iterator[str], progress_callback=None):
        word_freqs: Counter = Counter()

        for text in texts:
            tokens = self.pre_tokenizer.pre_tokenize(text)
            for token, token_type in tokens:
                if token_type == "word":
                    word_freqs[token] += 1

        word_splits: Dict[str, List[str]] = {}
        for word in word_freqs:
            word_splits[word] = self._word_to_chars(word)

        vocab = self._get_base_vocab()
        for word, chars in word_splits.items():
            vocab.update(chars)

        target_size = self.config.vocab_size - len(self.config.special_tokens)

        while len(vocab) < target_size:
            pair_freqs: Counter = Counter()
            for word, chars in word_splits.items():
                freq = word_freqs[word]
                for i in range(len(chars) - 1):
                    pair = (chars[i], chars[i + 1])
                    pair_freqs[pair] += freq

            if not pair_freqs:
                break

            best_pair = None
            best_freq = self.config.min_frequency - 1

            for pair, freq in pair_freqs.items():
                if freq > best_freq:
                    effective_freq = freq
                    if self.config.use_morphology:
                        if (
                            pair[0].startswith("▸")
                            or pair[0].startswith("◂")
                            or pair[0].startswith("▪")
                            or pair[1].startswith("▸")
                            or pair[1].startswith("◂")
                            or pair[1].startswith("▪")
                        ):
                            effective_freq *= 1 + self.config.morpheme_boundary_bonus

                    if effective_freq > best_freq:
                        best_freq = freq
                        best_pair = pair

            if best_pair is None or best_freq < self.config.min_frequency:
                break

            merged = best_pair[0] + best_pair[1].lstrip(self.config.continuing_subword_prefix)
            vocab.add(merged)
            self.merges.append(best_pair)

            for word in word_splits:
                word_splits[word] = self._merge_pair(word_splits[word], best_pair)

            if progress_callback:
                progress_callback(len(vocab), target_size)

        self._build_vocab(vocab)

    def _build_vocab(self, vocab: Set[str]):
        idx = len(self.config.special_tokens)

        for token in sorted(vocab):
            if token not in self.token_to_id:
                self.token_to_id[token] = idx
                self.id_to_token[idx] = token
                idx += 1

        self.merge_ranks = {merge: i for i, merge in enumerate(self.merges)}

    def _apply_bpe(self, word: str) -> List[str]:
        """Apply BPE to a single word."""
        if word in self._cache:
            return self._cache[word]

        tokens = self._word_to_chars(word)

        if len(tokens) <= 1:
            return tokens

        while True:
            best_pair = None
            best_rank = float("inf")

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self.merge_ranks:
                    rank = self.merge_ranks[pair]
                    if rank < best_rank:
                        best_rank = rank
                        best_pair = pair

            if best_pair is None:
                break

            tokens = self._merge_pair(tokens, best_pair)

        if len(self._cache) < self._cache_max_size:
            self._cache[word] = tokens

        return tokens

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into subword tokens.

        Args:
            text: Input text

        Returns:
            List of subword tokens
        """
        result = []
        pre_tokens = self.pre_tokenizer.pre_tokenize(text)

        for token, token_type in pre_tokens:
            if token_type == "word":
                result.extend(self._apply_bpe(token))
            elif token_type == "latin":
                result.extend(self._apply_bpe(token))
            elif token_type == "number":
                result.append(token)
            elif token_type == "space":
                if token == "\n":
                    result.append("<nl>")
            elif token_type == "punct":
                result.append(token)
            elif token_type in ("url", "email"):
                result.append(token)

        return result

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text
            add_special_tokens: Whether to add BOS/EOS tokens

        Returns:
            List of token IDs
        """
        tokens = self.tokenize(text)

        ids = []
        if add_special_tokens:
            ids.append(self.token_to_id["<bos>"])

        unk_id = self.token_to_id["<unk>"]
        for token in tokens:
            ids.append(self.token_to_id.get(token, unk_id))

        if add_special_tokens:
            ids.append(self.token_to_id["<eos>"])

        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text.

        Args:
            ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded text string
        """
        tokens = []
        special = set(self.config.special_tokens)

        for id_ in ids:
            token = self.id_to_token.get(id_, "<unk>")
            if skip_special_tokens and token in special:
                continue
            tokens.append(token)

        def strip_markers(s: str) -> str:
            """Remove all morpheme boundary markers."""
            return s.replace("▸", "").replace("◂", "").replace("▪", "")

        result = []
        for token in tokens:
            if token == "<nl>":
                result.append("\n")
            elif token.startswith(self.config.continuing_subword_prefix):
                clean = strip_markers(token[len(self.config.continuing_subword_prefix) :])
                result.append(clean)
            else:
                clean = strip_markers(token)
                if result and clean:
                    prev = result[-1]
                    needs_space = (
                        prev
                        and prev[-1].isalnum()
                        and clean[0].isalnum()
                        and not prev.endswith(("-", "'", "(", "[", "{", "«", '"'))
                    )
                    if needs_space:
                        result.append(" ")
                result.append(clean)

        return "".join(result)

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self.token_to_id)

    def save(self, path: str):
        """Save tokenizer to directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        config_dict = {
            "vocab_size": self.config.vocab_size,
            "min_frequency": self.config.min_frequency,
            "special_tokens": self.config.special_tokens,
            "morpheme_boundary_bonus": self.config.morpheme_boundary_bonus,
            "use_morphology": self.config.use_morphology,
            "lowercase": self.config.lowercase,
            "continuing_subword_prefix": self.config.continuing_subword_prefix,
        }
        with open(path / "config.json", "w", encoding="utf-8") as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)

        with open(path / "vocab.json", "w", encoding="utf-8") as f:
            json.dump(self.token_to_id, f, ensure_ascii=False, indent=2)

        merges_str = [f"{a} {b}" for a, b in self.merges]
        with open(path / "merges.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(merges_str))

    @classmethod
    def load(cls, path: str) -> "BelarusianBPETokenizer":
        """Load tokenizer from directory."""
        path = Path(path)

        with open(path / "config.json", "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        config = BPEConfig(**config_dict)

        tokenizer = cls(config)

        with open(path / "vocab.json", "r", encoding="utf-8") as f:
            tokenizer.token_to_id = json.load(f)
        tokenizer.id_to_token = {v: k for k, v in tokenizer.token_to_id.items()}

        with open(path / "merges.txt", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(" ", 1)
                    if len(parts) == 2:
                        tokenizer.merges.append((parts[0], parts[1]))
        tokenizer.merge_ranks = {merge: i for i, merge in enumerate(tokenizer.merges)}

        return tokenizer


class BelarusianUnigramTokenizer:
    def __init__(self, config: Optional[BPEConfig] = None):
        self.config = config or BPEConfig()
        self.pre_tokenizer = BelarusianPreTokenizer(lowercase=self.config.lowercase)
        self.morphological = MorphologicalTokenizer()

        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.token_scores: Dict[str, float] = {}

        self._init_special_tokens()

    def _init_special_tokens(self):
        for i, token in enumerate(self.config.special_tokens):
            self.token_to_id[token] = i
            self.id_to_token[i] = token
            self.token_scores[token] = 0.0

    def _viterbi_segment(self, word: str) -> List[str]:
        n = len(word)
        if n == 0:
            return []

        # best_score[i] = best score to reach position i
        # best_prev[i] = position from which we reached i with best score
        best_score = [float("-inf")] * (n + 1)
        best_prev = [-1] * (n + 1)
        best_score[0] = 0.0

        for i in range(n):
            if best_score[i] == float("-inf"):
                continue

            # Try all possible subwords starting at position i
            for j in range(i + 1, min(i + 20, n + 1)):  # max subword length 20
                subword = word[i:j]

                # Get score (use small penalty for unknown)
                if subword in self.token_scores:
                    score = self.token_scores[subword]
                else:
                    # Unknown token penalty
                    score = -10.0 - len(subword)

                total_score = best_score[i] + score
                if total_score > best_score[j]:
                    best_score[j] = total_score
                    best_prev[j] = i

        # Backtrack to find segmentation
        segments = []
        pos = n
        while pos > 0:
            prev = best_prev[pos]
            if prev == -1:
                # Fallback: single character
                segments.append(word[pos - 1 : pos])
                pos -= 1
            else:
                segments.append(word[prev:pos])
                pos = prev

        segments.reverse()
        return segments

    def train(self, texts: Iterator[str], progress_callback=None):
        substring_freqs: Counter = Counter()
        total_words = 0

        for text in texts:
            tokens = self.pre_tokenizer.pre_tokenize(text)
            for token, token_type in tokens:
                if token_type == "word":
                    total_words += 1
                    word = token.lower() if self.config.lowercase else token
                    n = len(word)
                    for i in range(n):
                        for j in range(i + 1, min(i + 21, n + 1)):
                            substring_freqs[word[i:j]] += 1

        # Initialize vocabulary with most frequent substrings
        initial_vocab_size = self.config.vocab_size * 2  # Start larger, prune down

        vocab_items = substring_freqs.most_common(initial_vocab_size)
        total_freq = sum(freq for _, freq in vocab_items)

        # Initialize scores as log probabilities
        for token, freq in vocab_items:
            self.token_scores[token] = math.log(freq / total_freq)

        # Add morphemes with boosted scores
        if self.config.use_morphology:
            morph = BelarusianMorphology()
            for prefix in morph.PREFIXES:
                if prefix not in self.token_scores:
                    self.token_scores[prefix] = math.log(0.001)
                else:
                    self.token_scores[prefix] += 1.0  # Boost

            for suffix in morph.SUFFIXES:
                if suffix not in self.token_scores:
                    self.token_scores[suffix] = math.log(0.001)
                else:
                    self.token_scores[suffix] += 1.0

            for ending in morph.ENDINGS:
                if ending not in self.token_scores:
                    self.token_scores[ending] = math.log(0.001)
                else:
                    self.token_scores[ending] += 1.0

        # Iteratively prune vocabulary
        target_size = self.config.vocab_size - len(self.config.special_tokens)

        while len(self.token_scores) > target_size:
            # Remove lowest-scoring tokens (that don't break segmentation too badly)
            sorted_tokens = sorted(
                [
                    (t, s)
                    for t, s in self.token_scores.items()
                    if t not in self.config.special_tokens and len(t) > 1
                ],
                key=lambda x: x[1],
            )

            # Remove bottom 10% of vocabulary
            remove_count = max(1, len(sorted_tokens) // 10)
            for token, _ in sorted_tokens[:remove_count]:
                del self.token_scores[token]

            if progress_callback:
                progress_callback(len(self.token_scores), target_size)

        # Build final vocabulary
        self._build_vocab()

    def _build_vocab(self):
        """Build vocabulary mappings."""
        idx = len(self.config.special_tokens)
        for token in sorted(self.token_scores.keys()):
            if token not in self.token_to_id:
                self.token_to_id[token] = idx
                self.id_to_token[idx] = token
                idx += 1

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using Unigram model."""
        result = []
        pre_tokens = self.pre_tokenizer.pre_tokenize(text)

        for token, token_type in pre_tokens:
            if token_type == "word":
                word = token.lower() if self.config.lowercase else token
                segments = self._viterbi_segment(word)
                result.extend(segments)
            elif token_type == "latin":
                segments = self._viterbi_segment(token)
                result.extend(segments)
            elif token_type == "number":
                result.append(token)
            elif token_type == "space":
                if token == "\n":
                    result.append("<nl>")
            elif token_type == "punct":
                result.append(token)
            else:
                result.append(token)

        return result

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to IDs."""
        tokens = self.tokenize(text)
        ids = []

        if add_special_tokens:
            ids.append(self.token_to_id["<bos>"])

        unk_id = self.token_to_id["<unk>"]
        for token in tokens:
            ids.append(self.token_to_id.get(token, unk_id))

        if add_special_tokens:
            ids.append(self.token_to_id["<eos>"])

        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode IDs to text."""
        tokens = []
        special = set(self.config.special_tokens)

        for id_ in ids:
            token = self.id_to_token.get(id_, "<unk>")
            if skip_special_tokens and token in special:
                continue
            tokens.append(token)

        # Simple join (Unigram doesn't use continuation markers)
        return "".join(tokens).replace("<nl>", "\n")

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)


# =============================================================================
# Complete Tokenizer Interface
# =============================================================================


class BelarusianTokenizer:
    """
    High-level tokenizer interface combining all components.

    This is the main class to use for tokenizing Belarusian text.
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        algorithm: str = "bpe",  # 'bpe' or 'unigram'
        use_morphology: bool = True,
        lowercase: bool = False,
    ):
        """
        Initialize Belarusian tokenizer.

        Args:
            vocab_size: Target vocabulary size
            algorithm: 'bpe' or 'unigram'
            use_morphology: Use morphological awareness
            lowercase: Convert text to lowercase
        """
        self.config = BPEConfig(
            vocab_size=vocab_size,
            use_morphology=use_morphology,
            lowercase=lowercase,
        )

        if algorithm == "bpe":
            self._tokenizer = BelarusianBPETokenizer(self.config)
        elif algorithm == "unigram":
            self._tokenizer = BelarusianUnigramTokenizer(self.config)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        self.algorithm = algorithm
        self.pre_tokenizer = BelarusianPreTokenizer(lowercase=lowercase)
        self.morphological = MorphologicalTokenizer()

    def train(self, texts: Iterator[str], progress_callback=None):
        """Train tokenizer on corpus."""
        self._tokenizer.train(texts, progress_callback)

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into subword tokens."""
        return self._tokenizer.tokenize(text)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs."""
        return self._tokenizer.encode(text, add_special_tokens)

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        return self._tokenizer.decode(ids, skip_special_tokens)

    def segment_morphemes(self, word: str) -> Dict[str, str]:
        """Segment word into morphemes (prefix, root, suffix, ending)."""
        return self.morphological.segment(word)

    @property
    def vocab_size(self) -> int:
        """Current vocabulary size."""
        return self._tokenizer.vocab_size

    @property
    def vocab(self) -> Dict[str, int]:
        """Vocabulary mapping."""
        return self._tokenizer.token_to_id.copy()

    def save(self, path: str):
        """Save tokenizer to directory."""
        if hasattr(self._tokenizer, "save"):
            self._tokenizer.save(path)
        else:
            # Save basic info
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)

            with open(path / "config.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "algorithm": self.algorithm,
                        "vocab_size": self.config.vocab_size,
                        "use_morphology": self.config.use_morphology,
                        "lowercase": self.config.lowercase,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

    @classmethod
    def load(cls, path: str) -> "BelarusianTokenizer":
        """Load tokenizer from directory."""
        path = Path(path)

        with open(path / "config.json", "r", encoding="utf-8") as f:
            config = json.load(f)

        tokenizer = cls(
            vocab_size=config.get("vocab_size", 32000),
            algorithm=config.get("algorithm", "bpe"),
            use_morphology=config.get("use_morphology", True),
            lowercase=config.get("lowercase", False),
        )

        if config.get("algorithm") == "bpe":
            tokenizer._tokenizer = BelarusianBPETokenizer.load(path)

        return tokenizer


def main():
    print("=" * 60)
    print("Belarusian Tokenizer Example")
    print("=" * 60)

    sample_texts = [
        "Беларуская мова — адна з усходнеславянскіх моў.",
        "Перакладчык працаваў над кнігай вельмі доўга.",
        "Настаўніца растлумачыла вучням новую тэму.",
        "Гэта найпрыгажэйшая краіна ў свеце!",
        "Дзяўчынка ўзяла кнігу і пачала чытаць.",
    ]

    print("\n1. Pre-tokenization:")
    print("-" * 40)
    pre_tok = BelarusianPreTokenizer()
    for text in sample_texts[:2]:
        print(f"Input: {text}")
        tokens = pre_tok.pre_tokenize(text)
        print(f"Tokens: {tokens}")
        print()

    print("\n2. Morphological Segmentation:")
    print("-" * 40)
    morph = MorphologicalTokenizer()
    test_words = [
        "перакладчык",  # пера + клад + чык
        "настаўніца",  # на + стаў + ніц + а
        "прыгажэйшая",  # пры + гаж + эйш + ая
        "растлумачыла",  # раз + тлумач + ыл + а
        "падручнік",  # пад + руч + нік
        "беларускай",  # белар + уск + ай
    ]

    for word in test_words:
        segments = morph.segment(word)
        print(f"{word}:")
        print(f"  prefix: '{segments['prefix']}'")
        print(f"  root:   '{segments['root']}'")
        print(f"  suffix: '{segments['suffix']}'")
        print(f"  ending: '{segments['ending']}'")
        print()

    print("\n3. Tokenizer Structure (untrained):")
    print("-" * 40)
    tokenizer = BelarusianTokenizer(vocab_size=8000, algorithm="bpe", use_morphology=True)

    print(f"Algorithm: {tokenizer.algorithm}")
    print(f"Target vocab size: {tokenizer.config.vocab_size}")
    print(f"Use morphology: {tokenizer.config.use_morphology}")
    print(f"Special tokens: {tokenizer.config.special_tokens}")

    print("\n4. Training on sample corpus:")
    print("-" * 40)

    training_corpus = (
        sample_texts * 10
        + [
            "Мінск — сталіца Беларусі.",
            "Беларусь — краіна ў цэнтры Еўропы.",
            "Беларуская літаратура мае багатую гісторыю.",
            "Янка Купала — вялікі беларускі паэт.",
            "Мова — душа народа.",
        ]
        * 5
    )

    def progress(current, target):
        if current % 100 == 0:
            print(f"  Vocab size: {current}/{target}")

    tokenizer.train(iter(training_corpus), progress_callback=progress)

    print(f"\nFinal vocab size: {tokenizer.vocab_size}")

    print("\n5. Tokenization Examples:")
    print("-" * 40)
    for text in sample_texts:
        print(f"Input: {text}")
        tokens = tokenizer.tokenize(text)
        print(f"Tokens: {tokens}")
        ids = tokenizer.encode(text)
        print(f"IDs: {ids[:10]}..." if len(ids) > 10 else f"IDs: {ids}")
        decoded = tokenizer.decode(ids)
        print(f"Decoded: {decoded}")
        print()


if __name__ == "__main__":
    main()
