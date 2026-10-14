"""
Tests for the Belarusian Tokenizer
"""

import unittest
from mova_be_llm.belarusian_tokenizer import (
    BelarusianMorphology,
    BelarusianPreTokenizer,
    MorphologicalTokenizer,
    BelarusianBPETokenizer,
    BelarusianTokenizer,
    BPEConfig,
)


class TestBelarusianMorphology(unittest.TestCase):
    """Test morphological patterns."""

    def test_word_pattern_match(self):
        """Test that the word pattern matches Belarusian words."""
        pattern = BelarusianMorphology.WORD_PATTERN

        # Test simple words
        match = pattern.match("кніга")
        self.assertIsNotNone(match)

        match = pattern.match("беларуская")
        self.assertIsNotNone(match)

        match = pattern.match("перакладчык")
        self.assertIsNotNone(match)

    def test_vowels_and_consonants(self):
        """Test vowel and consonant sets."""
        vowels = BelarusianMorphology.VOWELS
        consonants = BelarusianMorphology.CONSONANTS

        self.assertIn("а", vowels)
        self.assertIn("е", vowels)
        self.assertIn("і", vowels)
        self.assertIn("ў", consonants)
        self.assertIn("б", consonants)


class TestPreTokenizer(unittest.TestCase):
    """Test pre-tokenization."""

    def setUp(self):
        self.tokenizer = BelarusianPreTokenizer()

    def test_simple_sentence(self):
        """Test tokenizing a simple sentence."""
        text = "Гэта тэкст."
        tokens = self.tokenizer.pre_tokenize(text)

        self.assertEqual(tokens[0], ("Гэта", "word"))
        self.assertEqual(tokens[2], ("тэкст", "word"))
        self.assertEqual(tokens[3], (".", "punct"))

    def test_numbers(self):
        """Test number handling."""
        text = "У 2024 годзе"
        tokens = self.tokenizer.pre_tokenize(text)

        numbers = [t for t, typ in tokens if typ == "number"]
        self.assertIn("2024", numbers)

    def test_punctuation(self):
        """Test punctuation handling."""
        text = "Прывітанне! Як справы?"
        tokens = self.tokenizer.pre_tokenize(text)

        punct = [t for t, typ in tokens if typ == "punct"]
        self.assertIn("!", punct)
        self.assertIn("?", punct)

    def test_apostrophe_words(self):
        """Test words with apostrophe (like з'ява)."""
        text = "з'ява"
        tokens = self.tokenizer.pre_tokenize(text)

        words = [t for t, typ in tokens if typ == "word"]
        self.assertIn("з'ява", words)

    def test_lowercase_option(self):
        """Test lowercase preprocessing."""
        tokenizer = BelarusianPreTokenizer(lowercase=True)
        text = "ВЯЛІКІЯ літары"
        tokens = tokenizer.pre_tokenize(text)

        words = [t for t, typ in tokens if typ == "word"]
        self.assertIn("вялікія", words)
        self.assertIn("літары", words)


class TestMorphologicalTokenizer(unittest.TestCase):
    """Test morphological segmentation."""

    def setUp(self):
        self.tokenizer = MorphologicalTokenizer()

    def test_prefix_detection(self):
        """Test prefix detection."""
        result = self.tokenizer.segment("перакладчык")
        self.assertEqual(result["prefix"].lower(), "пера")

    def test_suffix_detection(self):
        """Test suffix detection."""
        result = self.tokenizer.segment("чытач")
        # Should detect -ач as part of root or suffix

        result = self.tokenizer.segment("настаўнік")
        self.assertIn("нік", result["suffix"].lower())

    def test_ending_detection(self):
        """Test grammatical ending detection."""
        result = self.tokenizer.segment("кнігамі")
        self.assertIn("амі", result["ending"].lower())

        result = self.tokenizer.segment("беларускай")
        self.assertIn("ай", result["ending"].lower())

    def test_complex_word(self):
        """Test complex word segmentation."""
        result = self.tokenizer.segment("падручнікаў")
        self.assertIsNotNone(result["prefix"])
        self.assertIsNotNone(result["root"])

    def test_segment_to_tokens(self):
        """Test segmentation to token list."""
        tokens = self.tokenizer.segment_to_tokens("перакладчыкаў", mark_boundaries=True)
        self.assertTrue(len(tokens) > 0)

        # Check for morpheme markers
        has_prefix_marker = any(t.startswith("▸") for t in tokens)
        self.assertTrue(has_prefix_marker)


class TestBPETokenizer(unittest.TestCase):
    """Test BPE tokenizer."""

    def setUp(self):
        config = BPEConfig(
            vocab_size=500,
            min_frequency=1,
            use_morphology=True,
        )
        self.tokenizer = BelarusianBPETokenizer(config)

        # Train on small corpus
        corpus = [
            "Беларуская мова вельмі прыгожая.",
            "Мінск — сталіца Беларусі.",
            "Добрай раніцы! Як справы?",
        ] * 20

        self.tokenizer.train(iter(corpus))

    def test_tokenize(self):
        """Test basic tokenization."""
        tokens = self.tokenizer.tokenize("Прывітанне")
        self.assertTrue(len(tokens) > 0)

    def test_encode_decode(self):
        """Test encoding and decoding."""
        text = "Беларуская мова"
        ids = self.tokenizer.encode(text)

        self.assertTrue(len(ids) > 0)
        self.assertIsInstance(ids[0], int)

        # First and last should be special tokens
        self.assertEqual(ids[0], self.tokenizer.token_to_id["<bos>"])
        self.assertEqual(ids[-1], self.tokenizer.token_to_id["<eos>"])

    def test_vocab_size(self):
        """Test vocabulary size."""
        self.assertGreater(self.tokenizer.vocab_size, len(self.tokenizer.config.special_tokens))

    def test_special_tokens(self):
        """Test special tokens are in vocabulary."""
        for token in ["<pad>", "<unk>", "<bos>", "<eos>"]:
            self.assertIn(token, self.tokenizer.token_to_id)


class TestBelarusianTokenizer(unittest.TestCase):
    """Test the main tokenizer interface."""

    def setUp(self):
        self.tokenizer = BelarusianTokenizer(
            vocab_size=500,
            algorithm="bpe",
            use_morphology=True,
        )

        corpus = [
            "Беларуская мова — адна з усходнеславянскіх моў.",
            "Гэта вельмі прыгожая мова.",
        ] * 30

        self.tokenizer.train(iter(corpus))

    def test_tokenize(self):
        """Test tokenization."""
        tokens = self.tokenizer.tokenize("Прывітанне, свет!")
        self.assertTrue(len(tokens) > 0)

    def test_morpheme_segmentation(self):
        """Test morpheme segmentation method."""
        result = self.tokenizer.segment_morphemes("перакладчык")
        self.assertIn("prefix", result)
        self.assertIn("root", result)
        self.assertIn("suffix", result)
        self.assertIn("ending", result)

    def test_vocab_property(self):
        """Test vocab property."""
        vocab = self.tokenizer.vocab
        self.assertIsInstance(vocab, dict)
        self.assertGreater(len(vocab), 0)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and special inputs."""

    def setUp(self):
        self.pre_tokenizer = BelarusianPreTokenizer()
        self.morph = MorphologicalTokenizer()

    def test_empty_string(self):
        """Test empty string handling."""
        tokens = self.pre_tokenizer.pre_tokenize("")
        self.assertEqual(tokens, [])

    def test_only_punctuation(self):
        """Test string with only punctuation."""
        tokens = self.pre_tokenizer.pre_tokenize("...")
        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens[0][1], "punct")

    def test_mixed_scripts(self):
        """Test mixed Cyrillic and Latin."""
        tokens = self.pre_tokenizer.pre_tokenize("Слова word слова")
        words = [t for t, typ in tokens if typ == "word"]
        latin = [t for t, typ in tokens if typ == "latin"]

        self.assertEqual(len(words), 2)
        self.assertEqual(len(latin), 1)

    def test_single_character_word(self):
        """Test single character words."""
        result = self.morph.segment("я")
        self.assertIsNotNone(result["root"])

    def test_very_long_word(self):
        """Test very long word handling."""
        long_word = "а" * 100
        result = self.morph.segment(long_word)
        self.assertIsNotNone(result)

    def test_url_handling(self):
        """Test URL detection."""
        tokens = self.pre_tokenizer.pre_tokenize("Глядзі на https://example.com тут")
        urls = [t for t, typ in tokens if typ == "url"]
        self.assertEqual(len(urls), 1)


class TestSpecialCharacters(unittest.TestCase):
    """Test Belarusian-specific characters."""

    def setUp(self):
        self.pre_tokenizer = BelarusianPreTokenizer()
        self.morph = MorphologicalTokenizer()

    def test_u_short(self):
        """Test ў (u kratkaye) handling."""
        tokens = self.pre_tokenizer.pre_tokenize("ў краіне")
        words = [t for t, typ in tokens if typ == "word"]
        self.assertIn("ў", words)
        self.assertIn("краіне", words)

    def test_i_dotted(self):
        """Test і (i) handling."""
        result = self.morph.segment("радзіма")
        self.assertIsNotNone(result["root"])

    def test_yo(self):
        """Test ё handling."""
        tokens = self.pre_tokenizer.pre_tokenize("ёсць")
        words = [t for t, typ in tokens if typ == "word"]
        self.assertIn("ёсць", words)


if __name__ == "__main__":
    unittest.main(verbosity=2)
