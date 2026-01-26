from dataclasses import dataclass
from enum import StrEnum


class TokenType(StrEnum):
    word = "word"
    latin = "latin"
    number = "number"
    space = "space"
    punct = "punct"
    url = "url"
    email = "email"


DEFAULT_MAX_NEW_TOKENS = 8000
DEFAULT_MIN_FREQUENCY = 5
DEFAULT_VOCAB_SIZE = 32000

@dataclass
class MergeConfig:
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    min_frequency: int = DEFAULT_MIN_FREQUENCY
    prioritize_morphemes: bool = True
    skip_existing: bool = True
    belarusian_token_prefix: str = ""
    reserved_ids: set[int] | None = None
