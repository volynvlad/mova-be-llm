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


@dataclass
class MergeConfig:
    max_new_tokens: int = 8000
    min_frequency: int = 10
    prioritize_morphemes: bool = True
    skip_existing: bool = True
    belarusian_token_prefix: str = ""
    reserved_ids: set[int] | None = None
