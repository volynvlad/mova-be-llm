import unicodedata
import re

CONTROL = dict.fromkeys(range(32))
CONTROL.pop(9)
CONTROL.pop(10)
CONTROL.pop(13)


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(CONTROL)
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")

    text = re.sub(r'[^\w\s\'".,!?;:()-]', " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_file(inp: str, out: str) -> None:
    with open(inp, encoding="utf-8") as f:
        text = f.read()
    with open(out, "w", encoding="utf-8") as f:
        f.write(clean_text(text))


if __name__ == "__main__":
    clean_file("data/interim/belarusian.txt", "data/processed/belarusian_cleaned_v2.txt")
