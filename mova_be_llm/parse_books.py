import sys
"""
Book Parser - Extract text from EPUB, PDF, and DJVU files.

Required dependencies:
    pip install ebooklib beautifulsoup4 PyMuPDF python-djvulibre

For DJVU support, you may also need system package:
    sudo apt install djvulibre-bin  # Ubuntu/Debian
"""

import re
import unicodedata
from pathlib import Path
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import os
from rich.progress import track
import fitz  # PyMuPDF
import djvu.decode

from logger import get_logger

logger = get_logger(__name__)
INPUT_DIR = "data/raw/"



class BookParser:
    """Parse EPUB, PDF, and DJVU books to plain text."""

    SUPPORTED_FORMATS = {'.epub', '.pdf', '.djvu'}
    OUTPUT_DIR = "data/interim"

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        self.extension = self.file_path.suffix.lower()

    def parse(self) -> str:
        """Parse the book and return cleaned text."""
        if self.extension == '.epub':
            text = self._parse_epub()
        elif self.extension == '.pdf':
            text = self._parse_pdf()
        elif self.extension == '.djvu':
            text = self._parse_djvu()
        else:
            return ""

        return self._clean_text(text)

    def save(self) -> str:
        filename = str(self.file_path).split("/")[-1].removesuffix(".pdf").removesuffix(".epub").removesuffix(".djvu")
        output_path = Path(f"{self.OUTPUT_DIR}/{filename}").with_suffix('.txt')
        try:
            if not (text:=self.parse()):
                return ""
        except Exception:
            logger.error(f"Error for {self.file_path}")
            return ""
        output_path.write_text(text, encoding='utf-8')
        return str(output_path)

    def _parse_epub(self) -> str:
        """Extract text from EPUB file."""
        book = epub.read_epub(str(self.file_path), options={'ignore_ncx': True})
        text_parts = []

        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                content = item.get_content()
                soup = BeautifulSoup(content, 'html.parser')

                for tag in soup(['script', 'style', 'meta', 'link', 'svg', 'img', 'figure']):
                    tag.decompose()

                text = soup.get_text(separator='\n')
                if text.strip():
                    text_parts.append(text)

        return '\n\n'.join(text_parts)

    def _parse_pdf(self) -> str:
        """Extract text from PDF file."""
        text_parts = []

        with fitz.open(str(self.file_path)) as doc:
            for page in doc:
                blocks = page.get_text("blocks")

                for block in blocks:
                    # block[6] == 0 means it's a text block (not image)
                    if block[6] == 0:
                        text = block[4]
                        if text.strip():
                            text_parts.append(text)

        return '\n'.join(text_parts)

    def _parse_djvu(self) -> str:
        """Extract text from DJVU file."""
        return self._parse_djvu_with_library()

    def _parse_djvu_with_library(self) -> str:
        """Parse DJVU using python-djvulibre."""

        context = djvu.decode.Context()
        document = context.new_document(djvu.decode.FileURI(str(self.file_path)))
        document.decoding_job.wait()

        text_parts = []
        for _, page in enumerate(document.pages):
            try:
                page.decode(wait=True)
                page_text = page.text.sexpr
                if page_text:
                    extracted = self._extract_djvu_text(page_text)
                    if extracted.strip():
                        text_parts.append(extracted)
            except (djvu.decode.JobFailed, AttributeError, Exception):
                # Page may not have text layer or failed to decode
                continue

        return '\n\n'.join(text_parts)

    def _extract_djvu_text(self, sexpr) -> str:
        """Recursively extract text from DJVU S-expression."""
        if isinstance(sexpr, str):
            return sexpr
        if isinstance(sexpr, (list, tuple)):
            texts = []
            for item in sexpr:
                if isinstance(item, (str, list, tuple)) and not isinstance(item, int):
                    extracted = self._extract_djvu_text(item)
                    if extracted:
                        texts.append(extracted)
            return ' '.join(texts)
        return ''

    def _clean_text(self, text: str) -> str:
        """Clean extracted text from artifacts and encoding issues."""
        if not text:
            return ''

        text = unicodedata.normalize('NFKC', text)
        text = ''.join(
            char for char in text
            if unicodedata.category(char) != 'Cc' or char in '\n\t'
        )

        text = text.replace('\x00', '')

        mojibake_patterns = [
            (r'Ã¢â‚¬â„¢', "'"),
            (r'Ã¢â‚¬Å"', '"'),
            (r'Ã¢â‚¬ï¿½', '"'),
            (r'Ã¢â‚¬â€œ', '—'),
            (r'Ã¢â‚¬â€˜', '–'),
            (r'Ã‚Â', ''),
            (r'â€™', "'"),
            (r'â€œ', '"'),
            (r'â€', '"'),
            (r'â€"', '—'),
            (r'â€"', '–'),
        ]

        for pattern, replacement in mojibake_patterns:
            text = text.replace(pattern, replacement)

        text = re.sub(r'[\x80-\x9f]+', '', text)

        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            if line.strip():
                letters = sum(1 for c in line if c.isalpha())
                total = len(line.replace(' ', ''))
                if total == 0 or letters / total > 0.3 or len(line) < 10:
                    cleaned_lines.append(line)
            else:
                cleaned_lines.append(line)

        text = '\n'.join(cleaned_lines)

        text = re.sub(r'[^\S\n]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = '\n'.join(line.strip() for line in text.split('\n'))
        return text.strip()


def iterate_books(input_path: str = INPUT_DIR):
    for root, _, files in os.walk(input_path):
        for file in files:
            if file:
                yield os.path.join(root, file)


if __name__ == '__main__':
    for file_path in track(iterate_books(), description="Parse books"):
        parser = BookParser(file_path)
        if (saved_path:=parser.save()):
            logger.debug(f"Text extracted to: {saved_path}")
