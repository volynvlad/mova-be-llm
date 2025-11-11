import os
from bs4 import BeautifulSoup
from ebooklib import epub, ITEM_DOCUMENT
from PyPDF2 import PdfReader
from rich.progress import track

from logger import get_logger

logger = get_logger(__name__)

INPUT_DIR = "data/raw/"


class Parser:
    output_dir = "data/interim"

    def parse_book(self, path):
        if path.endswith(".pdf"):
            self._parse_pdf(path)
        elif path.endswith(".epub"):
            self._parse_epub(path)

    def _parse_pdf(self, path: str):
        filename = path.split("/")[-1].replace(".pdf", ".txt")
        output_file = f"{self.output_dir}/{filename}"
        f = open(output_file, "w")
        try:
            reader = PdfReader(path)
            for page in track(reader.pages, description=path):
                f.write(page.extract_text())
            logger.debug(f"Finish writing to {output_file}")
        except Exception as e:
            logger.error(e)
            logger.error(f"Failed to parse {path}")
        finally:
            f.close()


    def _parse_epub(self, path: str):
        filename = path.split("/")[-1].replace(".epub", ".txt")
        output_file = f"{self.output_dir}/{filename}"
        f = open(output_file, "w")
        try:
            book = epub.read_epub(path)
            for item in track(book.get_items(), description=path):
                if item.get_type() == ITEM_DOCUMENT:
                    soup = BeautifulSoup(item.get_content(), 'html.parser')
                    f.write(soup.get_text())
            logger.debug(f"Finish writing to {output_file}")
        except epub.EpubException as e:
            pass
        except Exception as e:
            logger.error(e)
            logger.error(f"Failed to parse {path}")
        finally:
            f.close()



def iterate_books(input_path: str = INPUT_DIR):
    for root, _, files in os.walk(input_path):
        for file in files:
            if file:
                yield os.path.join(root, file)


def main():
    parser = Parser()
    for file_path in track(iterate_books(), description="Parse books"):
        parser.parse_book(file_path)


if __name__ == "__main__":
    main()
