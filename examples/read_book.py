from ebooklib import epub
from PyPDF2 import PdfReader
import djvu

def read_pdf(path: str):
    reader = PdfReader(path)
    for page in reader.pages:
        print(page.extract_text())

def read_epub(path: str):
    book = epub.read_epub(path)
    for item in book.get_items():
        if item.get_type() == epub.ITEM_DOCUMENT:
            print(item.get_content().decode('utf-8'))

def read_djvu(path: str):
    context = djvu.Context()
    document = context.new_document(djvu.decode_file(path))
    text = document.get_page_text(0)  # Page number
    print(text)


def main():
    read_pdf("../data/raw/knihi_books/Latina-BY.pdf")
    read_epub("../data/raw/knihi_books/Mova.epub")
    read_djvu("../data/raw/knihi_books/Katechizis.djvu")


if __name__ == "__main__":
    main()
