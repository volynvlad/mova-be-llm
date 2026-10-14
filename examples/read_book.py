from bs4 import BeautifulSoup
from ebooklib import epub, ITEM_DOCUMENT
from PyPDF2 import PdfReader


def read_pdf(path: str):
    print("\t\t=== Reading pdf book ===")
    reader = PdfReader(path)
    for i, page in enumerate(reader.pages):
        print(page.extract_text())
        if i > 5:
            break


def read_epub(file_path):
    print("\t\t=== Reading epub book ===")
    book = epub.read_epub(file_path)
    for item in book.get_items():
        if item.get_type() == ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            text = soup.get_text()
            print(text)


def read_book(path: str):
    if path.endswith(".pdf"):
        read_pdf(path)
    elif path.endswith(".epub"):
        read_epub(path)
    else:
        raise ValueError(f"Doesn't support given format: {path}")


def main():
    read_book("data/raw/knihi_books/Latina-BY.pdf")
    read_book("data/raw/knihi_books/Mova.epub")


if __name__ == "__main__":
    main()
