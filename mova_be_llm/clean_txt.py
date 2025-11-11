import re

def clean_text(text):
    pattern = r'[^а-яА-ЯёЁa-zA-Z\-\s,\.!?;:"]'
    return re.sub(pattern, '', text)

def clean_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    cleaned_text = clean_text(text)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(cleaned_text)

clean_file("data/interim/belarusian.txt", "data/interim/belarusian_cleaned.txt")
