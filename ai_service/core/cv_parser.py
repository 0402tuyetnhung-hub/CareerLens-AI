import re
import fitz  # pymupdf

def parse_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def parse_pdf(file_path):
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def parse_cv(file_path):
    if file_path.lower().endswith('.pdf'):
        return parse_pdf(file_path)
    elif file_path.lower().endswith('.txt'):
        return parse_txt(file_path)
    else:
        raise ValueError("Unsupported file format. Use PDF or TXT.")

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
