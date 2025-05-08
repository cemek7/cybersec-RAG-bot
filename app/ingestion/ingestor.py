# === File: app/ingestion/ingestor.py ===
import os
import fitz  # PyMuPDF
import docx
import markdown
import requests
from bs4 import BeautifulSoup


def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    return "\n".join([page.get_text() for page in doc])

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_md(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return markdown.markdown(f.read())

def extract_text_from_url(url):
    res = requests.get(url)
    soup = BeautifulSoup(res.text, 'html.parser')
    return soup.get_text()

def ingest_file(file_path):
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == '.pdf':
        content = extract_text_from_pdf(file_path)
        doc_type = 'pdf'
    elif ext == '.docx':
        content = extract_text_from_docx(file_path)
        doc_type = 'docx'
    elif ext == '.md':
        content = extract_text_from_md(file_path)
        doc_type = 'markdown'
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    return {
        'content': content,
        'source': file_path,
        'doc_type': doc_type
    }

def ingest_url(url):
    content = extract_text_from_url(url)
    return {
        'content': content,
        'source': url,
        'doc_type': 'html'
    }
