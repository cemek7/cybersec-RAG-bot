# === app/utils/batch_ingest.py ===

import os
import argparse
from tqdm import tqdm
from app.ingestion import ingestor
from app.db.vector_store import get_vector_store
from langchain_core.documents import Document

SUPPORTED_EXT = {'.pdf', '.docx', '.md', '.txt'}

def ingest_all_files(dir_path: str, vs):
    files = []
    for root, _, fnames in os.walk(dir_path):
        for f in fnames:
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXT:
                files.append(os.path.join(root, f))
    for fp in tqdm(files, desc="Ingesting files"):
        try:
            data = ingestor.ingest_file(fp)
            doc = Document(page_content=data["content"],
                           metadata={"source": data["source"], "type": data["doc_type"]})
            vs.add_documents([doc])
        except Exception as e:
            print(f"Error ingesting {fp}: {e}")

def ingest_urls(url_file: str, vs):
    with open(url_file) as f:
        urls = [l.strip() for l in f if l.strip()]
    for url in tqdm(urls, desc="Ingesting URLs"):
        try:
            data = ingestor.ingest_url(url)
            doc = Document(page_content=data["content"],
                           metadata={"source": data["source"], "type": data["doc_type"]})
            vs.add_documents([doc])
        except Exception as e:
            print(f"Error ingesting URL {url}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", help="Directory of docs", default="data")
    parser.add_argument("--urls", help="File of URLs", default=None)
    args = parser.parse_args()

    vs = get_vector_store()
    if args.dir:
        ingest_all_files(args.dir, vs)
    if args.urls:
        ingest_urls(args.urls, vs)
