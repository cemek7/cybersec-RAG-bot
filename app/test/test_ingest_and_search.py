# === File: test_ingest_and_search.py ===
from app.ingestion.ingestor import ingest_file, ingest_url
from app.db.vector_store import get_vector_store, add_to_vector_store, search_vector_store
import sys

# Example usage:
def run():
    vector_store = get_vector_store()

    # Example: Ingest a local Markdown file
    md_path = "data/Xss.md"  # Replace with actual file path
    try:
        doc = ingest_file(md_path)
        add_to_vector_store([doc], vector_store)
        print(f"✅ Ingested and added: {md_path}")
    except Exception as e:
        print(f"Error with Markdown file: {e}")

    # Example: Ingest a URL
    url = "https://www.acunetix.com/blog/articles/cross-site-scripting/"
    try:
        doc_url = ingest_url(url)
        add_to_vector_store([doc_url], vector_store)
        print(f"✅ Ingested and added: {url}")
    except Exception as e:
        print(f"Error with URL: {e}")

    # Perform a semantic search
    query = "how to exploit xss"
    results = search_vector_store(query, vector_store)

    print(f"\n🔍 Top results for: '{query}'")
    for idx, res in enumerate(results):
        print(f"\nResult {idx+1}:")
        print(f"Source: {res.metadata.get('source')}")
        print(res.page_content[:500])

if __name__ == "__main__":
    run()
