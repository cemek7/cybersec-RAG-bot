# === app/server/main.py ===

import os
import shutil
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from app.rag.rag_pipeline import RAGPipeline
from app.ingestion import ingestor
from app.db.vector_store import get_vector_store
from app.api.endpoints.ask import router as ask_router  # <-- Import the modular ask router
from langchain_core.documents import Document

app = FastAPI()

# Mount the /ask endpoint from the ask router
app.include_router(ask_router, prefix="")  # No prefix since ask.py already defines /ask

# Initialize components
pipeline = RAGPipeline()
vs = get_vector_store()

# API for ingesting files into the vector store
@app.post("/ingest")
async def ingest_file_api(file: UploadFile = File(...)):
    filepath = os.path.join("data", file.filename)
    with open(filepath, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    # Ingest and add document to vector store
    data = ingestor.ingest_file(filepath)
    doc = Document(
        page_content=data["content"],
        metadata={"source": data["source"], "type": data["doc_type"]}
    )
    vs.add_documents([doc])

    return {"status": "success", "file": file.filename}
