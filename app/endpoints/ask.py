# app/api/endpoints/ask.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from app.rag.rag_pipeline import RAGPipeline
from app.utils.qa_history import load_history

router = APIRouter()
pipeline = RAGPipeline()

class AskRequest(BaseModel):
    query: str
    output_format: str = "markdown"  # Optional field with default
    intent: str = "inform"
    audience: str = "security engineer"
    style: str = "clear"
    doc_type: str = "technical"

class AskResponse(BaseModel):
    answer: str
    final_prompt: str
    sources: List[str]
    follow_ups: List[str]

@router.post("/ask", response_model=AskResponse)
async def ask_question(data: AskRequest):
    try:
        result = pipeline.generate_answer(data.query, output_format=data.output_format)
        return AskResponse(
            answer=result["answer"],
            final_prompt=result["prompt"],
            sources=result["sources"],
            follow_ups=result["follow_up_questions"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history")
def get_history():
    
    return load_history()
