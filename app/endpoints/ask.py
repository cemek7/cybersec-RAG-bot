# === app/api/endpoints/ask.py ===

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import List
from app.prompts.prompt_engine import PromptEngine
from app.vectorstore.search import fetch_relevant_chunks
from app.llm_client.colab_llm import query_llm
from app.utils.qa_history import log_qa_interaction

router = APIRouter()
prompt_engine = PromptEngine()

class AskRequest(BaseModel):
    query: str
    config: dict = None  # Example: intent, audience, format, style, doc_type

class AskResponse(BaseModel):
    answer: str
    final_prompt: str
    sources: List[str]
    follow_ups: List[str]

@router.post("/ask", response_model=AskResponse)
async def ask_question(data: AskRequest, request: Request):
    try:
        # Fetch relevant chunks from the vectorstore
        chunks = fetch_relevant_chunks(data.query)
        if not chunks:
            raise HTTPException(status_code=404, detail="No relevant context found.")

        context_chunks = [c.page_content for c in chunks]
        sources = [
            f"{c.metadata.get('source')} ({c.metadata.get('type')})"
            for c in chunks
        ]

        # Generate a structured prompt using PromptEngine
        prompt = prompt_engine.generate_base_prompt(
            query=data.query,
            context_chunks=context_chunks,
            config=data.config
        )

        # Query the LLM with the generated prompt
        answer = query_llm(prompt)

        # Generate possible follow-up questions
        follow_ups = prompt_engine.generate_follow_up_questions(answer)

        # Log the Q&A interaction into qa_history.json
        log_qa_interaction(
            query=data.query,
            prompt=prompt,
            answer=answer,
            follow_ups=follow_ups,
            sources=sources
        )

        return AskResponse(
            answer=answer,
            final_prompt=prompt,
            sources=sources,
            follow_ups=follow_ups
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
