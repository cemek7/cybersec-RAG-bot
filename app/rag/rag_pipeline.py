# === app/rag/rag_pipeline.py ===

from app.db.vector_store import get_vector_store
from app.prompts.prompt_formatter import format_prompt
from app.llm_client.colab_llm import call_colab_llm
from app.llm_client.tiny_llm import clarify_query, generate_followups

class RAGPipeline:
    def __init__(self, top_k: int = 5):
        self.vs = get_vector_store()
        self.top_k = top_k

    def generate_answer(self, raw_query: str, role: str = None, output_format: str = "markdown"):
        # 1. Clarify via mini-LLM
        query = clarify_query(raw_query)

        # 2. Retrieve top-k docs
        docs = self.vs.similarity_search(query, k=self.top_k)

        # 3. Build prompt
        prompt = format_prompt(query, docs, output_format=output_format)

        # 4. Call Colab-hosted LLM
        answer = call_colab_llm(prompt)

        # 5. Generate follow-up questions
        follow_ups = generate_followups(answer)

        # 6. Return structured result
        return {
            "question": query,
            "answer": answer,
            "follow_up_questions": follow_ups,
            "sources": [d.metadata for d in docs]
        }
