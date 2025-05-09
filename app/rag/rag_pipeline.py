from app.db.vector_store import get_vector_store
from app.llm_client.colab_llm import call_colab_llm
from app.llm_client.tiny_llm import clarify_query, generate_followups,tiny_llm
from app.prompts.prompt_engine import PromptEngine
from app.utils.qa_history import log_qa_interaction

class RAGPipeline:
    def __init__(self, top_k: int = 5):
        self.vs = get_vector_store()
        self.top_k = top_k
        self.prompt_engine = PromptEngine(mini_llm=tiny_llm)

    def generate_answer(self, raw_query: str, output_format: str = "markdown", **prompt_config):
        # Step 1: Clarify query
        query = clarify_query(raw_query)

        # Step 2: Retrieve top-k documents
        docs = self.vs.similarity_search(query, k=self.top_k)
        if not docs:
            raise ValueError("No relevant documents found.")

        # Step 3: Prepare context
        context_chunks = [doc.page_content for doc in docs]

        # Step 4: Build the final prompt using the enhanced PromptEngine
        config = self.prompt_engine.build_config(output_format=output_format, **prompt_config)
        prompt = self.prompt_engine.generate_base_prompt(query, context_chunks, config=config)

        # Step 5: Query the LLM
        answer = call_colab_llm(prompt)

        # Step 6: Follow-up suggestions
        follow_ups = generate_followups(answer)
        if not follow_ups:
            follow_ups = self.prompt_engine.generate_follow_up_questions(answer)

        # Step 7: Log
        log_qa_interaction(
            query=raw_query,
            prompt=prompt,
            answer=answer,
            follow_ups=follow_ups,
            sources=[d.metadata.get("source") for d in docs]
        )

        return {
            "question": query,
            "prompt": prompt,
            "answer": answer,
            "follow_up_questions": follow_ups,
            "sources": [d.metadata.get("source") for d in docs]
        }
