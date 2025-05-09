from typing import List, Optional

class PromptEngine:
    def __init__(self, mini_llm=None):
        self.mini_llm = mini_llm  # Optional tiny model to assist refinement

    def build_config(
        self,
        intent: str = "inform",
        audience: str = "cybersecurity analyst",
        format: str = "markdown",
        style: str = "clear",
        doc_type: str = "technical"
    ) -> dict:
        return {
            "intent": intent,
            "audience": audience,
            "format": format,
            "style": style,
            "doc_type": doc_type
        }

    def generate_base_prompt(
        self,
        query: str,
        context_chunks: List[str],
        config: Optional[dict] = None
    ) -> str:
        if not config:
            config = self.build_config()

        system_prompt = f"""
You are a helpful cybersecurity assistant.

- Intent: {config['intent']}
- Audience: {config['audience']}
- Response Format: {config['format']}
- Writing Style: {config['style']}
- Source Type: {config['doc_type']}

Use the given context to answer the user's question accurately.
If the context is lacking, use your best reasoning without fabricating information.
        """.strip()

        context_text = "\n\n---\n\n".join(context_chunks[:5])
        user_prompt = f"{query}\n\nContext:\n{context_text}"

        final_prompt = f"{system_prompt}\n\n{user_prompt}"

        if self.mini_llm:
            return self.mini_llm.refine_prompt(system_prompt, user_prompt)

        return final_prompt

    def generate_follow_up_questions(self, answer: str) -> List[str]:
        if self.mini_llm:
            return self.mini_llm.suggest_follow_ups(answer)
        return []

    def refine_prompt_with_context(
        self, raw_prompt: str, context_chunks: List[str]
    ) -> str:
        context_text = "\n\n".join(context_chunks[:3])
        if self.mini_llm:
            return self.mini_llm.enhance_prompt_with_context(raw_prompt, context_text)
        return f"{raw_prompt}\n\nContext:\n{context_text}"
