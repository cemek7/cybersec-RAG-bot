# === app/llm_client/tiny_llm.py ===

from llama_cpp import Llama
import os

MODEL_PATH = os.getenv("TINYLLM_PATH", "models/tinyllama.gguf")
llm = Llama(model_path=MODEL_PATH, n_ctx=1024, n_threads=4)

def clarify_query(raw_query: str) -> str:
    prompt = (
        "You are a prompt refiner for a cybersecurity assistant. "
        "Rewrite the following user question to improve clarity and detail.\n\n"
        f"User Question: {raw_query}\n\nRefined Question:"
    )
    res = llm(prompt, max_tokens=100, stop=["\n"])
    return res["choices"][0]["text"].strip()

def generate_followups(answer: str) -> list[str]:
    prompt = (
        "Based on this answer from a cybersecurity AI, suggest 2 follow-up questions "
        "that a security practitioner might ask next.\n\n"
        f"Answer: {answer}\n\nFollow-up Questions:"
    )
    res = llm(prompt, max_tokens=100, stop=["\n"])
    lines = res["choices"][0]["text"].split("\n")
    return [line.strip("- ").strip() for line in lines if line.strip()]
