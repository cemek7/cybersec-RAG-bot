# === app/llm_client/colab_llm.py ===

import requests
import os

COLAB_LLM_URL = os.getenv("COLAB_LLM_URL", "http://<your-colab-ngrok-url>/generate")

def call_colab_llm(prompt: str) -> str:
    try:
        resp = requests.post(COLAB_LLM_URL, json={"prompt": prompt}, timeout=60)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        return f"[Error calling Colab LLM] {e}"
