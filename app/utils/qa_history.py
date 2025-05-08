# === File: app/utils/qa_history.py ===

import json
from datetime import datetime
from pathlib import Path

HISTORY_FILE = Path("qa_history.json")

def load_history():
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_entry(entry: dict):
    history = load_history()
    history.append(entry)

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

def log_qa_interaction(query: str, prompt: str, answer: str, follow_ups: list, sources: list):
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "query": query,
        "final_prompt": prompt,
        "answer": answer,
        "follow_ups": follow_ups,
        "sources": sources
    }
    save_entry(entry)
