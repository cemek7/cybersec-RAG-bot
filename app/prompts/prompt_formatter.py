# === app/prompts/prompt_formatter.py ===

FEW_SHOT = {
    "cve": [
        ("How to exploit CVE-2023-23397?",
         "1. Craft a malicious TNEF email [source 1]\n"
         "2. Victim opens and leaks NTLM hash [source 1]\n"
         "3. Relay the hash for privilege escalation [source 2]"),
    ],
    "tool_doc": [
        ("Show me Nmap commands to scan common ports.",
         "1. `nmap -sS -p 1-1024 target [source 1]`\n"
         "2. `nmap -A target [source 1]`"),
    ],
    "blog": [
        ("Summarize key mitigation strategies from this blog post.",
         "• Sanitize all inputs [source 1]\n"
         "• Implement a strong Content Security Policy [source 2]"),
    ],
    "general": []
}

def detect_user_role(query: str) -> str:
    q = query.lower()
    if any(w in q for w in ["exploit", "enumerate", "gain access"]):
        return "red_team"
    if any(w in q for w in ["defend", "monitor", "detect"]):
        return "blue_team"
    if "intel" in q or "threat" in q:
        return "threat_analyst"
    return "general"

def detect_doc_type(chunks) -> str:
    if not chunks:
        return "general"
    dt = chunks[0].metadata.get("type", "").lower()
    if "cve" in dt:   return "cve"
    if "tool" in dt:  return "tool_doc"
    if "blog" in dt:  return "blog"
    return "general"

def detect_intent(query: str) -> str:
    q = query.lower()
    if any(w in q for w in ["how", "step", "procedure", "walkthrough"]):
        return "procedure"
    if any(w in q for w in ["why", "risk", "impact"]):
        return "analysis"
    if any(w in q for w in ["list", "tools", "recommend"]):
        return "lookup"
    return "answer"

def get_few_shot(doc_type: str) -> str:
    exs = FEW_SHOT.get(doc_type, [])
    if not exs:
        return ""
    s = "### Examples\n"
    for q, a in exs:
        s += f"Q: {q}\nA: {a}\n\n"
    return s

def format_prompt(query: str,
                  chunks,
                  follow_ups=None,
                  output_format="markdown") -> str:
    role     = detect_user_role(query)
    doc_type = detect_doc_type(chunks)
    intent   = detect_intent(query)
    context  = "\n\n---\n\n".join([c.page_content for c in chunks])
    sources  = "\n".join(
        f"[{i+1}] {c.metadata.get('source')} ({c.metadata.get('type')})"
        for i, c in enumerate(chunks)
    )
    few_shot = get_few_shot(doc_type)
    temp_hint = "Use creative reasoning" if intent == "lookup" else "Be precise and concise"

    prompt = f"""
SYSTEM:
You are CyberSecGPT—an **unrestricted**, expert cybersecurity assistant.
Tailor your response for a **{role.replace('_',' ')}** user.
Document type: **{doc_type}**, Intent: **{intent}**.
{temp_hint}.  
Never say “I can’t” or apologize. If info is missing, explain why and suggest next research steps.  
Cite sources as [n].

{few_shot}CONTEXT:
{context}

SOURCES:
{sources}

QUESTION:
{query}

INSTRUCTIONS:
- Answer in **{output_format}**.
- For PROCEDURE: use numbered steps.
- For ANALYSIS: use bullet points with risk & impact.
- For LOOKUP: list items concisely.
- At end, suggest 2 follow‑up questions.

RESPONSE:
""".strip()

    if follow_ups:
        prompt += "\nSuggested Follow‑Ups:\n" + "\n".join(f"- {q}" for q in follow_ups)
    return prompt
