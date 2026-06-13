"""
rag_engine.py
─────────────────────────────────────────────────────────────────
Two main functions:
  1. analyse_document() — full-document structured analysis via Ollama
  2. ask_question()     — RAG Q&A: retrieve relevant chunks → Ollama answer
"""

import re
import json
from typing import List, Dict
import requests
from dotenv import load_dotenv
from vector_store import retrieve
from extractor import extract_full_text, get_page_count
from vector_store import store_chunks

load_dotenv()

MODEL        = "llama3.2"
OLLAMA_URL   = "http://localhost:11434/api/generate"


# ── Helpers ────────────────────────────────────────────────────────────────────
def _truncate(text: str, max_chars: int = 12000) -> str:
    """Truncate text to fit within the model's context safely."""
    return text[:max_chars] + "\n\n[... truncated for length]" if len(text) > max_chars else text


def _ollama_generate(prompt: str, max_tokens: int = 2048) -> str:
    """Send a prompt to Ollama and return the response text."""
    payload = {
        "model":  MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens,   # ← was 1024, now 2048
            "temperature": 0.1,
        },
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=180)  # longer timeout too
    response.raise_for_status()
    return response.json().get("response", "").strip()


def _repair_truncated_json(text: str) -> dict:
    """
    Last-resort repair for JSON that was cut off mid-stream.
    Closes any open strings, arrays, and objects, then parses.
    """
    s = text.strip()

    # Close any open string (odd number of unescaped quotes after last complete value)
    # Simple heuristic: if last char isn't a closer, append enough closers
    in_string = False
    for i, ch in enumerate(s):
        if ch == '"' and (i == 0 or s[i-1] != '\\'):
            in_string = not in_string

    if in_string:
        s += '"'        # close the open string

    # Remove trailing comma before we add closers
    s = re.sub(r",\s*$", "", s)

    # Count unclosed brackets
    opens = s.count('{') - s.count('}')
    arrs  = s.count('[') - s.count(']')

    # Close arrays first, then objects
    s += ']' * max(arrs, 0)
    s += '}' * max(opens, 0)

    return json.loads(s)


def _parse_json_from_response(text: str) -> dict:
    """Extract JSON from Ollama response — handles prose wrappers, fences, truncation."""

    # 1. Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Strip ```json ... ``` fences
    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # 3. Extract first balanced { ... } block
    start = text.find('{')
    if start != -1:
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    candidate = text[start:i+1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        # Fix trailing commas and single quotes
                        fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
                        fixed = re.sub(r"'([^']*)'", r'"\1"', fixed)
                        try:
                            return json.loads(fixed)
                        except json.JSONDecodeError:
                            break

        # 4. ← NEW: JSON was truncated — brace scan never closed, try to repair
        candidate = text[start:]
        # Fix trailing commas first
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            return _repair_truncated_json(candidate)
        except (json.JSONDecodeError, Exception):
            pass

    raise ValueError(
        f"Ollama did not return valid JSON.\n"
        f"First 500 chars of response:\n{text[:500]}"
    )


# ── Full document analysis ─────────────────────────────────────────────────────
def analyse_document(full_text: str, page_count: int, chunk_count: int) -> Dict:
    """
    Send the full document text to Ollama and get a structured health analysis.

    Returns a dict matching the frontend's expected shape:
    {
        risk_level: str,
        pages: int,
        chunks: int,
        diseases_found: int,
        districts_found: int,
        key_findings: List[str],
        recommendations: List[str],
        summary: str,
    }
    """
    truncated = _truncate(full_text)

    prompt = f"""You are Shastho AI, a public health intelligence system for Bangladesh.

Analyse the following health report and respond ONLY with a valid JSON object — no extra text, no markdown fences.

JSON schema (fill every field):
{{
  "risk_level": "<critical|high|medium|low>",
  "diseases_found": <integer count of distinct diseases mentioned>,
  "districts_found": <integer count of distinct districts mentioned>,
  "key_findings": [<3-6 concise bullet strings, each under 120 chars>],
  "recommendations": [<3-5 actionable recommendation strings, each under 120 chars>],
  "summary": "<2-3 sentence plain-language executive summary>"
}}

Rules:
- risk_level is the OVERALL risk level of the situation described in the report.
- key_findings must be specific facts extracted from the report (numbers, names, trends).
- recommendations must be concrete and actionable for health officials.
- Respond ONLY with JSON — absolutely no other text.

REPORT TEXT:
─────────────────────────────
{truncated}
─────────────────────────────"""

    raw    = _ollama_generate(prompt, max_tokens=1024)
    result = _parse_json_from_response(raw)

    # Merge in metadata from our own processing
    result["pages"]  = page_count
    result["chunks"] = chunk_count

    return result

'''
stordata  = extract_full_text('./pdf/T1.2.pdf')
pagecount = get_page_count('./pdf/T1.2.pdf')


resultstor = analyse_document(stordata, pagecount, 6)
print(resultstor)

'''

# ── RAG Q&A ────────────────────────────────────────────────────────────────────
def ask_question(doc_id: str, question: str) -> Dict:
    """
    1. Embed the question and retrieve top-5 relevant chunks from ChromaDB.
    2. Build a context-rich prompt and send to Ollama.
    3. Return the answer + source citations.

    Returns:
    {
        "answer":  str,
        "sources": [{ "page": int, "snippet": str }, ...]
    }
    """
    # Step 1 — retrieve relevant chunks
    chunks = retrieve(doc_id, question, top_k=5)
    

    if not chunks:
        return {
            "answer":  "I couldn't find relevant information in the report to answer that question.",
            "sources": [],
        }

    # Step 2 — build context block
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Source {i} — Page {chunk['page']} — Relevance {chunk['score']}]\n{chunk['text']}"
        )
    context = "\n\n".join(context_parts)

    # Step 3 — prompt Ollama
    prompt = f"""You are Shastho AI, a public health intelligence assistant for Bangladesh.

Answer the user's question using ONLY the report excerpts provided below.
- Be concise and specific. Use numbers and names from the excerpts when available.
- If the excerpts don't contain enough information, say so honestly.
- Do NOT make up information not present in the excerpts.
- Format your answer in clear, plain language suitable for a health official.

REPORT EXCERPTS:
─────────────────────────────
{context}
─────────────────────────────

QUESTION: {question}

ANSWER:"""

    answer = _ollama_generate(prompt, max_tokens=800)

    # Step 4 — format sources for frontend
    sources = [
        {
            "page":    chunk["page"],
            "snippet": chunk["text"][:120] + "..." if len(chunk["text"]) > 120 else chunk["text"],
        }
        for chunk in chunks[:3]   # show top 3 sources
    ]

    return {"answer": answer, "sources": sources}


finalanswer=ask_question("91101","Is there any risk")
print(finalanswer)