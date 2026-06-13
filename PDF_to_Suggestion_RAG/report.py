"""
routers/report.py
─────────────────────────────────────────────────────────────────
Four endpoints consumed by the React frontend:

  POST /api/report/upload      — upload PDF, extract, chunk, embed
  POST /api/report/analyse     — full-doc structured analysis via Claude
  POST /api/report/ask         — RAG Q&A for the chat interface
  GET  /api/report/download/{doc_id} — stream generated analysis PDF
"""

import os
import uuid
import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from extractor     import extract_pages, extract_full_text, get_page_count
from vector_store  import chunk_pages, store_chunks, delete_doc
from rag_engine    import analyse_document, ask_question
from pdf_generator import generate_report_pdf

load_dotenv()

router     = APIRouter()
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))
UPLOAD_DIR.mkdir(exist_ok=True)

# In-memory store for analysis results and filenames
# (use Redis / a DB in production)
_analysis_cache: dict[str, dict]  = {}
_filename_cache: dict[str, str]   = {}


# ── Request / Response models ──────────────────────────────────────────────────

class AnalyseRequest(BaseModel):
    doc_id: str

class AskRequest(BaseModel):
    doc_id: str
    question: str


# ── 1. Upload ──────────────────────────────────────────────────────────────────

@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Accept a PDF, save it, extract text page-by-page,
    chunk it, and embed into ChromaDB.

    Returns { doc_id } which the frontend uses for all subsequent calls.
    """
    # Validate file type
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Read and size-check
    content = await file.read()
    max_bytes = int(os.getenv("MAX_PDF_MB", 20)) * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail="File too large. Maximum 20 MB.")

    # Generate a unique doc ID and save file
    doc_id    = str(uuid.uuid4()).replace("-", "")[:16]
    pdf_path  = UPLOAD_DIR / f"{doc_id}.pdf"
    pdf_path.write_bytes(content)

    # Cache original filename
    _filename_cache[doc_id] = file.filename or "report.pdf"

    try:
        # Extract text per page
        pages = extract_pages(str(pdf_path))
        if not pages:
            raise HTTPException(status_code=422, detail="PDF has no extractable text.")

        # Chunk and embed into ChromaDB
        chunks = chunk_pages(pages)
        store_chunks(doc_id, chunks)

    except HTTPException:
        raise
    except Exception as e:
        # Clean up on failure
        pdf_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

    return {"doc_id": doc_id, "pages": len(pages), "chunks": len(chunks)}


# ── 2. Analyse ────────────────────────────────────────────────────────────────

@router.post("/analyse")
async def analyse(req: AnalyseRequest):
    """
    Run full-document structured analysis via Claude.
    Result is cached in memory for subsequent download requests.
    """
    pdf_path = UPLOAD_DIR / f"{req.doc_id}.pdf"
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="Document not found. Please upload again.")

    try:
        full_text   = extract_full_text(str(pdf_path))
        page_count  = get_page_count(str(pdf_path))

        # Chunk count from ChromaDB collection (approximate via page count)
        # We re-derive it rather than storing separately
        from extractor import extract_pages as _ep
        from vector_store import chunk_pages as _cp
        chunks      = _cp(_ep(str(pdf_path)))
        chunk_count = len(chunks)

        result = analyse_document(full_text, page_count, chunk_count)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    # Cache the result
    _analysis_cache[req.doc_id] = result
    return result


# ── 3. Ask ─────────────────────────────────────────────────────────────────────

@router.post("/ask")
async def ask(req: AskRequest):
    """
    RAG Q&A — retrieve relevant chunks from ChromaDB and answer via Claude.
    Returns { answer, sources }.
    """
    if not (UPLOAD_DIR / f"{req.doc_id}.pdf").exists():
        raise HTTPException(status_code=404, detail="Document not found.")

    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = ask_question(req.doc_id, req.question)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG failed: {str(e)}")

    return result


# ── 4. Download ────────────────────────────────────────────────────────────────

@router.get("/download/{doc_id}")
async def download(doc_id: str):
    """
    Generate and stream a professional PDF analysis report.
    Uses cached analysis if available, otherwise re-analyses.
    """
    pdf_path = UPLOAD_DIR / f"{doc_id}.pdf"
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="Document not found.")

    # Get or re-generate analysis
    analysis = _analysis_cache.get(doc_id)
    if not analysis:
        try:
            full_text   = extract_full_text(str(pdf_path))
            page_count  = get_page_count(str(pdf_path))
            from extractor import extract_pages as _ep
            from vector_store import chunk_pages as _cp
            chunk_count = len(_cp(_ep(str(pdf_path))))
            analysis    = analyse_document(full_text, page_count, chunk_count)
            _analysis_cache[doc_id] = analysis
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Re-analysis failed: {str(e)}")

    original_filename = _filename_cache.get(doc_id, "report.pdf")

    try:
        pdf_bytes = generate_report_pdf(analysis, doc_id, original_filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")

    return StreamingResponse(
        iter([pdf_bytes]),  
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="shastho-ai-{doc_id}.pdf"',
            "Content-Length": str(len(pdf_bytes)),
        },
    )


# ── 5. Delete (optional cleanup) ──────────────────────────────────────────────

@router.delete("/delete/{doc_id}")
async def delete(doc_id: str):
    """Remove uploaded PDF and its vector store."""
    pdf_path = UPLOAD_DIR / f"{doc_id}.pdf"
    pdf_path.unlink(missing_ok=True)
    delete_doc(doc_id)
    _analysis_cache.pop(doc_id, None)
    _filename_cache.pop(doc_id, None)
    return {"deleted": doc_id}
