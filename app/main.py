import os
from dotenv import load_dotenv
load_dotenv()

import requests
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from . import storage

api = FastAPI(title="AI Chat for Documents", version="0.0.1")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

MAX_FILE_MB = int(os.getenv("MAX_FILE_MB", "50"))

@api.get("/health")
def health():
    status = {"api": True, "ollama": {"reachable": False, "models": []}}
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        if r.ok:
            status["ollama"]["reachable"] = True
            data = r.json() or {}
            models = [m.get("name") for m in data.get("models", [])]
            status["ollama"]["models"] = models
    except Exception:
        pass
    return status

@api.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_FILE_MB:
        raise HTTPException(status_code=413, detail=f"File exceeds size limit of {MAX_FILE_MB} MB.")
    
    doc_id = storage.new_doc_id(file.filename)
    pdf_path = storage.save_source_pdf(doc_id, content)
    meta = storage.save_meta(doc_id, file.filename, pdf_path)

    kb_summary = {}
    try:
        kb_summary = storage.build_knowledge_base(doc_id, pdf_path)
    except storage.ProcessingError as e:
        kb_summary = {"error": str(e)}  

    return JSONResponse({
        "doc_id": doc_id,
        "filename": file.filename,
        "size_bytes": len(content),
        "created_at": meta["created_at"],
        "kb": kb_summary,
    })

@api.get("/")
def root():
    return {"ok": True, "docs": "/docs", "health": "/health"}
