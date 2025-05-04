import os
import requests
from fastapi import FastAPI

api = FastAPI(title="AI Chat for Documents", version="0.0.1")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

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

@api.get("/")
def root():
    return {"ok": True, "docs": "/docs", "health": "/health"}
