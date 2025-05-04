import os
import time
from typing import Dict

DATA_DIR = os.path.join(os.getcwd(), "data")

def ensure_doc_dir(doc_id: str) -> str:
    d = os.path.join(DATA_DIR, doc_id)
    os.makedirs(d, exist_ok=True)
    return d

def new_doc_id(filename: str) -> str:
    ts = int(time.time() * 1000)
    safe = filename.replace(" ", "_")
    return f"{ts}_{safe}"

def save_source_pdf(doc_id: str, content_bytes: bytes) -> str:
    d = ensure_doc_dir(doc_id)
    src_path = os.path.join(d, "source.pdf")
    with open(src_path, "wb") as f:
        f.write(content_bytes)
    return src_path

def save_meta(doc_id: str, filename: str, pdf_path: str) -> Dict:
    meta = {
        "doc_id": doc_id,
        "filename": filename,
        "pdf_path": pdf_path,
        "created_at": int(time.time()),
    }
    with open(os.path.join(DATA_DIR, doc_id, "meta.json"), "w", encoding="utf-8") as f:
        import json
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return meta
