# AI-doc-chat

FastAPI service for uploading PDFs, processing them into knowledge bases (text extraction, chunking, embeddings), and answering questions with retrieval-augmented generation. Supports document management, quality guardrails, and optional OCR for scanned PDFs.

**Didn't want to pay, so it's free:** Uses local AI models via Ollama – no cloud fees.

## Architecture Diagram

```
Upload PDF → Extract Text (PyMuPDF + OCR) → Chunk Text → Embed Chunks → Store KB
                                                                 ↓
Ask Question → Embed Query → Retrieve Top-K → Build Context → Generate Answer
```

## Quick Start

### CPU-only (default)
```bash
git clone https://github.com/Argotoss/AI-docs-chat.git
cd AI-docs-chat
docker-compose up --build
```

### With GPU acceleration
```bash
# For NVIDIA GPU
docker-compose --profile nvidia up --build

# For AMD GPU  
docker-compose --profile amd up --build
```

Starts Ollama, pulls `nomic-embed-text` and `llama3.1:8b` models, and launches the API on port 8000.

> **Performance Note**: GPU acceleration can reduce response times from 30+ seconds to 1-3 seconds for typical questions.

Visit `http://localhost:8000` for the web UI or use the API endpoints below.

## Demo Script
```bash
# Upload a PDF
curl -X POST "http://localhost:8000/upload" -F "file=@sample.pdf" -o upload.json
DOC_ID=$(jq -r '.doc_id' upload.json)

# Ask a question
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d "{\"doc_id\": \"$DOC_ID\", \"question\": \"What is the main topic?\"}" \
  -o answer.json

# View answer
cat answer.json
```

## Endpoints
- `GET /health` - Check API and Ollama status.
- `POST /upload` - Upload PDF, process into KB, return metadata.
- `POST /ask` - Query KB: `{"doc_id": "...", "question": "..."}` → `{"answer": "...", "citations": [...]}` or refusal if low relevance.
- `GET /documents` - List all documents with metadata and chunk counts.
- `GET /documents/{doc_id}` - Get detailed info for a document.
- `DELETE /documents/{doc_id}` - Delete a document and all files.

## Upload & Processing Flow
1. POST `/upload` with PDF file.
2. File saved to `data/<doc_id>/source.pdf`.
3. Metadata saved to `meta.json`.
4. Text extracted per page (PyMuPDF); OCR fallback if enabled and page has no text.
5. Text chunked into overlapping windows (filtered for min length).
6. Chunks embedded via Ollama (`nomic-embed-text`).
7. Chunks saved to `chunks.jsonl`; embeddings to `embeddings.npy`.

## Q&A Flow
1. Embed question with `nomic-embed-text`.
2. Compute cosine similarity to stored embeddings.
3. Retrieve top-k chunks (if similarity > threshold).
4. Generate answer with `llama3.1:8b` (concise, cited, deterministic).
5. Return answer + citations, or "Not enough relevant information" if low similarity.

## Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama base URL |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `CHAT_MODEL` | `llama3.1:8b` | Chat model for answers |
| `CHUNK_CHARS` | `1000` | Max chars per chunk |
| `CHUNK_OVERLAP` | `150` | Chunk overlap |
| `TOP_K` | `5` | Top chunks for retrieval |
| `MAX_FILE_MB` | `50` | Max upload size (MB) |
| `SIMILARITY_THRESHOLD` | `0.3` | Min similarity to answer |
| `MAX_ANSWER_LENGTH` | `500` | Max answer words |
| `MIN_CHUNK_LENGTH` | `100` | Min chunk chars (filter noise) |
| `ENABLE_OCR` | `false` | Enable OCR fallback for image PDFs |
| `MAX_PAGES` | `100` | Max pages per PDF |
| `MAX_CONTEXT_CHARS` | `8000` | Max context chars for Q&A |

## Data Artifacts per Document
```
data/<doc_id>/
	source.pdf
	meta.json
	chunks.jsonl        # {id, doc_id, index, page, start, end, text}
	embeddings.npy      # float32 matrix (num_chunks x dim)
```

## Logs
Query logs are saved to `logs/queries.jsonl` with details: timestamp, doc_id, question, latency, top-k scores, truncated flag, refused flag.

## Known Issues
- OCR requires Tesseract installed on host (for ENABLE_OCR=true).
- Large PDFs may take time to process; tune MAX_PAGES.
- Models must be pulled manually if not auto-pulled by Ollama.

