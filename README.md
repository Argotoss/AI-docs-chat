# AI-doc-chat

Simple FastAPI service to upload PDFs, extract text, chunk, embed with Ollama, and store a lightweight knowledge base for future Q&A.

## Quick Start
```bash
docker-compose up --build
```
This will start Ollama, pull the `nomic-embed-text` model, and launch the API on port 8000.

## Upload Flow
1. POST `/upload` with a PDF file
2. File saved under `data/<doc_id>/source.pdf`
3. Metadata saved to `meta.json`
4. Text extracted per page using PyMuPDF
5. Text chunked into overlapping windows (defaults 1000 chars, 150 overlap)
6. Each chunk embedded via Ollama embeddings API using model `nomic-embed-text`
7. Chunks written to `chunks.jsonl` (one JSON object per line)
8. Embeddings stored as NumPy array `embeddings.npy`

## Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Base URL for local Ollama instance |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model name to request from Ollama |
| `CHUNK_CHARS` | `1000` | Target max characters per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap size between consecutive chunks |
| `MAX_FILE_MB` | `50` | Max upload size in megabytes |

## Data Artifacts per Document
```
data/<doc_id>/
	source.pdf
	meta.json
	chunks.jsonl        # {id, doc_id, index, page, start, end, text}
	embeddings.npy      # float32 matrix (num_chunks x dim)
```

## Future Work
- Optional OCR fallback (Tesseract) when a page has little/no extracted text.
- /ask endpoint to perform retrieval + LLM answer generation.

