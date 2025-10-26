# DocQuery

Local service for question answering over your own PDFs — fully offline. Upload documents, index their text into embeddings, and query them using open models via **Ollama**. No cloud costs, no API keys.

## Architecture

```
Upload → Extract (PyMuPDF / OCR) → Chunk → Embed → Store
Ask    → Embed → Retrieve → Generate → Answer
```

## Quick Start

```bash
git clone https://github.com/Argotoss/DocQuery.git
cd DocQuery
docker compose up --build
# Add --profile nvidia or --profile amd for GPU acceleration
```

On first run, this starts Ollama, pulls `nomic-embed-text` and `llama3.1:8b`, and launches the API on port **8000**.

Access the web UI at <http://localhost:8000> or use the REST API directly.

## Example

```bash
# Upload a PDF
curl -X POST "http://localhost:8000/upload" -F "file=@sample.pdf" -o upload.json
DOC_ID=$(jq -r '.doc_id' upload.json)

# Ask a question
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d "{\"doc_id\": \"$DOC_ID\", \"question\": \"What is the main topic?\"}" \
  -o answer.json

cat answer.json
```

## API Overview

Method | Endpoint | Description
------ | -------- | -----------
GET | `/health` | Check API and Ollama status
POST | `/upload` | Upload and process a PDF into a local knowledge base
POST | `/ask` | Ask a question: returns answer and cited chunks
GET | `/documents` | List indexed documents
GET | `/documents/{doc_id}` | Get document metadata
DELETE | `/documents/{doc_id}` | Remove a document and its data

## Configuration

Common environment variables:

Variable | Default | Description
-------- | ------- | -----------
`OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama base URL
`CHAT_MODEL` | `llama3.1:8b` | Model used for answers
`EMBED_MODEL` | `nomic-embed-text` | Embedding model
`ENABLE_OCR` | `false` | Enable OCR for scanned PDFs
`MAX_FILE_MB` | `50` | Max upload size (MB)
`TOP_K` | `5` | Number of chunks retrieved

## Data Layout

Each document creates a local directory:

```
data/<doc_id>/
├─ source.pdf
├─ meta.json
├─ chunks.jsonl      # text chunks with metadata
└─ embeddings.npy    # vector matrix
```

## Stack

FastAPI · Python · Ollama · PyMuPDF · NumPy · Docker

## License

MIT License · built by [Daniel Kozak](https://github.com/Argotoss)
