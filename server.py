# server.py
import sys
import os
import shutil
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from src.ingest import extract, chunk_pages, embed_chunks
from src.query import ask
load_dotenv()

try:
    import ollama as _ollama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@app.get("/")
def index():
    return FileResponse("index.html")


@app.get("/setup/status")
def setup_status():
    """Check if Ollama is ready. Return ready=false with setup instructions if not."""
    ollama_bin = shutil.which("ollama") is not None
    models_ready = False
    if ollama_bin and HAS_OLLAMA:
        try:
            names = [m.model for m in _ollama.list().models]
            models_ready = (
                any("mxbai-embed-large" in n for n in names) and
                any("phi3" in n for n in names)
            )
        except Exception:
            pass
    return {"ready": ollama_bin and models_ready}


@app.get("/setup/stream")
async def setup_stream():
    """Stream setup error message and instructions to the UI."""
    async def generate():
        yield "data: ERROR: Ollama is not installed or not available.\n\n"
        yield "data: \n\n"
        yield "data: To install Ollama, please run the setup script:\n\n"
        yield "data:   bash setup_ollama.sh\n\n"
        yield "data: \n\n"
        yield "data: This script will download and install Ollama with the required models.\n\n"
        yield "data: __ERROR__\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })


@app.post("/ingest")
async def ingest(file: UploadFile = File(...), provider: str = "ollama"): 
    if provider == "ollama" and not HAS_OLLAMA:
        return JSONResponse({
            "error": "Ollama is not installed. Please run: bash setup_ollama.sh"
        }, status_code=500)
    
    try:
        dest = UPLOAD_DIR / file.filename
        ALLOWED_EXTENSIONS = {".pdf", ".txt", ".csv", ".glb"}
        if Path(file.filename).suffix.lower() not in ALLOWED_EXTENSIONS:
            return JSONResponse(
                {"error": f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"},
                status_code=400,
            )
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        pages = extract(str(dest))
        chunks = chunk_pages(pages)
        embed_chunks(chunks, provider=provider)                
        return {"source": str(dest), "pages": len(pages), "chunks": len(chunks)}
    except RuntimeError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": f"Ingest failed: {str(e)}"}, status_code=500)


@app.post("/chat")
async def chat(payload: dict):
    query = payload.get("query", "").strip()
    source = payload.get("source")       # can be str, list, or None
    provider = payload.get("provider", "ollama")
    if not query:
        return JSONResponse({"error": "query is required"}, status_code=400)
    if not source:
        return JSONResponse({"error": "source is required"}, status_code=400)
    
    if provider == "ollama" and not HAS_OLLAMA:
        return JSONResponse({
            "error": "Ollama is not installed. Please run: bash setup_ollama.sh"
        }, status_code=500)
    
    return ask(query, source=source, provider=provider)