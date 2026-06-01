# server.py
import sys
import os
import shutil
import asyncio
import tempfile
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

def get_setup_script() -> str:
    if getattr(sys, "frozen", False):
        base = sys._MEIPASS
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "setup_ollama.sh")
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
    script = get_setup_script()

    async def generate():
        if not os.path.isfile(script):
            yield "data: ERROR: setup_ollama.sh not found\n\n"
            yield "data: __ERROR__\n\n"
            return
        os.chmod(script, 0o755)
        process = await asyncio.create_subprocess_exec(
            "bash", script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        async for line in process.stdout:
            text = line.decode(errors="replace").rstrip()
            if text:
                yield f"data: {text}\n\n"
        await process.wait()
        if process.returncode == 0:
            yield "data: __DONE__\n\n"
        else:
            yield f"data: __ERROR__ (exit code {process.returncode})\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })


@app.post("/ingest")
async def ingest(file: UploadFile = File(...), provider: str = "ollama"): 
    try:
        dest = UPLOAD_DIR / file.filename
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
@app.post("/chat")
async def chat(payload: dict):
    query = payload.get("query", "").strip()
    source = payload.get("source")       # can be str, list, or None
    provider = payload.get("provider", "ollama")
    if not query:
        return JSONResponse({"error": "query is required"}, status_code=400)
    if not source:
        return JSONResponse({"error": "source is required"}, status_code=400)
    return ask(query, source=source, provider=provider)