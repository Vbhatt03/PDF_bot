# server.py
import shutil
import tempfile
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from src.ingest import extract, chunk_pages, embed_chunks
from src.query import ask

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


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    dest = UPLOAD_DIR / file.filename
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    pages = extract(str(dest))
    chunks = chunk_pages(pages)
    embed_chunks(chunks)

    return {"source": str(dest), "pages": len(pages), "chunks": len(chunks)}


@app.post("/chat")
async def chat(payload: dict):
    query = payload.get("query", "").strip()
    source = payload.get("source", "")
    if not query or not source:
        return JSONResponse({"error": "query and source are required"}, status_code=400)

    result = ask(query, source=source)
    return result