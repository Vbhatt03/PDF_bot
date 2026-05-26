# server.py
import shutil
import tempfile
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from src.ingest import extract, chunk_pages, embed_chunks
from src.query import ask
load_dotenv()
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
async def chat(payload: dict):
    try:
        query = payload.get("query", "").strip()
        source = payload.get("source", "")
        provider = payload.get("provider", "ollama")
        if not query or not source:
            return JSONResponse({"error": "query and source are required"}, status_code=400)
        return ask(query, source=source, provider=provider)
    except RuntimeError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": f"Chat failed: {str(e)}"}, status_code=500)