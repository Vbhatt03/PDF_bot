# extractor.py
import fitz  # pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import chromadb
from chromadb.config import Settings
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv() 

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "pdf_bot"


def extract_pdf(pdf_path: str) -> list[dict]:
    """Extract text page-by-page from a PDF. Returns list of {page, text} dicts."""
    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        if text.strip():
            pages.append({"page": page_num, "text": text})
    doc.close()
    return pages


def extract_txt(txt_path: str) -> list[dict]:
    """Extract text from a plain .txt file. Treated as a single page."""
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()
    return [{"page": 1, "text": text}] if text.strip() else []


def extract(file_path: str) -> list[dict]:
    """
    Unified entry point. Dispatches to PDF or TXT extractor based on extension.
    Each returned dict has:
        - 'page'  : int  (1-indexed page number)
        - 'text'  : str  (extracted text for that page)
        - 'source': str  (original file path)
    """
    if file_path.lower().endswith(".pdf"):
        pages = extract_pdf(file_path)
    elif file_path.lower().endswith(".txt"):
        pages = extract_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

    for page in pages:
        page["source"] = file_path

    return pages

def chunk_pages(pages: list[dict], chunk_size: int = 1500, chunk_overlap: int = 300) -> list[dict]:
    """
    Split extracted pages into overlapping chunks.
    Each returned dict has: 'text', 'page', 'source', 'chunk_index'.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = []
    for page in pages:
        splits = splitter.split_text(page["text"])
        for i, split in enumerate(splits):
            chunks.append({
                "text": split,
                "page": page["page"],
                "source": page["source"],
                "chunk_index": i,
            })
    return chunks

def get_collection():
    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # explicit cosine — don't rely on default
    )


def embed_chunks(chunks: list[dict]) -> None:
    """Embed each chunk with Gemini text-embedding-004 and upsert into Chroma."""
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    collection = get_collection()

    texts = [c["text"] for c in chunks]

    # retrieval_document task type — optimised for ingestion side
    response = genai.embed_content(
        model="models/gemini-embedding-2",
        content=texts,
        task_type="retrieval_document",
    )
    embeddings = response["embedding"]  # list of lists

    ids = [
        f"{c['source']}::{c['page']}::{c['chunk_index']}"
        for c in chunks
    ]
    metadatas = [
        {"source": c["source"], "page": c["page"], "chunk_index": c["chunk_index"]}
        for c in chunks
    ]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    print(f"Upserted {len(chunks)} chunks into '{COLLECTION_NAME}'.")

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ingest.py <file.pdf|file.txt>")
        sys.exit(1)

    pages = extract(sys.argv[1])
    print(f"Extracted {len(pages)} page(s).")

    chunks = chunk_pages(pages)
    print(f"Produced {len(chunks)} chunk(s).")

    embed_chunks(chunks)