import fitz  # pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import chromadb
from chromadb.config import Settings

try:
    import ollama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False
import google.generativeai as genai                          

CHROMA_DIR = "chroma_db"
COLLECTION_NAMES = {                                       
    "ollama": "pdf_bot_ollama",
    "gemini": "pdf_bot_gemini",
}


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

def extract_csv(csv_path: str, rows_per_page: int = 50) -> list[dict]:
    """Extract text from a CSV file. Groups rows_per_page rows into each 'page'."""
    import csv
    pages = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for i in range(0, len(rows), rows_per_page):
        batch = rows[i:i + rows_per_page]
        lines = []
        for row in batch:
            lines.append(", ".join(f"{k}: {v}" for k, v in row.items()))
        text = "\n".join(lines)
        if text.strip():
            pages.append({"page": i // rows_per_page + 1, "text": text})
    return pages

def extract_pdf_ocr(pdf_path: str) -> list[dict]:
    import pdfplumber
    import pytesseract
    from pdf2image import convert_from_path
    from PIL import ImageDraw

    pages = []
    images = convert_from_path(pdf_path)  # one PIL image per page

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, (plumber_page, image) in enumerate(zip(pdf.pages, images), start=1):
            parts = []

            # --- Tables: only use pdfplumber if it can actually read the cells ---
            masked_bboxes = []
            for tbl_obj, tbl_data in zip(plumber_page.find_tables(), plumber_page.extract_tables()):
                if not tbl_data:
                    continue
                flat = [cell for row in tbl_data for cell in row if cell and cell.strip()]
                if flat:
                    # pdfplumber could read the cell text — format as markdown and mask from OCR
                    rows = [[cell or "" for cell in row] for row in tbl_data]
                    header = rows[0]
                    md  = "| " + " | ".join(header) + " |\n"
                    md += "| " + " | ".join(["---"] * len(header)) + " |\n"
                    for row in rows[1:]:
                        md += "| " + " | ".join(row) + " |\n"
                    parts.append(md)
                    masked_bboxes.append(tbl_obj.bbox)
                # else: scanned table — leave it for OCR to handle

            # --- OCR: mask only tables that pdfplumber already extracted ---
            img_copy = image.copy()
            draw = ImageDraw.Draw(img_copy)
            pw, ph = plumber_page.width, plumber_page.height
            iw, ih = image.size

            for x0, top, x1, bottom in masked_bboxes:
                draw.rectangle([
                    int(x0 / pw * iw), int(top / ph * ih),
                    int(x1 / pw * iw), int(bottom / ph * ih)
                ], fill="white")

            text = pytesseract.image_to_string(img_copy)
            if text.strip():
                parts.append(text)

            combined = "\n\n".join(parts)
            if combined.strip():
                pages.append({"page": page_num, "text": combined})

    return pages

def extract(file_path: str) -> list[dict]:
    """
    Unified entry point. Dispatches to PDF or TXT extractor based on extension.
    For PDFs, automatically falls back to OCR if extract_pdf returns empty or minimal content.
    Each returned dict has:
        - 'page'  : int  (1-indexed page number)
        - 'text'  : str  (extracted text for that page)
        - 'source': str  (original file path)
    
    Args:
        file_path: Path to PDF, TXT, or CSV file
    """
    if file_path.lower().endswith(".pdf"):
        pages = extract_pdf(file_path)
        # Fall back to OCR if extraction returned empty or very little content
        if not pages or sum(len(p["text"]) for p in pages) < 100:
            print(f"Low content from standard extraction, attempting OCR...")
            pages = extract_pdf_ocr(file_path)
    elif file_path.lower().endswith(".txt"):
        pages = extract_txt(file_path)
    elif file_path.lower().endswith(".csv"):
        pages = extract_csv(file_path)
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

def get_collection(provider: str = "ollama"):                # ADD provider param
    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAMES[provider],                     # USE dict
        metadata={"hnsw:space": "cosine"},
    )



def embed_chunks(chunks: list[dict], provider: str = "ollama") -> None:   # ADD provider param
    if provider == "ollama" and not HAS_OLLAMA:
        raise RuntimeError(
            "Ollama is not installed. Install it with: pip install ollama\n"
            "Or install the optional dependency: pip install -e .[ollama]\n"
            "Once Ollama installation is complete, run: ollama pull mxbai-embed-large\n"
            "ollama pull phi3:mini"
        )
    
    collection = get_collection(provider)                    # PASS provider
    texts = [c["text"] for c in chunks]

    if provider == "gemini":                                 # ADD branch
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        result = genai.embed_content(
            model="models/gemini-embedding-2",
            content=texts,
            task_type="retrieval_document",
        )
            
        embeddings = result["embedding"]
    else:
        response = ollama.embed(model="mxbai-embed-large", input=texts)
        embeddings = response["embeddings"]

    ids = [f"{c['source']}::{c['page']}::{c['chunk_index']}" for c in chunks]
    metadatas = [
        {"source": c["source"], "page": c["page"], "chunk_index": c["chunk_index"]}
        for c in chunks
    ]

    collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    print(f"Upserted {len(chunks)} chunks into '{COLLECTION_NAMES[provider]}'.")


if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("--provider", choices=["ollama", "gemini"], default="ollama")
    args = parser.parse_args()

    pages = extract(args.file)
    print(f"Extracted {len(pages)} page(s).")
    chunks = chunk_pages(pages)
    print(f"Produced {len(chunks)} chunk(s).")
    embed_chunks(chunks, provider=args.provider)