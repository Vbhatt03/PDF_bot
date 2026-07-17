import fitz  # pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import chromadb
from chromadb.config import Settings
from . import tokens

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
    """Extract text from a CSV file. Formats as markdown table for efficiency."""
    import csv
    pages = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    if not rows:
        return pages
    
    headers = list(rows[0].keys())
    
    for i in range(0, len(rows), rows_per_page):
        batch = rows[i:i + rows_per_page]
        
        # Build markdown table
        md = "| " + " | ".join(headers) + " |\n"
        md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        for row in batch:
            values = [str(row.get(h, "")).strip()[:50] for h in headers]  # truncate long cells
            md += "| " + " | ".join(values) + " |\n"
        
        if md.strip():
            pages.append({"page": i // rows_per_page + 1, "text": md})
    
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


def extract_glb(glb_path: str) -> list[dict]:
    """Parse a GLB/GLTF file and return structured text pages for each metadata category."""
    from pygltflib import GLTF2
    import struct, base64

    # Read raw bytes once — avoids extension-based dispatch and text-mode encoding issues
    with open(glb_path, "rb") as _fh:
        _raw = _fh.read()

    if _raw[:4] == b"glTF":
        # Standard binary GLB container
        gltf = GLTF2.load_from_bytes(_raw)
    else:
        # JSON-based GLTF (text), possibly with UTF-8 BOM
        try:
            _text = _raw.decode("utf-8-sig")   # strips BOM if present
        except UnicodeDecodeError:
            _text = _raw.decode("latin-1")     # fallback for non-UTF-8 text
        gltf = GLTF2.gltf_from_json(_text)
        gltf.filename = glb_path

    pages = []

    # ── Page 1: Scene Hierarchy ───────────────────────────────────────────────
    def node_name(idx: int) -> str:
        n = gltf.nodes[idx]
        return n.name if n.name else f"Node_{idx}"

    def build_tree(idx: int, depth: int = 0) -> list[str]:
        lines = ["  " * depth + f"- {node_name(idx)}"]
        node = gltf.nodes[idx]
        for child_idx in (node.children or []):
            lines.extend(build_tree(child_idx, depth + 1))
        return lines

    if gltf.scenes:
        scene = gltf.scenes[gltf.scene if gltf.scene is not None else 0]
        scene_name = scene.name if scene.name else "Scene"
        tree_lines = [f"Scene: {scene_name}"]
        for root_idx in (scene.nodes or []):
            tree_lines.extend(build_tree(root_idx))
        pages.append({"page": 1, "text": "Scene Hierarchy:\n" + "\n".join(tree_lines)})

    # ── Page 2: Object / Node Names ──────────────────────────────────────────
    object_lines = []
    for i, node in enumerate(gltf.nodes or []):
        name = node.name if node.name else f"Node_{i}"
        kind = "Empty"
        if node.mesh is not None:
            mesh_name = gltf.meshes[node.mesh].name or f"Mesh_{node.mesh}"
            kind = f"Mesh ({mesh_name})"
        elif node.camera is not None:
            kind = "Camera"
        elif node.skin is not None:
            kind = "Skinned Mesh"
        object_lines.append(f"  {name}: {kind}")
    if object_lines:
        pages.append({"page": 2, "text": "Objects (Nodes):\n" + "\n".join(object_lines)})

    # ── Page 3: Materials ─────────────────────────────────────────────────────
    mat_lines = []
    for i, mat in enumerate(gltf.materials or []):
        name = mat.name if mat.name else f"Material_{i}"
        props = [f"name={name}"]
        pbr = mat.pbrMetallicRoughness
        if pbr:
            if pbr.baseColorFactor:
                r, g, b, a = pbr.baseColorFactor
                props.append(f"baseColor=rgba({r:.2f},{g:.2f},{b:.2f},{a:.2f})")
            if pbr.metallicFactor is not None:
                props.append(f"metallic={pbr.metallicFactor:.2f}")
            if pbr.roughnessFactor is not None:
                props.append(f"roughness={pbr.roughnessFactor:.2f}")
        if mat.alphaMode:
            props.append(f"alphaMode={mat.alphaMode}")
        if mat.doubleSided:
            props.append("doubleSided=true")
        mat_lines.append("  " + ", ".join(props))
    if mat_lines:
        pages.append({"page": 3, "text": "Materials:\n" + "\n".join(mat_lines)})

    # ── Page 4: Animations ────────────────────────────────────────────────────
    anim_lines = []
    for i, anim in enumerate(gltf.animations or []):
        name = anim.name if anim.name else f"Animation_{i}"
        targets = []
        for ch in (anim.channels or []):
            target_node_idx = ch.target.node
            target_name = node_name(target_node_idx) if target_node_idx is not None else "?"
            targets.append(f"{target_name}.{ch.target.path}")
        anim_lines.append(f"  {name}: targets=[{', '.join(targets)}]")
    if anim_lines:
        pages.append({"page": 4, "text": "Animations:\n" + "\n".join(anim_lines)})
    else:
        pages.append({"page": 4, "text": "Animations: none"})

    # ── Page 5: Dimensions (bounding boxes per mesh) ─────────────────────────
    def read_accessor_min_max(accessor):
        """Return (min_xyz, max_xyz) from accessor metadata — no buffer decode needed."""
        if accessor.min and accessor.max:
            return accessor.min, accessor.max
        return None, None

    dim_lines = []
    for mesh_idx, mesh in enumerate(gltf.meshes or []):
        mesh_name = mesh.name if mesh.name else f"Mesh_{mesh_idx}"
        agg_min = [float("inf")] * 3
        agg_max = [float("-inf")] * 3
        has_data = False
        for prim in (mesh.primitives or []):
            if prim.attributes.POSITION is None:
                continue
            accessor = gltf.accessors[prim.attributes.POSITION]
            mn, mx = read_accessor_min_max(accessor)
            if mn and mx:
                agg_min = [min(agg_min[j], mn[j]) for j in range(3)]
                agg_max = [max(agg_max[j], mx[j]) for j in range(3)]
                has_data = True
        if has_data:
            w = agg_max[0] - agg_min[0]
            h = agg_max[1] - agg_min[1]
            d = agg_max[2] - agg_min[2]
            cx = (agg_min[0] + agg_max[0]) / 2
            cy = (agg_min[1] + agg_max[1]) / 2
            cz = (agg_min[2] + agg_max[2]) / 2
            dim_lines.append(
                f"  {mesh_name}: width={w:.4f}, height={h:.4f}, depth={d:.4f} "
                f"| center=({cx:.4f}, {cy:.4f}, {cz:.4f}) [model units]"
            )
    if dim_lines:
        pages.append({"page": 5, "text": "Dimensions (per mesh, model units):\n" + "\n".join(dim_lines)})

    return pages

def extract_stl(stl_path: str) -> list[dict]:
    """Parse an STL file and return structured text pages."""
    from stl import mesh as stl_mesh
    import numpy as np

    m = stl_mesh.Mesh.from_file(stl_path)

    pages = []

    # ── Page 1: Header / Object Name ─────────────────────────────────────────
    # Binary STL has an 80-byte header; ASCII STL has a 'solid <name>' line
    with open(stl_path, "rb") as _fh:
        _raw = _fh.read(80)
    is_binary = _raw[:5] != b"solid"
    if is_binary:
        header = _raw.rstrip(b"\x00").decode("latin-1").strip()
        name = header if header else "Unnamed"
        fmt = "binary"
    else:
        first_line = _raw.decode("latin-1").split("\n")[0].strip()
        name = first_line[6:].strip() if first_line.lower().startswith("solid") else "Unnamed"
        fmt = "ASCII"
    pages.append({"page": 1, "text": f"STL Model Info:\n  name={name}\n  format={fmt}\n  triangles={len(m.vectors)}"})

    # ── Page 2: Dimensions ────────────────────────────────────────────────────
    verts = m.vectors.reshape(-1, 3)
    mn = verts.min(axis=0)
    mx = verts.max(axis=0)
    w = mx[0] - mn[0]
    h = mx[1] - mn[1]
    d = mx[2] - mn[2]
    cx, cy, cz = (mn + mx) / 2
    pages.append({"page": 2, "text": (
        f"Dimensions (model units):\n"
        f"  width={w:.4f}, height={h:.4f}, depth={d:.4f}\n"
        f"  min=({mn[0]:.4f}, {mn[1]:.4f}, {mn[2]:.4f})\n"
        f"  max=({mx[0]:.4f}, {mx[1]:.4f}, {mx[2]:.4f})\n"
        f"  center=({cx:.4f}, {cy:.4f}, {cz:.4f})"
    )})

    # ── Page 3: Mesh Statistics ───────────────────────────────────────────────
    areas = np.sqrt(np.sum(np.cross(
        m.vectors[:, 1] - m.vectors[:, 0],
        m.vectors[:, 2] - m.vectors[:, 0]
    ) ** 2, axis=1)) / 2
    surface_area = float(areas.sum())
    pages.append({"page": 3, "text": (
        f"Mesh Statistics:\n"
        f"  triangles={len(m.vectors)}\n"
        f"  surface_area={surface_area:.4f} [model units²]"
    )})

    return pages

def extract_json(json_path: str, rows_per_page: int = 50) -> list[dict]:
    """Extract text from a JSON file. Formats as markdown table for arrays of objects."""
    import json
    pages = []
    
    with open(json_path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    
    # Handle different JSON structures
    if isinstance(data, list):
        # Array of objects - treat like CSV
        if data and isinstance(data[0], dict):
            rows = data
            headers = list(data[0].keys())
            
            for i in range(0, len(rows), rows_per_page):
                batch = rows[i:i + rows_per_page]
                
                md = "| " + " | ".join(headers) + " |\n"
                md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
                for row in batch:
                    values = [str(row.get(h, "")).strip()[:50] for h in headers]
                    md += "| " + " | ".join(values) + " |\n"
                
                if md.strip():
                    pages.append({"page": i // rows_per_page + 1, "text": md})
        else:
            # Array of primitives or mixed types
            text = "\n".join(str(item) for item in data)
            if text.strip():
                pages.append({"page": 1, "text": text})
    
    elif isinstance(data, dict):
        # Object - convert to readable key-value format
        lines = []
        def flatten(obj, prefix=""):
            result = []
            if isinstance(obj, dict):
                for k, v in obj.items():
                    result.extend(flatten(v, f"{prefix}{k}."))
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    result.extend(flatten(item, f"{prefix}[{i}]."))
            else:
                result.append(f"{prefix.rstrip('.')}: {obj}")
            return result
        
        lines = flatten(data)
        if lines:
            pages.append({"page": 1, "text": "\n".join(lines)})
    
    else:
        # Primitive values
        text = str(data)
        if text.strip():
            pages.append({"page": 1, "text": text})
    
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
        file_path: Path to PDF, TXT, CSV, JSON, GLB, or STL file
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
    elif file_path.lower().endswith(".glb"):
        pages = extract_glb(file_path)
    elif file_path.lower().endswith(".stl"):
        pages = extract_stl(file_path)
    elif file_path.lower().endswith(".json"):
        pages = extract_json(file_path)
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



def embed_chunks(chunks: list[dict], provider: str = "ollama") -> None:
    if provider == "ollama" and not HAS_OLLAMA:
        raise RuntimeError(
            "Ollama is not installed. Install it with: pip install ollama\n"
            "Or install the optional dependency: pip install -e .[ollama]\n"
            "Once Ollama installation is complete, run: ollama pull mxbai-embed-large\n"
            "ollama pull phi3:mini"
        )
    
    collection = get_collection(provider)
    texts = [c["text"] for c in chunks]

    if provider == "gemini":
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        
        # Pre-check token usage
        total_text = "\n".join(texts)
        estimated = tokens.count_tokens_text(total_text, "gemini-embedding-2")
        tokens.warn_if_over_budget(estimated, model_name="gemini-embedding-2", label="ingest")
        
        result = genai.embed_content(
            model="models/gemini-embedding-2",
            content=texts,
            task_type="retrieval_document",
        )
        embeddings = result["embedding"]
        
        # Record usage
        tokens.record_ingest(
            source=chunks[0]["source"],
            chunk_count=len(chunks),
            token_estimate=estimated,
            model_name="gemini-embedding-2"
        )
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