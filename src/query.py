# query.py
from .ingest import get_collection
import os
import google.generativeai as genai
from . import tokens
try:
    import ollama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False  

SIMILARITY_THRESHOLD = 0.4

SYSTEM_PROMPT = """You are a precise document and 3D model assistant. Follow these rules without exception:
1. Answer ONLY using information present in the CONTEXT provided below.
2. If the answer is not in the context, respond with exactly: [NOT FOUND]
3. When you use information from the context, cite the page number inline, e.g. (Page 3).
4. Do not infer, extrapolate, or use any external knowledge.
5. When the answer contains numeric/comparative data with 2 or more values, present it as a
   markdown table AND add a ```chart code block immediately after using this exact JSON shape:
   {"type":"bar","title":"...","labels":[...],"datasets":[{"label":"...","data":[...]}]}
   Supported types: bar, line, pie. Use pie only for parts-of-a-whole data.
   For the chart, include ONLY rows that have actual numeric values — skip any null, empty,
   or "not specified" entries. Always emit the chart block if at least 2 numeric values exist.
6. For 3D model context pages:
   - "Scene Hierarchy" pages describe the parent-child tree of objects in the scene.
   - "Objects (Nodes)" pages list every object with its type (Mesh, Camera, Empty, etc.).
   - "Materials" pages list material names and PBR properties (baseColor, metallic, roughness).
   - "Animations" pages list animation clip names and which object properties they drive.
   - "Dimensions" pages report per-mesh bounding box sizes in model units (not real-world units
     unless the file author set a specific scale). Width=X axis, Height=Y axis, Depth=Z axis.
   - "STL Model Info" pages report the mesh name (from the header), format (ASCII/binary), and triangle count.
   - STL files have no materials, hierarchy, or animations — only geometry and dimensions are available.
"""

def embed_query(query: str, provider: str = "ollama") -> list[float]:   # ADD provider
    if provider == "ollama" and not HAS_OLLAMA:
        raise RuntimeError(
            "Ollama is not installed. Install it with: pip install ollama\n"
            "Or install the optional dependency: pip install -e .[ollama]\n"
            "Once Ollama installation is complete, run: ollama pull mxbai-embed-large\n"
            "ollama pull phi3:mini"
        )
    
    if provider == "gemini":                                 # ADD branch
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        return genai.embed_content(
            model="models/gemini-embedding-2",
            content=query,
            task_type="retrieval_query",
        )["embedding"]
    response = ollama.embed(model="mxbai-embed-large", input=query)
    return response["embeddings"][0]

def retrieve_chunks(query_embedding, k=6, source=None, provider="ollama"):
    collection = get_collection(provider)

    if isinstance(source, list) and len(source) > 1:
        where_filter = {"source": {"$in": source}}
    elif isinstance(source, str) and source:
        where_filter = {"source": source}
    elif isinstance(source, list) and len(source) == 1:
        where_filter = {"source": source[0]}
    else:
        where_filter = None  # search across all ingested docs

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
        where=where_filter,
    )

    chunks = []
    for text, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        similarity = 1 - distance
        if similarity >= SIMILARITY_THRESHOLD:
            chunks.append({
                "text": text,
                "source": metadata["source"],
                "page": metadata["page"],
                "chunk_index": metadata["chunk_index"],
                "similarity": round(similarity, 4),
            })
    return chunks

def build_prompt(query: str, chunks: list[dict]) -> str:
    """
    Build the user-side prompt by injecting retrieved chunks with page labels.
    Chunks are pre-filtered — nothing below the similarity threshold reaches here.
    """
    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        filename = chunk['source'].split('/')[-1]
        context_blocks.append(
            f"--- chunk {i} (File: {filename}, Page {chunk['page']}) ---\n{chunk['text']}"
        )

    context = "\n\n".join(context_blocks)

    return f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"

def ask(query: str, source=None, provider: str = "ollama") -> dict:
    chunks = retrieve_chunks(embed_query(query, provider), source=source, provider=provider)
    if not chunks:
        return {
            "answer": "[NOT FOUND]",
            "chunks": [],
            "provider": provider,
            "token_usage": tokens.get_status(),
        }

    user_prompt = build_prompt(query, chunks)

    if provider == "gemini":
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        model = genai.GenerativeModel("gemini-3.5-flash", system_instruction=SYSTEM_PROMPT)
        
        # Count input tokens and check budget
        input_tokens = model.count_tokens(user_prompt).total_tokens
        tokens.warn_if_over_budget(input_tokens, model_name="gemini-3.5-flash", label="query")
        
        # Configure output token limit
        config = None
        if tokens.MAX_OUTPUT_TOKENS > 0:
            config = genai.types.GenerationConfig(max_output_tokens=tokens.MAX_OUTPUT_TOKENS)
        
        # Generate response
        response = model.generate_content(user_prompt, generation_config=config)
        answer = response.text.strip()
        
        # Extract output tokens from usage metadata
        usage = response.usage_metadata
        output_tokens = usage.candidates_token_count if usage else 0
        
        # Record token usage (headers not available in current SDK)
        tokens.record_query(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model_name="gemini-3.5-flash",
            remaining_requests=None,
            remaining_tokens=None,
        )
    else:
        if not HAS_OLLAMA:
            raise RuntimeError(
                "Ollama is not installed. Install it with: pip install ollama\n"
                "Or install the optional dependency: pip install -e .[ollama]\n"
                "Once Ollama installation is complete, run: ollama pull mxbai-embed-large\n"
                "ollama pull phi3:mini"
            )
        response = ollama.chat(
            model="phi3:mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        answer = response["message"]["content"].strip()

    if "[NOT FOUND]" in answer:
        return {
            "answer": "The PDF does not contain sufficient information to answer the question.",
            "chunks": [],
            "provider": provider,
            "token_usage": tokens.get_status(),
        }

    return {
        "answer": answer,
        "chunks": chunks,
        "provider": provider,
        "token_usage": tokens.get_status(),
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python query.py \"your question here\"")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    result = ask(query)

    print(f"\nAnswer:\n{result['answer']}\n")
    if result["chunks"]:
        print("Sources:")
        for c in result["chunks"]:
            print(f"  Page {c['page']}  (similarity: {c['similarity']})  {c['source']}")