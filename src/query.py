# query.py
from .ingest import get_collection
import os
import google.generativeai as genai

try:
    import ollama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False  

SIMILARITY_THRESHOLD = 0.4

SYSTEM_PROMPT = """You are a precise document assistant. Follow these rules without exception:
1. Answer ONLY using information present in the CONTEXT provided below.
2. If the answer is not in the context, respond with exactly: [NOT FOUND]
3. When you use information from the context, cite the page number inline, e.g. (Page 3).
4. Do not infer, extrapolate, or use any external knowledge."""



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
        if source or similarity >= SIMILARITY_THRESHOLD:
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
        context_blocks.append(
            f"--- chunk {i} (Page {chunk['page']}) ---\n{chunk['text']}"
        )

    context = "\n\n".join(context_blocks)

    return f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"

def ask(query: str, source=None, provider: str = "ollama") -> dict:
    chunks = retrieve_chunks(embed_query(query, provider), source=source, provider=provider)
    if not chunks:
        return {"answer": "[NOT FOUND]", "chunks": [], "provider": provider}

    user_prompt = build_prompt(query, chunks)

    if provider == "gemini":                                 # ADD branch
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        model = genai.GenerativeModel("gemini-3.5-flash", system_instruction=SYSTEM_PROMPT)
        response = model.generate_content(user_prompt)
        answer = response.text.strip()
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
        return {"answer": "The PDF does not contain sufficient information to answer the question.", "chunks": [], "provider": provider}

    return {"answer": answer, "chunks": chunks, "provider": provider}


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