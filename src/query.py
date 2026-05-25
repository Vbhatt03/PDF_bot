# query.py
from .ingest import get_collection
import ollama

SIMILARITY_THRESHOLD = 0.4

SYSTEM_PROMPT = """You are a precise document assistant. Follow these rules without exception:
1. Answer ONLY using information present in the CONTEXT provided below.
2. If the answer is not in the context, respond with exactly: [NOT FOUND]
3. When you use information from the context, cite the page number inline, e.g. (Page 3).
4. Do not infer, extrapolate, or use any external knowledge."""



def embed_query(query: str) -> list[float]:
    response = ollama.embed(model="mxbai-embed-large", input=query)
    return response["embeddings"][0]

def retrieve_chunks(query_embedding: list[float], k: int = 6, source: str= None) -> list[dict]:
    """
    Query Chroma for the top-k most similar chunks.
    Converts Chroma's cosine distance to similarity (similarity = 1 - distance).
    Discards any chunk below SIMILARITY_THRESHOLD.
    Returns [] if nothing passes — caller should return [NOT FOUND] without calling the LLM.
    """
    collection = get_collection()
    where_filter = {"source": source} if source else None
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
        where = where_filter,
    )

    chunks = []
    for text, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        similarity = 1 - distance
        #print(f"DEBUG similarity={similarity:.4f}  chunk: {text[:60]!r}")   
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

def ask(query: str, source: str = None) -> dict:
    """
    Full query pipeline: embed → retrieve → prompt → generate.
    Returns a dict with 'answer' and 'chunks' (the sources used).
    """

    # Layer 1: similarity threshold — free, no LLM call
    chunks = retrieve_chunks(embed_query(query), source = source)
    if not chunks:
        return {"answer": "[NOT FOUND]", "chunks": []}

    # Build prompt and call LLM
    user_prompt = build_prompt(query, chunks)

    response = ollama.chat(
        model="phi3:mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    answer = response["message"]["content"].strip()

    # Layer 2: LLM's own declaration
    if "[NOT FOUND]" in answer:
        return {"answer": "The PDF does not contain sufficient information to answer the question.", "chunks": []}

    return {"answer": answer, "chunks": chunks}


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