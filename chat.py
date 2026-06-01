#!/usr/bin/env python3
# chat.py
import argparse
import sys
import os
import shutil
from src.ingest import extract, chunk_pages, embed_chunks
from src.query import ask
from dotenv import load_dotenv
load_dotenv()

try:
    import ollama as _ollama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False

def check_ollama():
    """Check if ollama is installed (both binary and Python module). If not, show error message and instructions."""
    ollama_bin = shutil.which("ollama") is not None
    
    if not ollama_bin or not HAS_OLLAMA:
        print("ERROR: Ollama is not installed or not available.")
        print("\nTo install Ollama, please run the setup script:")
        print("  bash setup_ollama.sh")
        print("\nThis script will download and install Ollama with the required models.")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="One or more PDF/TXT/CSV files")
    parser.add_argument(
        "--provider", choices=["ollama", "gemini"], default="gemini",
        help="Backend to use (default: gemini)"
    )
    args = parser.parse_args()

    if args.provider == "ollama":
        check_ollama()

    print(f"Using provider: {args.provider}")
    sources = []
    for f in args.files:
        print(f"Ingesting {f}...")
        pages = extract(f)
        print(f"  Extracted {len(pages)} page(s).")
        chunks = chunk_pages(pages)
        print(f"  Produced {len(chunks)} chunk(s).")
        embed_chunks(chunks, provider=args.provider)
        sources.append(f)
    print("Ingestion complete.\n")

    print("Chat with your document. Type 'exit' to quit.\n")
    while True:
        query = input("You: ").strip()
        if query.lower() == "exit":
            print("Goodbye!")
            break
        if not query:
            continue
        result = ask(query, source=sources, provider=args.provider)
        print(f"\nBot: {result['answer']}\n")
        if result["chunks"]:
            print("Sources:")
            for c in result["chunks"]:
                print(f"  Page {c['page']}  (similarity: {c['similarity']})  {c['source']}")
            print()
        print(f"[Answered by: {result['provider'].capitalize()}]")


if __name__ == "__main__":
    main()