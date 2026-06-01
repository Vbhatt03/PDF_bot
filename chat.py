#!/usr/bin/env python3
# chat.py
import argparse
import sys
import os
import shutil
import subprocess
from src.ingest import extract, chunk_pages, embed_chunks
from src.query import ask
from dotenv import load_dotenv
load_dotenv()

def ensure_ollama():
    """If ollama is not installed, find the bundled setup script and run it."""
    if shutil.which("ollama") is not None:
        return  # already installed, nothing to do

    # Resolve path to the bundled script
    if getattr(sys, "frozen", False):          # running as PyInstaller exe
        base = sys._MEIPASS
    else:                                       # running from source
        base = os.path.dirname(os.path.abspath(__file__))

    script = os.path.join(base, "setup_ollama.sh")

    if not os.path.isfile(script):
        print("Warning: setup_ollama.sh not bundled; cannot auto-install Ollama.")
        return

    print("Ollama not found. Running setup script (this may take a few minutes)...")
    os.chmod(script, 0o755)
    subprocess.run(["bash", script], check=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="One or more PDF/TXT/CSV files")
    parser.add_argument(
        "--provider", choices=["ollama", "gemini"], default="gemini",
        help="Backend to use (default: gemini)"
    )
    args = parser.parse_args()

    if args.provider == "ollama":
        ensure_ollama()

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