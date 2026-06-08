#!/usr/bin/env python3
# chat.py
import argparse
import sys
import os
import shutil
from src.ingest import extract, chunk_pages, embed_chunks
from src.query import ask
from dotenv import load_dotenv
from src import tokens
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
    parser.add_argument("files", nargs="+", help="One or more PDF/TXT/CSV/STL files")
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
        if args.provider == "gemini":
            status = tokens.get_status()
            print(f"  [Tokens] Estimated: {status['last_ingest_tokens']} | Session total: {status['total_tokens_used']}")
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
        if query == "NOBOT:Tokens?":
            if args.provider == "gemini":
                print("\n" + tokens.format_summary() + "\n")
            else:
                print("\n[Token tracking is only available for Gemini provider]\n")
            continue
        result = ask(query, source=sources, provider=args.provider)
        print(f"\nBot: {result['answer']}\n")
        if result["chunks"]:
            print("Sources:")
            for c in result["chunks"]:
                print(f"  Page {c['page']}  (similarity: {c['similarity']})  {c['source']}")
            print()
        if args.provider == "gemini" and "token_usage" in result:
            usage = result["token_usage"]
            if usage.get("last_input_tokens"):
                print(f"[Tokens] Input: {usage['last_input_tokens']} | Output: {usage['last_output_tokens']} | Session total: {usage['total_tokens_used']}")
                if usage.get("last_remaining_requests") is not None:
                    print(f"[Rate limits] Remaining requests: {usage['last_remaining_requests']} | Remaining tokens: {usage['last_remaining_tokens']}")    
        print(f"[Answered by: {result['provider'].capitalize()}]")


if __name__ == "__main__":
    main()