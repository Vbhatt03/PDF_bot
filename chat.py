#!/usr/bin/env python3
# chat.py
import argparse
from src.ingest import extract, chunk_pages, embed_chunks
from src.query import ask
from dotenv import load_dotenv
load_dotenv()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="PDF or TXT file")
    parser.add_argument(
        "--provider", choices=["ollama", "gemini"], default="ollama",
        help="Backend to use (default: ollama)"
    )
    args = parser.parse_args()

    print(f"Using provider: {args.provider}")
    print(f"Ingesting {args.file}...")
    pages = extract(args.file)
    print(f"Extracted {len(pages)} page(s).")
    chunks = chunk_pages(pages)
    print(f"Produced {len(chunks)} chunk(s).")
    embed_chunks(chunks, provider=args.provider)
    print("Ingestion complete.\n")

    print("Chat with your document. Type 'exit' to quit.\n")
    while True:
        query = input("You: ").strip()
        if query.lower() == "exit":
            print("Goodbye!")
            break
        if not query:
            continue
        result = ask(query, source=args.file, provider=args.provider)
        print(f"\nBot: {result['answer']}\n")
        if result["chunks"]:
            print("Sources:")
            for c in result["chunks"]:
                print(f"  Page {c['page']}  (similarity: {c['similarity']})  {c['source']}")
            print()
        print(f"[Answered by: {result['provider'].capitalize()}]")


if __name__ == "__main__":
    main()