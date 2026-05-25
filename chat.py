#!/usr/bin/env python3
# chat.py
import sys
from src.ingest import extract, chunk_pages, embed_chunks
from src.query import ask


def main():
    if len(sys.argv) < 2:
        print("Usage: python chat.py <file.pdf|file.txt>")
        sys.exit(1)

    file_path = sys.argv[1]

    # Run the ingest pipeline
    print(f"Ingesting {file_path}...")
    pages = extract(file_path)
    print(f"Extracted {len(pages)} page(s).")

    chunks = chunk_pages(pages)
    print(f"Produced {len(chunks)} chunk(s).")

    embed_chunks(chunks)
    print("✓ Ingestion complete.\n")

    # Interactive chat loop
    print("Chat with your document. Type 'exit' to quit.\n")
    while True:
        query = input("You: ").strip()
        if query.lower() == "exit":
            print("Goodbye!")
            break
        if not query:
            continue

        result = ask(query, source = file_path)
        print(f"\nBot: {result['answer']}\n")
        if result["chunks"]:
            print("Sources:")
            for c in result["chunks"]:
                print(f"  Page {c['page']}  (similarity: {c['similarity']})  {c['source']}")
            print()


if __name__ == "__main__":
    main()
