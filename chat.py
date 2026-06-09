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

# Import tool calling module
from src import tool_call as tool_call_module

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
    parser.add_argument(
        "--use-tools", action="store_true",
        help="Use Gemini tool calling for CSV files (skips embedding entirely). "
             "Only works with --provider=gemini and CSV files."
    )
    args = parser.parse_args()

    if args.provider == "ollama":
        check_ollama()

    # Validate --use-tools flag
    if args.use_tools and args.provider != "gemini":
        print("ERROR: --use-tools flag currently only works with --provider=gemini")
        sys.exit(1)

    # Filter CSV files for tool calling mode
    csv_files = [f for f in args.files if f.lower().endswith('.csv')]
    
    if args.use_tools:
        # Tool calling mode - skip embedding, use Gemini function calling
        print("=" * 60)
        print("TOOL CALLING MODE ENABLED")
        print("  - Skipping embedding entirely")
        print("  - Using Gemini function calling for CSV analysis")
        print("=" * 60 + "\n")
        
        if not csv_files:
            print("WARNING: --use-tools specified but no CSV files provided.")
            print("         Tool calling is only available for CSV files.\n")
        
        # Initialize tool caller with CSV files
        tool_caller = None
        if csv_files:
            print(f"Initialized tool caller for: {csv_files}\n")
            tool_caller = tool_call_module.CSVToolCaller(csv_files)
        
        print("Chat with your document. Type 'exit' to quit.")
        print("Note: Embedding disabled - full context sent with each query.\n")
        
        while True:
            query = input("You: ").strip()
            if query.lower() == "exit":
                print("Goodbye!")
                break
            if not query:
                continue
            if query == "NOBOT:Tokens?":
                print("\n" + tokens.format_summary() + "\n")
                continue
            
            if tool_caller:
                result = tool_caller.ask(query)
            else:
                # Fallback to regular ask if no CSV files
                print("No CSV files available for tool calling. Use regular embedding mode.")
                continue
            
            print(f"\nBot: {result['answer']}\n")
            
            if result.get("tool_results"):
                print("Tool calls made:")
                for tr in result["tool_results"]:
                    print(f"  - {tr['tool']}: {tr['arguments']}")
                    # Show first 200 chars of result
                    result_preview = tr['result'][:200] + "..." if len(tr['result']) > 200 else tr['result']
                    print(f"    Result: {result_preview}")
                print()
            
            if "token_usage" in result:
                usage = result["token_usage"]
                print(f"[Tokens] Input: {usage.get('prompt_token_count', 'N/A')} | "
                      f"Output: {usage.get('candidates_token_count', 'N/A')} | "
                      f"Total: {usage.get('total_token_count', 'N/A')}")
                status = tokens.get_status()
                if status.get("last_remaining_requests") is not None:
                    print(f"[Rate limits] Remaining requests: {status['last_remaining_requests']} | "
                        f"Remaining tokens: {status['last_remaining_tokens']}")

            if "session_token_usage" in result:
                print(f"[Session Tokens] Total used so far: {result['session_token_usage']}")

            
            print(f"[Answered by: {result['provider'].capitalize()}]")
            
    else:
        # Regular embedding mode (existing behavior)
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