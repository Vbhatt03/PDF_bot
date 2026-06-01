#!/usr/bin/env bash
set -e

if ! command -v ollama &>/dev/null; then
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "Ollama already installed, skipping."
fi

echo "Pulling embedding model (~670 MB)..."
ollama pull mxbai-embed-large

echo "Pulling chat model (~2.3 GB)..."
ollama pull phi3:mini

echo "Starting Ollama server in background..."
ollama serve &>/dev/null &
disown

echo "Done. You can now run: ./chatbot <file.pdf/.txt> --provider ollama"