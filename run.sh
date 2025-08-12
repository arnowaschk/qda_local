#!/bin/bash
mkdir -p output

# Resolve symlinks to actual paths and export them
export HUGGINGFACE_CACHE=$(readlink -f ~/.cache/huggingface)
export HF_HOME=/root/.cache/huggingface
export SPACY_CACHE=$(readlink -f ~/.local/share/spacy)
export TORCH_CACHE=$(readlink -f ~/.cache/torch)
export SENTENCE_TRANSFORMERS_HOME=$(readlink -f ~/.cache/sentence_transformers)

docker compose down -v
docker compose build

echo "Using cache paths:"
echo "  HuggingFace: $HUGGINGFACE_CACHE"
echo "  spaCy: $SPACY_CACHE"
echo "  Torch: $TORCH_CACHE"

# Run docker-compose with resolved paths
echo '....starting'
docker compose up &

echo '.... wait'
sleep 6
docker ps -a
echo curl -X POST "http://localhost:8009/analyze?input_path=/app/data/$1.csv&out_dir=/app/out/$1"
