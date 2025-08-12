# Global Model Caching System

This document explains how the global model caching system works in the QDA Worker to ensure fast model loading and persistence across container restarts.

## Overview

The caching system uses **global cache directories** that are shared across all your projects:

- Models are downloaded once and stored in your system's global cache
- Models are shared between different projects and containers
- No need to re-download models for different projects
- Standard behavior that most developers expect

## Global Cache Locations

The system uses these standard global cache directories:

- **Transformers**: `~/.cache/huggingface/` (Linux) or `%USERPROFILE%\.cache\huggingface\` (Windows)
- **spaCy**: `~/.local/share/spacy/` (Linux) or `%APPDATA%\spacy\` (Windows)  
- **PyTorch**: `~/.cache/torch/` (Linux) or `%USERPROFILE%\.cache\torch\` (Windows)
- **Sentence Transformers**: `~/.cache/torch/sentence_transformers/` (Linux)

## How It Works

### 1. Container Build
During the Docker build process:
- Global cache directories are created in the container
- spaCy German model is downloaded to the global cache
- Environment variables are set to point to global cache locations

### 2. Runtime
During container runtime:
- Global cache directories are mounted from your host system
- Models are loaded from your global cache
- If not found, models are downloaded and cached globally for future use
- Cache persists across all projects and containers

### 3. Persistence
The global cache is mounted as volumes in `docker-compose.yml`:
```yaml
volumes:
  - ~/.cache/huggingface:/root/.cache/huggingface
  - ~/.local/share/spacy:/root/.local/share/spacy
  - ~/.cache/torch:/root/.cache/torch
```

This ensures the cache persists across all your projects and containers.

## Environment Variables

The following environment variables are set to use global cache locations:

```bash
TRANSFORMERS_CACHE=/root/.cache/huggingface
HF_HOME=/root/.cache/huggingface
SPACY_DATA=/root/.local/share/spacy
TORCH_HOME=/root/.cache/torch
```

## Cached Models

The following models are pre-cached during build:

### spaCy
- `de_core_news_lg` - German language model for NER

### Transformers (downloaded on first use)
- `oliverguhr/german-sentiment-bert` - German sentiment analysis

### Sentence Transformers (downloaded on first use)
- `BAAI/bge-m3` - Multilingual embeddings
- `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` - Multilingual paraphrasing

## Benefits of Global Caching

- **Shared Across Projects**: Models downloaded once are available to all projects
- **No Redownloads**: Avoid downloading the same models multiple times
- **Standard Behavior**: Uses the same cache locations as your local Python environment
- **Easier Maintenance**: No custom cache management needed
- **Space Efficient**: Models are stored once on your system

## Monitoring Cache Status

### Check Global Cache at Runtime
You can check global cache status by running:
```bash
docker exec qda_worker python check_global_cache.py
```

### Manual Cache Verification
To manually verify cache contents on your host system:
```bash
# Check transformers cache
ls -la ~/.cache/huggingface/models/

# Check spaCy cache
ls -la ~/.local/share/spacy/

# Check PyTorch cache
ls -la ~/.cache/torch/
```

### Check from Host
You can also check the cache directly from your host system:
```bash
# List all cached models
find ~/.cache/huggingface/models -type d -maxdepth 1
find ~/.local/share/spacy -type d -maxdepth 1
```

## Using Global Cache in Other Projects

Since the cache is global, you can use it in other projects:

### Python Scripts
```python
# Models will automatically use the global cache
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("oliverguhr/german-sentiment-bert")
```

### Other Docker Containers
Mount the same cache directories:
```yaml
volumes:
  - ~/.cache/huggingface:/root/.cache/huggingface
  - ~/.local/share/spacy:/root/.local/share/spacy
  - ~/.cache/torch:/root/.cache/torch
```

### Local Development
Your local Python environment will use the same cache directories automatically.

## Troubleshooting

### Models Not Cached
If models are not being cached:
1. Check if the global cache directories exist on your host
2. Verify the volume mounts in docker-compose.yml
3. Check container logs for any download errors
4. Ensure sufficient disk space for model downloads

### Cache Permission Issues
If you encounter permission issues:
1. Check ownership of cache directories: `ls -la ~/.cache/`
2. Ensure your user owns the cache directories
3. Fix permissions: `chown -R $USER:$USER ~/.cache/`

### Cache Corruption
If cache becomes corrupted:
1. Remove the specific model from the global cache
2. Restart the container to re-download the model
3. Check network connectivity for model downloads

## Cache Size

Typical cache sizes for the models used:
- spaCy German model: ~500MB
- Transformers sentiment model: ~500MB  
- Sentence transformer models: ~1-2GB
- Total cache size: ~2-3GB

The cache is shared across all projects, so this space is used efficiently.

## Migration from Project-Specific Cache

If you were using the old project-specific cache system:
1. The new system automatically uses global caches
2. Your existing global cache will be used if available
3. No need to copy or migrate cache files
4. Models will be downloaded to global cache on first use 