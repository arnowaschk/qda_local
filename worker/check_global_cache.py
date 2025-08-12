#!/usr/bin/env python3
"""
Global cache verification script to check the status of cached models.
This script checks the global cache directories used by transformers, spaCy, etc.
"""

import os
import pathlib
import logging
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_cache_paths():
    """Get the cache paths from environment variables or defaults"""
    cache_paths = {
        "transformers": os.environ.get('TRANSFORMERS_CACHE', os.path.expanduser('~/.cache/huggingface')),
        "spacy": os.environ.get('SPACY_DATA', os.path.expanduser('~/.local/share/spacy')),
        "torch": os.environ.get('TORCH_HOME', os.path.expanduser('~/.cache/torch')),
        "sentence_transformers": os.path.expanduser('~/.cache/torch/sentence_transformers')
    }
    return cache_paths

def check_cache_directory(name, path):
    """Check a specific cache directory"""
    cache_path = pathlib.Path(path)
    
    if not cache_path.exists():
        logger.warning(f"✗ {name} cache directory does not exist: {path}")
        return False
    
    logger.info(f"✓ {name} cache directory exists: {path}")
    
    # Check if it contains models
    if name == "transformers":
        check_transformers_cache(cache_path)
    elif name == "spacy":
        check_spacy_cache(cache_path)
    elif name == "torch":
        check_torch_cache(cache_path)
    elif name == "sentence_transformers":
        check_sentence_transformers_cache(cache_path)
    
    return True

def check_transformers_cache(cache_path):
    """Check transformers cache for models"""
    logger.info(f"  Checking transformers cache: {cache_path}")
    
    # Look for any models
    models_dir = cache_path / "models"
    if models_dir.exists():
        model_dirs = [d for d in models_dir.iterdir() if d.is_dir()]
        logger.info(f"    Found {len(model_dirs)} model directories")
        
        for model_dir in model_dirs[:5]:  # Show first 5
            logger.info(f"      - {model_dir.name}")
        
        if len(model_dirs) > 5:
            logger.info(f"      ... and {len(model_dirs) - 5} more")
    else:
        logger.info("    No models directory found")
    
    # Check total size
    total_size = sum(f.stat().st_size for f in cache_path.rglob('*') if f.is_file())
    logger.info(f"    Total cache size: {total_size / (1024*1024):.1f} MB")

def check_spacy_cache(cache_path):
    """Check spaCy cache for models"""
    logger.info(f"  Checking spaCy cache: {cache_path}")
    
    # Look for models
    model_dirs = [d for d in cache_path.iterdir() if d.is_dir()]
    logger.info(f"    Found {len(model_dirs)} model directories")
    
    for model_dir in model_dirs:
        logger.info(f"      - {model_dir.name}")
    
    # Check total size
    total_size = sum(f.stat().st_size for f in cache_path.rglob('*') if f.is_file())
    logger.info(f"    Total cache size: {total_size / (1024*1024):.1f} MB")

def check_torch_cache(cache_path):
    """Check PyTorch cache for models"""
    logger.info(f"  Checking PyTorch cache: {cache_path}")
    
    # Check total size
    total_size = sum(f.stat().st_size for f in cache_path.rglob('*') if f.is_file())
    logger.info(f"    Total cache size: {total_size / (1024*1024):.1f} MB")

def check_sentence_transformers_cache(cache_path):
    """Check sentence-transformers cache for models"""
    logger.info(f"  Checking sentence-transformers cache: {cache_path}")
    
    if not cache_path.exists():
        logger.info("    Cache directory does not exist")
        return
    
    # Look for models
    models_dir = cache_path / "models"
    if models_dir.exists():
        model_dirs = [d for d in models_dir.iterdir() if d.is_dir()]
        logger.info(f"    Found {len(model_dirs)} model directories")
        
        for model_dir in model_dirs:
            logger.info(f"      - {model_dir.name}")
    else:
        logger.info("    No models directory found")
    
    # Check total size
    total_size = sum(f.stat().st_size for f in cache_path.rglob('*') if f.is_file())
    logger.info(f"    Total cache size: {total_size / (1024*1024):.1f} MB")

def check_environment_variables():
    """Check environment variables"""
    logger.info("\nEnvironment variables:")
    env_vars = [
        "TRANSFORMERS_CACHE", "HF_HOME", "SPACY_DATA", 
        "SENTENCE_TRANSFORMERS_HOME", "TORCH_HOME"
    ]
    
    for var in env_vars:
        value = os.environ.get(var, "NOT SET (using default)")
        logger.info(f"  {var}: {value}")

def check_host_cache():
    """Check if we can access host cache directories"""
    logger.info("\nHost cache access:")
    
    try:
        # Try to list host cache directories
        host_cache = os.path.expanduser('~/.cache')
        if os.path.exists(host_cache):
            logger.info(f"✓ Host cache directory exists: {host_cache}")
            
            # Check what's in the host cache
            host_contents = os.listdir(host_cache)
            logger.info(f"  Host cache contents: {', '.join(host_contents)}")
        else:
            logger.warning(f"✗ Host cache directory does not exist: {host_cache}")
    except Exception as e:
        logger.error(f"Failed to check host cache: {e}")

def main():
    """Main function to check global cache status"""
    logger.info("Checking global cache status...")
    
    try:
        # Get cache paths
        cache_paths = get_cache_paths()
        
        # Check each cache directory
        for name, path in cache_paths.items():
            check_cache_directory(name, path)
        
        # Check environment variables
        check_environment_variables()
        
        # Check host cache access
        check_host_cache()
        
        logger.info("\nGlobal cache status check completed")
        
    except Exception as e:
        logger.error(f"Global cache status check failed with exception: {e}")

if __name__ == "__main__":
    main() 