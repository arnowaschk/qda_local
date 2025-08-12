#!/usr/bin/env python3
"""
Cache verification script to check the status of cached models.
This script can be run to verify that models are properly cached.
"""

import os
import pathlib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_cache_status():
    """Check the status of all cache directories and models"""
    cache_dir = pathlib.Path("/app/cache")
    
    if not cache_dir.exists():
        logger.error("Cache directory does not exist: /app/cache")
        return False
    
    logger.info("Cache directory exists: /app/cache")
    
    # Check each cache subdirectory
    cache_dirs = {
        "transformers": cache_dir / "transformers",
        "spacy": cache_dir / "spacy", 
        "sentence_transformers": cache_dir / "sentence_transformers",
        "torch": cache_dir / "torch"
    }
    
    for name, cache_path in cache_dirs.items():
        if cache_path.exists():
            logger.info(f"✓ {name} cache directory exists: {cache_path}")
            
            # Check if it contains models
            if name == "transformers":
                check_transformers_cache(cache_path)
            elif name == "spacy":
                check_spacy_cache(cache_path)
            elif name == "sentence_transformers":
                check_sentence_transformers_cache(cache_path)
            elif name == "torch":
                check_torch_cache(cache_path)
        else:
            logger.warning(f"✗ {name} cache directory missing: {cache_path}")
    
    # Check environment variables
    logger.info("\nEnvironment variables:")
    env_vars = [
        "TRANSFORMERS_CACHE", "HF_HOME", "SPACY_DATA", 
        "SENTENCE_TRANSFORMERS_HOME", "TORCH_HOME"
    ]
    
    for var in env_vars:
        value = os.environ.get(var, "NOT SET")
        logger.info(f"  {var}: {value}")
    
    return True

def check_transformers_cache(cache_path):
    """Check transformers cache for specific models"""
    logger.info(f"  Checking transformers cache: {cache_path}")
    
    # Look for the German sentiment model
    sentiment_model_path = cache_path / "models--oliverguhr--german-sentiment-bert"
    if sentiment_model_path.exists():
        logger.info(f"    ✓ German sentiment model found: {sentiment_model_path}")
        
        # Check for model files
        model_files = list(sentiment_model_path.rglob("*.bin"))
        tokenizer_files = list(sentiment_model_path.rglob("*.json"))
        
        if model_files:
            logger.info(f"      Model files: {len(model_files)} found")
        if tokenizer_files:
            logger.info(f"      Tokenizer files: {len(tokenizer_files)} found")
    else:
        logger.warning(f"    ✗ German sentiment model not found")
    
    # Check total size
    total_size = sum(f.stat().st_size for f in cache_path.rglob('*') if f.is_file())
    logger.info(f"    Total cache size: {total_size / (1024*1024):.1f} MB")

def check_spacy_cache(cache_path):
    """Check spaCy cache for models"""
    logger.info(f"  Checking spaCy cache: {cache_path}")
    
    # Look for the German model
    german_model_path = cache_path / "de_core_news_lg"
    if german_model_path.exists():
        logger.info(f"    ✓ German NER model found: {german_model_path}")
        
        # Check for model files
        model_files = list(german_model_path.rglob("*.bin"))
        if model_files:
            logger.info(f"      Model files: {len(model_files)} found")
    else:
        logger.warning(f"    ✗ German NER model not found")
    
    # Check total size
    total_size = sum(f.stat().st_size for f in cache_path.rglob('*') if f.is_file())
    logger.info(f"    Total cache size: {total_size / (1024*1024):.1f} MB")

def check_sentence_transformers_cache(cache_path):
    """Check sentence-transformers cache for models"""
    logger.info(f"  Checking sentence-transformers cache: {cache_path}")
    
    models_path = cache_path / "models"
    if models_path.exists():
        # Look for specific models
        models_to_check = ["BAAI_bge-m3", "sentence-transformers_paraphrase-multilingual-mpnet-base-v2"]
        
        for model_name in models_to_check:
            model_path = models_path / model_name
            if model_path.exists():
                logger.info(f"    ✓ Model found: {model_name}")
                
                # Check for model files
                model_files = list(model_path.rglob("*.bin"))
                if model_files:
                    logger.info(f"      Model files: {len(model_files)} found")
            else:
                logger.warning(f"    ✗ Model not found: {model_name}")
    else:
        logger.warning(f"    ✗ Models directory not found")
    
    # Check total size
    total_size = sum(f.stat().st_size for f in cache_path.rglob('*') if f.is_file())
    logger.info(f"    Total cache size: {total_size / (1024*1024):.1f} MB")

def check_torch_cache(cache_path):
    """Check PyTorch cache for models"""
    logger.info(f"  Checking PyTorch cache: {cache_path}")
    
    # Check total size
    total_size = sum(f.stat().st_size for f in cache_path.rglob('*') if f.is_file())
    logger.info(f"    Total cache size: {total_size / (1024*1024):.1f} MB")

def main():
    """Main function to check cache status"""
    logger.info("Checking cache status...")
    
    try:
        success = check_cache_status()
        if success:
            logger.info("Cache status check completed")
        else:
            logger.error("Cache status check failed")
            
    except Exception as e:
        logger.error(f"Cache status check failed with exception: {e}")

if __name__ == "__main__":
    main() 