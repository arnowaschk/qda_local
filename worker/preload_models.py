#!/usr/bin/env python3
"""
Pre-load models script to ensure they are cached during container build.
This script downloads the required models and saves them to the cache directories.
"""

import os
import sys
import logging
import pathlib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_cache_directories():
    """Set up cache directories and environment variables"""
    cache_dir = pathlib.Path("/app/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    transformers_cache = cache_dir / "transformers"
    spacy_cache = cache_dir / "spacy"
    sentence_transformers_cache = cache_dir / "sentence_transformers"
    torch_cache = cache_dir / "torch"
    
    for cache_path in [transformers_cache, spacy_cache, sentence_transformers_cache, torch_cache]:
        cache_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created cache directory: {cache_path}")
    
    # Set environment variables
    os.environ["TRANSFORMERS_CACHE"] = str(transformers_cache)
    #os.environ["HF_HOME"] = str(transformers_cache)
    os.environ["SPACY_DATA"] = str(spacy_cache)
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(sentence_transformers_cache)
    os.environ["TORCH_HOME"] = str(torch_cache)
    
    logger.info("Cache environment variables configured")
    return transformers_cache, spacy_cache, sentence_transformers_cache

def preload_spacy_models(spacy_cache):
    """Pre-load spaCy models"""
    try:
        import spacy
        logger.info("Pre-loading spaCy German model...")
        
        # Download the model
        import subprocess
        result = subprocess.run([
            sys.executable, "-m", "spacy", "download", "de_core_news_lg"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("spaCy German model downloaded successfully")
        else:
            logger.error(f"Failed to download spaCy model: {result.stderr}")
            
    except ImportError:
        logger.warning("spaCy not available, skipping spaCy model preload")

def preload_transformers_models(transformers_cache):
    """Pre-load transformers models"""
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        model_name = "oliverguhr/german-sentiment-bert"
        logger.info(f"Pre-loading transformers model: {model_name}")
        
        # This will download and cache the model
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(transformers_cache))
        model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=str(transformers_cache))
        
        logger.info("Transformers model pre-loaded successfully")
        
    except ImportError:
        logger.warning("Transformers not available, skipping transformers model preload")
    except Exception as e:
        logger.error(f"Failed to pre-load transformers model: {e}")

def preload_sentence_transformers_models(sentence_transformers_cache):
    """Pre-load sentence-transformers models"""
    try:
        from sentence_transformers import SentenceTransformer
        
        models = ["BAAI/bge-m3", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"]
        
        for model_name in models:
            try:
                logger.info(f"Pre-loading sentence-transformers model: {model_name}")
                
                # This will download and cache the model
                model = SentenceTransformer(model_name, cache_folder=str(sentence_transformers_cache))
                logger.info(f"Model {model_name} pre-loaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to pre-load model {model_name}: {e}")
                
    except ImportError:
        logger.warning("Sentence-transformers not available, skipping sentence-transformers model preload")

def main():
    """Main function to pre-load all models"""
    logger.info("Starting model pre-loading process...")
    
    try:
        # Set up cache directories
        transformers_cache, spacy_cache, sentence_transformers_cache = setup_cache_directories()
        
        # Pre-load models
        preload_spacy_models(spacy_cache)
        preload_transformers_models(transformers_cache)
        preload_sentence_transformers_models(sentence_transformers_cache)
        
        logger.info("Model pre-loading completed successfully")
        
    except Exception as e:
        logger.error(f"Model pre-loading failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 