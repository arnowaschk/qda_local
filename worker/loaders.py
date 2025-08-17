import pathlib, os, yaml, sys
import traceback
from util import logger
from dynamics import (
    GERMAN_STOPWORDS
)
import spacy
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline as hf_pipeline
from sentence_transformers import SentenceTransformer
import torch
import chardet  
import pandas as pd
from typing import Optional
from cache import (
    load_persistent_cache,
    save_persistent_cache,
    get_cached_result,
    process_texts_in_batch,
    process_texts_batch,
    clear_ner_cache,
)

def load_policies(cfg_dir: pathlib.Path) -> dict:
    """Load policy definitions from a YAML configuration file.
    
    Args:
        cfg_dir: Directory containing the policies.yaml file
        
    Returns:
        Dictionary containing policy definitions or empty dict if loading fails
    """
    logger.info(f"Loading policies from: {cfg_dir}")
    f = cfg_dir / "policies.yaml"
    if f.exists():
        try:
            policies = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
            logger.info(f"Successfully loaded policies: {len(policies)} policy items")
            return policies
        except Exception as e:
            logger.error(f"Failed to load policies: {e}")
            return {}
    else:
        logger.info("No policies.yaml found, using default configuration")
    return {}

def load_ner(HAS_SPACY: bool) -> Optional[spacy.Language]:
    """Load and configure the German language model with enhanced processing.
    
    Args:
        HAS_SPACY: Boolean indicating if spaCy is available
        
    Returns:
        Configured spaCy language model for German or None if loading fails
        
    Note:
        - Adds custom German stopwords
        - Configures pipeline with optimized settings for German
        - Attempts to download model if not found in cache
    """
    if not HAS_SPACY:
        logger.warning("spaCy not available, NER functionality disabled")
        return None
        
    try:
        logger.info("Loading and configuring German language model...")
        logger.info(f"Using spaCy cache directory: {os.environ.get('SPACY_DATA')}")
        
        # Try to load from cache first
        try:
            # Load the model with optimized settings
            nlp = spacy.load(
                "de_core_news_lg",
                disable=["parser", "textcat"]  # Disable unused components for efficiency
            )
            
            # Add custom German stopwords
            custom_stopwords = {
                'eigentlich', 'irgendwie', 'halt', 'eben', 'mal', 'ja',
                'schon', 'doch', 'auch', 'nur', 'denn', 'etwa', 'etwas',
                'gar', 'gern', 'irgendwas', 'irgendwo', 'man', 'na',
                'nämlich', 'nun', 'sehr', 'vielleicht', 'viel', 'vom',
                'überhaupt', 'weiter', 'wieder', 'wirklich', 'zu', 'zurück'
            }
            
            # Add custom stopwords to the default set
            for word in custom_stopwords:
                nlp.Defaults.stop_words.add(word)
                nlp.vocab[word].is_stop = True
            
            # Configure the pipeline for better German processing
            if 'sentencizer' not in nlp.pipe_names:
                nlp.add_pipe('sentencizer')
            
            logger.info("German language model loaded and configured successfully")
            return nlp
            
        except OSError as e:
            logger.info(f"Model not found in cache, downloading... {e}")
            # Download the model
            try:
                import subprocess
                result = subprocess.run(
                    [sys.executable, "-m", "spacy", "download", "de_core_news_lg"],
                    capture_output=True, 
                    text=True
                )
                if result.returncode == 0:
                    nlp = spacy.load("de_core_news_lg", disable=["parser", "textcat"])
                    logger.info("German language model downloaded and loaded successfully")
                    return nlp
                else:
                    logger.error(f"Failed to download spaCy model: {result.stderr}")
                    return None
            except Exception as e:
                logger.error(f"Error downloading spaCy model: {e}")
                return None
                
    except Exception as e:
        logger.error(f"Failed to load German language model: {e}")
        return None

def load_sentiment(HAS_TRANSFORMERS: bool) -> Optional[hf_pipeline]:
    """Load and configure the German sentiment analysis model.
    
    Args:
        HAS_TRANSFORMERS: Boolean indicating if transformers library is available
        
    Returns:
        Configured Hugging Face pipeline for sentiment analysis or None if loading fails
        
    Note:
        - Uses 'oliverguhr/german-sentiment-bert' model
        - Configures for GPU if available
        - Sets up tokenizer with appropriate settings for German text
    """
    if not HAS_TRANSFORMERS:
        logger.warning("transformers not available, sentiment analysis disabled")
        return None
    
    try:
        logger.info("Loading German sentiment analysis model...")
        
        # Configuration for sentiment analysis
        model_config = {
            "model_name": "oliverguhr/german-sentiment-bert",
            "max_length": 512,
            "truncation": True,
            "padding": "max_length"
        }
        
        cache_dir = pathlib.Path(os.environ.get('SENT_HOME', '~/.cache/huggingface'))
        logger.info(f"Using cache directory: {cache_dir}")
        os.system("ls -l " + str(cache_dir))
        # Check if model exists in cache
        model_path = cache_dir / f"models--{model_config['model_name'].replace('/', '--')}"
        logger.info(f"Model path: {model_path}")
        
        if not model_path.exists():
            logger.info("Model not found in cache, will download on first use")
        
        try:
            # Load tokenizer with custom settings
            tokenizer = AutoTokenizer.from_pretrained(
                model_config["model_name"],
                cache_dir=str(cache_dir),
                use_fast=True,
                local_files_only=False
            )
            
            # Load model with custom settings
            model = AutoModelForSequenceClassification.from_pretrained(
                model_config["model_name"],
                cache_dir=str(cache_dir),
                local_files_only=False,
                num_labels=3  # positive, negative, neutral
            )
            
            # Create pipeline with custom settings
            pipeline = hf_pipeline(
                "sentiment-analysis",
                model=model,
                tokenizer=tokenizer,
                device=0 if torch.cuda.is_available() else -1,  # Use GPU if available
                framework="pt",
                return_all_scores=False,
                truncation=True,
                max_length=model_config["max_length"]
            )
            
            logger.info("German sentiment model loaded and configured successfully")
            return pipeline
            
        except Exception as e:
            logger.error(f"Error initializing sentiment model: {e}")
            logger.error(traceback.format_exc())
            return None
            
    except Exception as e:
        logger.error(f"Failed to load sentiment analysis: {e}")
        logger.error(traceback.format_exc())
        return None

def load_embedder(HAS_SENTENCE_TRANSFORMERS: bool) -> Optional[SentenceTransformer]:
    """Load a sentence transformer model for generating text embeddings.
    
    Args:
        HAS_SENTENCE_TRANSFORMERS: Boolean indicating if sentence-transformers is available
        
    Returns:
        Configured SentenceTransformer model or None if loading fails
        
    Note:
        - Tries multiple model variants in order of preference
        - Uses persistent caching for model weights
        - Falls back to CPU if CUDA is not available
    """
    if not HAS_SENTENCE_TRANSFORMERS:
        logger.warning("sentence-transformers not available, embeddings disabled")
        return None
    
    # Ensure persistent cache is loaded
    if not hasattr(load_embedder, '_cache_loaded'):
        load_persistent_cache()
        load_embedder._cache_loaded = True
    
    # Get cache directory with fallback
    cache_dir = os.environ.get('SENTENCE_TRANSFORMERS_HOME')
    if not cache_dir:
        # Fallback to HF_HOME if SENTENCE_TRANSFORMERS_HOME is not set
        cache_dir = os.environ.get('HF_HOME', './cache')
        logger.warning(f"SENTENCE_TRANSFORMERS_HOME not set, using fallback: {cache_dir}")
    
    logger.info(f"Using sentence-transformers cache directory: {cache_dir}")
    
    for name in ["BAAI/bge-m3", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"]:
        try:
            logger.info(f"Loading embedding model: {name}")
            
            # Check if model exists in cache
            cache_path = pathlib.Path(cache_dir) / f"models--{name.split('/')[0]}--{name.split('/')[1]}"
            if cache_path.exists():
                logger.info(f"Model found in cache: {cache_path}")
            else:
                logger.info(f"Model not found in cache {cache_path}, will download on first use")
            
            embedder = SentenceTransformer(name, cache_folder=str(cache_dir))
            logger.info(f"Embedding model {name} loaded successfully")
            return embedder
        except Exception as e:
            logger.warning(f"Failed to load embedding model {name}: {e}")
            continue
    logger.error("All embedding models failed to load")
    return None

def load_csv_with_fallback(file_path: str) -> pd.DataFrame:
    """Load a CSV file with robust encoding detection and fallback mechanisms.
    
    Args:
        file_path: Path to the CSV file to load
        
    Returns:
        DataFrame containing the loaded CSV data
        
    Raises:
        ValueError: If all encoding attempts fail
        
    Note:
        - Attempts to automatically detect file encoding
        - Tries multiple common encodings if detection fails
        - Cleans the resulting DataFrame by removing empty rows/columns
    """
    try:
        # First, try to detect the file encoding
        with open(file_path, 'rb') as f:
            rawdata = f.read(10000)  # Read first 10KB for detection
            result = chardet.detect(rawdata)
            encoding = result['encoding']
            
        # Fallback to utf-8 if detection fails
        if not encoding:
            encoding = 'utf-8'
            
        logger.info(f"Detected encoding: {encoding} (confidence: {result.get('confidence', 0) * 100:.1f}%)")
        
        # Try reading with detected encoding
        df = pd.read_csv(
            file_path,
        ).fillna("")
        
        # Clean up the dataframe
        df = df.dropna(how='all')  # Remove completely empty rows
        df = df.loc[:, (df != '').any(axis=0)]  # Remove completely empty columns
        df = df.apply(lambda x: x.str.strip() if x.dtype == 'object' else x)  # Strip whitespace
        
        logger.info(f"Successfully loaded CSV with {len(df)} rows and {len(df.columns)} columns")
        logger.info(str(df.head()))
        print (df.head())
        return df
        
    except Exception as e:
        logger.info(f"Encoding test failed with {e}")
        # If first attempt fails, try with common encodings
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        
        for enc in encodings:
            try:
                logger.info(f"Trying fallback encoding: {enc}")
                df = pd.read_csv(
                    file_path,
                    encoding=enc,
                    dtype=str,
                    on_bad_lines='warn',
                    engine='python'
                )
                logger.info(f"Successfully loaded with {enc}")
                return df
            except Exception:
                continue
                
        # If all attempts fail, raise the original error
        logger.error(f"Failed to load CSV after trying {len(encodings) + 1} encodings")
        raise
                
    
    raise ValueError("Failed to load CSV file after trying multiple encodings")

