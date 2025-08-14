# All comments in English.
import argparse, json, os, pathlib, re, sys, traceback, datetime
import chardet,csv
import logging
from typing import List, Dict, Any, Tuple, Set
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import combinations
from wordcloud import WordCloud
from PIL import Image
import matplotlib.colors as mcolors
from typing import Dict, List, Any, Optional, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import yaml
import networkx as nx
import pickle
import hashlib
import signal
import torch

# Import the dynamics module for dynamic content generation
from dynamics import (
    hybrid_keyword_generation,
    generate_dynamic_policies,
    extract_document_themes,
    generate_comprehensive_stance_patterns as dynamics_generate_comprehensive_stance_patterns,
    analyze_stances_with_dynamic_patterns as dynamics_analyze_stances_with_dynamic_patterns,
    GERMAN_STOPWORDS
)

# Import the html module for HTML report generation
from html_report import generate_html_report

# Global timeout configuration (5 hours = 18000 seconds)
TIMEOUT_SECONDS = 18000

class TimeoutError(Exception):
    """Custom timeout exception"""
    pass

def timeout_handler(signum, frame):
    """Handle timeout signal"""
    raise TimeoutError(f"Pipeline timed out after {TIMEOUT_SECONDS} seconds ({TIMEOUT_SECONDS/3600:.1f} hours)")

def setup_timeout():
    """Setup timeout handler"""
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(TIMEOUT_SECONDS)
        logger.info(f"Timeout set to {TIMEOUT_SECONDS} seconds ({TIMEOUT_SECONDS/3600:.1f} hours)")
    except Exception as e:
        logger.warning(f"Could not set timeout signal: {e}")

def clear_timeout():
    """Clear timeout"""
    try:
        signal.alarm(0)
        logger.info("Timeout cleared")
    except Exception as e:
        logger.warning(f"Could not clear timeout: {e}")

# Simple direct logging to stderr for immediate output
import sys

class DirectLogger:
    def __init__(self, name):
        self.name = name
    
    def _get_caller_info(self):
        """Get filename and line number of the caller."""
        import inspect
        frame = inspect.currentframe()
        try:
            # Go back 3 frames to get the actual caller:
            # 1. _log
            # 2. debug/info/warning/error method
            # 3. The actual caller
            for _ in range(3):
                frame = frame.f_back
                if frame is None:
                    break
            
            if frame:
                return f"{frame.f_code.co_filename}:{frame.f_lineno}"
            return "unknown:0"
        finally:
            del frame
    
    def _log(self, level, msg, *args):
        if args:
            msg = msg % args
        caller = self._get_caller_info()
        now=datetime.datetime.now().strftime("%Y%m%d %H:%M:%S") 
        print(f"{now}|{level} - {self.name} - [{caller}] - {msg}", 
              file=sys.stderr, flush=True)
    
    def debug(self, msg, *args):
        self._log("DEBUG", msg, *args)
    
    def info(self, msg, *args):
        self._log("INFO ", msg, *args)
    
    def warning(self, msg, *args):
        self._log("WARN ", msg, *args)
    
    def error(self, msg, *args):
        self._log("ERROR", msg, *args)
    
    def exception(self, msg, *args, exc_info=None):
        self._log("EXCEPT", msg, *args)
        if exc_info is None:
            exc_info = sys.exc_info()
        if any(exc_info):  # If there's an exception to log
            import traceback
            tb_lines = traceback.format_exception(*exc_info)
            for line in tb_lines:
                for subline in line.rstrip().split('\n'):
                    if subline.strip():
                        print(f"TRACEBACK - {self.name} - {subline}", 
                              file=sys.stderr, flush=True)
    
    def critical(self, msg, *args):
        self._log("CRIT ", msg, *args)
        sys.exit(1)

# Use our direct logger
logger = DirectLogger(__name__)

# Log the cache directories being used (these are set by environment variables in Dockerfile)
logger.info("Using global cache directories:")
logger.info(f"  HF_HOME: {os.environ.get('HF_HOME', 'default')}")
logger.info(f"  SPACY_DATA: {os.environ.get('SPACY_DATA', 'default')}")
logger.info(f"  TORCH_HOME: {os.environ.get('TORCH_HOME', 'default')}")

import spacy
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline as hf_pipeline

# Set flags for backward compatibility
HAS_SPACY = True
HAS_SENTENCE_TRANSFORMERS = True
HAS_TRANSFORMERS = True

# Import custom utilities
from util import (
        keyword_hits,
        TECH_KEYWORDS,
        stance,
        apply_policies
    )

# Global cache for NLP results with persistence
NLP_CACHE = {}
CACHE_FILE = None
CACHE_SAVE_COUNTER = 0
CACHE_SAVE_INTERVAL = 1000

def get_cache_file_path():
    """Get the path for the persistent cache file"""
    # Use HF_HOME which should already be mounted and persistent outside container
    cache_dir = pathlib.Path(os.environ.get('HF_HOME', './cache')) / "qda_cache"
    logger.info(f"Cache directory from HF_HOME: {os.environ.get('HF_HOME', 'NOT_SET')}")
    logger.info(f"Full cache path: {cache_dir}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Cache directory created/verified: {cache_dir}")
    return cache_dir / "nlp_cache.pkl"

def load_persistent_cache():
    """Load cache from disk if available"""
    global NLP_CACHE, CACHE_FILE, CACHE_SAVE_COUNTER
    CACHE_FILE = get_cache_file_path()
    
    logger.info(f"Using cache directory: {CACHE_FILE.parent}")
    logger.info(f"Cache file path: {CACHE_FILE}")
    
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, 'rb') as f:
                NLP_CACHE = pickle.load(f)
            CACHE_SAVE_COUNTER = len(NLP_CACHE)
            logger.info(f"Loaded persistent cache with {len(NLP_CACHE)} entries from {CACHE_FILE}")
        except Exception as e:
            logger.warning(f"Failed to load cache file: {e}, starting with empty cache")
            NLP_CACHE = {}
            CACHE_SAVE_COUNTER = 0
    else:
        logger.info("No existing cache file found, starting with empty cache")
        NLP_CACHE = {}
        CACHE_SAVE_COUNTER = 0

def save_persistent_cache():
    """Save cache to disk"""
    global NLP_CACHE, CACHE_FILE
    logger.info(f"Attempting to save cache: {len(NLP_CACHE)} entries to {CACHE_FILE}")
    if NLP_CACHE and CACHE_FILE:
        try:
            with open(CACHE_FILE, 'wb') as f:
                pickle.dump(NLP_CACHE, f)
            logger.info(f"Saved persistent cache with {len(NLP_CACHE)} entries to {CACHE_FILE}")
        except Exception as e:
            logger.error(f"Failed to save cache file: {e}")
            logger.error(f"Cache file path: {CACHE_FILE}")
            logger.error(f"Cache directory exists: {CACHE_FILE.parent.exists()}")
            logger.error(f"Cache directory writable: {os.access(CACHE_FILE.parent, os.W_OK)}")
    else:
        logger.warning(f"Cannot save cache: NLP_CACHE={len(NLP_CACHE) if NLP_CACHE else 'None'}, CACHE_FILE={CACHE_FILE}")

def auto_save_cache_if_needed():
    """Automatically save cache if it has grown significantly"""
    global CACHE_SAVE_COUNTER, CACHE_SAVE_INTERVAL
    CACHE_SAVE_COUNTER += 1
    
    logger.debug(f"Cache counter: {CACHE_SAVE_COUNTER}, interval: {CACHE_SAVE_INTERVAL}, total entries: {len(NLP_CACHE)}")
    
    if CACHE_SAVE_COUNTER % CACHE_SAVE_INTERVAL == 0:
        logger.info(f"Auto-saving cache after {CACHE_SAVE_COUNTER} new entries (total: {len(NLP_CACHE)})")
        save_persistent_cache()

def get_cached_result(text: str, function_name: str, func, *args, **kwargs):
    """Get cached result or compute and cache it"""
    # Use more robust hashing for cache keys
    text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
    cache_key = f"{function_name}:{text_hash}"
    
    if cache_key in NLP_CACHE:
        logger.debug(f"Cache HIT for {function_name}: {text_hash[:8]}...")
        return NLP_CACHE[cache_key]
    
    logger.debug(f"Cache MISS for {function_name}: {text_hash[:8]}...")
    
    # Compute result
    try:
        result = func(text, *args, **kwargs)
        NLP_CACHE[cache_key] = result
        
        logger.debug(f"Cached result for {function_name}: {text_hash[:8]}... (total cache size: {len(NLP_CACHE)})")
        
        # Auto-save cache every 1000 entries
        auto_save_cache_if_needed()
        
        return result
    except Exception as e:
        logger.warning(f"Error in {function_name} for text: {e}")
        # Return default value on error
        if function_name == "sentiment":
            return "neutral"
        elif function_name == "stance":
            return "neutral"
        elif function_name == "rule_codes":
            return ["unspecified"]
        elif function_name == "ner":
            return [], []
        return None

def process_texts_batch(texts: List[str], nlp, senti, policies, keyword_fallback, strict, keyword_dict: dict = None, use_static_stances: bool = True):
    """Process texts in batches with caching"""
    logger.info(f"Processing {len(texts)} texts with caching...")
    
    sentiments, persons, orgs, stances, codes_all = [], [], [], [], []
    
    for i, text in enumerate(texts):
        if i % 100 == 0:
            logger.info(f"Processing text {i+1}/{len(texts)}")
        
        # Get cached or compute sentiment
        if senti:
            sentiment_result = get_cached_result(text, "sentiment", lambda t: senti(t)[0]["label"])
            sentiments.append(sentiment_result)
        else:
            sentiments.append("neutral")
        
        # Get cached or compute stance (only if static stances requested)
        if use_static_stances:
            stance_result = get_cached_result(text, "stance", stance)
            stances.append(stance_result)
        else:
            stances.append({})
        
        # Get cached or compute rule codes
        codes_result = get_cached_result(text, "rule_codes", rule_codes, policies, keyword_fallback, strict, keyword_dict)
        codes_all.append(codes_result)
        
        # Get cached or compute NER
        if nlp:
            ner_result = get_cached_result(text, "ner", lambda t: nlp(t))
            if ner_result:
                doc = ner_result
                persons.append([ent.text for ent in doc.ents if ent.label_ in ("PER", "PERSON")])
                orgs.append([ent.text for ent in doc.ents if ent.label_ in ("ORG", "ORG")])
            else:
                persons.append([])
                orgs.append([])
        else:
            persons.append([])
            orgs.append([])
    
    logger.info(f"Text processing completed. Cache size: {len(NLP_CACHE)}")
    
    # Save cache after processing batch (even if not 1000 entries)
    if len(NLP_CACHE) > 0:
        logger.info(f"Saving cache after batch processing: {len(NLP_CACHE)} entries")
        save_persistent_cache()
    
    return sentiments, persons, orgs, stances, codes_all

def load_policies(cfg_dir: pathlib.Path) -> dict:
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

def load_ner():
    """
    Load and configure the German language model with enhanced processing.
    
    Returns:
        spaCy language model with German language support or None if loading fails
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

def load_sentiment():
    """
    Load and configure the German sentiment analysis model with enhanced processing.
    
    Returns:
        Hugging Face pipeline for sentiment analysis or None if loading fails
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

def load_embedder():
    if not HAS_SENTENCE_TRANSFORMERS:
        logger.warning("sentence-transformers not available, embeddings disabled")
        return None
    
    # Get cache directory with fallback
    cache_dir = os.environ.get('SENTENCE_TRANSFORMERS_HOME')
    if not cache_dir:
        # Fallback to HF_HOME if SENTENCE_TRANSFORMERS_CACHE is not set
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

def detect_encoding(file_path: str, sample_size: int = 1024) -> str:
    """Detect the encoding of a text file."""
    
    with open(file_path, 'rb') as f:
        raw_data = f.read(sample_size)
    
    result = chardet.detect(raw_data)
    return result['encoding'] or 'utf-8'

def clean_text(text: str, lang: str = 'de') -> str:
    """
    Clean and normalize text input with language-specific processing.
    
    Args:
        text: Input text to clean
        lang: Language code ('de' for German, 'en' for English, etc.)
        
    Returns:
        Cleaned and normalized text
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Remove any non-printable characters except newlines and tabs
    import re
    text = re.sub(r'[^\x20-\x7E\n\t\räüößÄÜÖ]', ' ', text)
    
    # Normalize whitespace but preserve paragraph breaks
    text = '\n\n'.join(
        ' '.join(part.split()) 
        for part in text.split('\n\n')
    )
    
    # German-specific text normalization
    if lang.lower() == 'de':
        # Handle common German quotation marks and special characters
        text = text.replace('„', '"')
        text = text.replace('"', '"')
        #text = text.replace('''''', "'")
        
        # Replace common German abbreviations with full forms for better analysis
        abbrev_map = {
            r'\bzzgl\.\s*': 'bezüglich ',
            r'\bbzw\.\s*': 'beziehungsweise ',
            r'\bca\.\s*': 'circa ',
            r'\bz\.B\.\s*': 'zum Beispiel ',
            r'\bu\.a\.\s*': 'unter anderem ',
            r'\betc\.\s*': 'und so weiter ',
            r'\binsb\.\s*': 'insbesondere ',
            r'\bggf\.\s*': 'gegebenenfalls ',
            r'\bz\.T\.\s*': 'zum Teil ',
            r'\bi\.d\.R\.\s*': 'in der Regel ',
            r'\bz\.Z\.\s*': 'zur Zeit ',
            r'\bbzw\.\s*': 'beziehungsweise ',
            r'\bMrd\.\s*': 'Milliarden ',
            r'\bMio\.\s*': 'Millionen ',
            r'\bS\.\s*': 'Seite ',
            r'\bff\.\s*': 'fortfolgende ',
            r'\bNr\.\s*': 'Nummer ',
            r'\bJh\.\s*': 'Jahrhundert ',
        }
        
        for pattern, replacement in abbrev_map.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Handle common German compound word splits
        text = re.sub(r'(\w+)([A-ZÄÖÜ][a-zäöüß]+)', r'\1 \2', text)
    
    # Additional cleaning steps
    text = re.sub(r'\s+', ' ', text)  # Normalize all whitespace
    text = text.strip()
    
    return text

def load_csv_with_fallback(file_path):
    """Load a CSV file with simple and reliable error handling."""
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

def segment_rows(df: pd.DataFrame, min_text_length: int = 7) -> pd.DataFrame:
    """
    Segment DataFrame rows into individual text segments.
    
    Args:
        df: Input DataFrame with survey responses
        min_text_length: Minimum length of text to be considered valid
        
    Returns:
        DataFrame with segmented text data
    """
    logger.info(f"Segmenting {len(df)} rows into individual text segments")
    
    # Ensure all columns are strings and clean them
    df = df.astype(str).applymap(clean_text)
    
    # Get all column names and identify potential metadata columns
    all_columns = list(df.columns)
    metadata_columns = []
    text_columns = []
    
    # Simple heuristic to identify metadata columns (assuming they have mostly unique values)
    for col in all_columns:
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio > 0.9:  # Mostly unique values (likely IDs or metadata)
            metadata_columns.append(col)
        else:
            text_columns.append(col)
    
    # If we couldn't identify metadata, assume last column is metadata (common survey format)
    if not metadata_columns and len(all_columns) > 1:
        metadata_columns = [all_columns[-1]]
        text_columns = all_columns[:-1]
    
    logger.info(f"Identified text columns: {text_columns}")
    logger.info(f"Identified metadata columns: {metadata_columns}")
    logger.info(str(df.head()))
    segments = []
    
    for idx, row in df.iterrows():
        # Extract metadata (if any)
        metadata = {col: row[col] for col in metadata_columns if col in row}
        
        # Process each text column
        for col_idx, col in enumerate(text_columns, start=1):
            text = row[col].strip()
            
            # Skip empty or very short texts
            if not text or len(text) < min_text_length:
                continue
                
            # Split text into paragraphs (double newlines)
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            
            # Create a segment for each paragraph
            for para_idx, para in enumerate(paragraphs, start=1):
                # Further split long paragraphs into sentences
                sentences = [s.strip() for s in para.split('.') if s.strip()]
                
                # Create a segment for the full paragraph
                segments.append({
                    "set_id": idx + 1,
                    "segment_id": f"{idx+1}.{col_idx}.{para_idx}",
                    "question_idx": col_idx,
                    "question": col,
                    "text": para,
                    "text_type": "paragraph",
                    "char_count": len(para),
                    "word_count": len(para.split()),
                    "sentence_count": len(sentences),
                    **metadata  # Add all metadata fields
                })
                
                # Optionally create segments for individual sentences
                for sent_idx, sent in enumerate(sentences, start=1):
                    if len(sent) >= min_text_length:
                        segments.append({
                            "set_id": idx + 1,
                            "segment_id": f"{idx+1}.{col_idx}.{para_idx}.{sent_idx}",
                            "question_idx": col_idx,
                            "question": col,
                            "text": sent,
                            "text_type": "sentence",
                            "char_count": len(sent),
                            "word_count": len(sent.split()),
                            "sentence_count": 1,
                            "parent_paragraph": para_idx,
                            **metadata
                        })
    
    logger.info(f"Created {len(segments)} text segments from {len(df)} rows")
    
    # Convert to DataFrame and set appropriate data types
    if segments:
        result_df = pd.DataFrame(segments)
        
        # Set data types
        type_mapping = {
            'set_id': 'int32',
            'question_idx': 'int32',
            'char_count': 'int32',
            'word_count': 'int32',
            'sentence_count': 'int32',
            'text_type': 'category'
        }
        
        for col, dtype in type_mapping.items():
            if col in result_df.columns:
                result_df[col] = result_df[col].astype(dtype, errors='ignore')
        
        return result_df
    
    return pd.DataFrame()
    return pd.DataFrame(segments)

def compute_embeddings(texts: List[str], embedder):
    if embedder is None:
        logger.info("No embedder available, skipping embeddings")
        return None
    logger.info(f"Computing embeddings for {len(texts)} texts...")
    embeddings = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    logger.info(f"Embeddings computed successfully: {embeddings.shape}")
    return embeddings

def tfidf_features(texts: List[str]):
    logger.info(f"Computing TF-IDF features for {len(texts)} texts...")
    
    if not texts or not any(isinstance(t, str) and t.strip() for t in texts):
        logger.error("No valid text input provided")
        return None, None
        
    # Log first few texts for debugging
    sample_texts = [t[:100] + '...' if isinstance(t, str) and len(t) > 100 
                   else str(t) for t in texts[:3]]
    logger.info(f"Sample texts: {sample_texts}")
    
    # Preprocess texts with more lenient approach
    processed_texts = []
    for text in texts:
        if not isinstance(text, str) or not text.strip():
            continue
            
        # Basic cleaning - preserve more text
        cleaned = re.sub(r'[^\w\s]', ' ', text.lower()).strip()
        words = cleaned.split()
        
        # Keep text even if it only contains stopwords
        if words:
            processed_texts.append(' '.join(words))
    
    if not processed_texts:
        logger.error("No valid texts remaining after preprocessing")
        return None, None
    
    try:
        # First try with standard settings but keep stopwords initially
        vec = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=1,
            stop_words=None,  # Don't remove stopwords initially
            token_pattern=r'(?u)\b\w+\b',
            analyzer='word'
        )
        
        X = vec.fit_transform(processed_texts)
        logger.info(f"TF-IDF features computed successfully: {X.shape}")
        
        # If no features were created, try with character n-grams
        if X.shape[1] == 0:
            logger.warning("No word features created, trying character n-grams...")
            vec = TfidfVectorizer(
                analyzer='char_wb',
                ngram_range=(2, 5),
                min_df=1
            )
            X = vec.fit_transform(processed_texts)
            logger.info(f"TF-IDF features computed successfully with n-grams: {X.shape}")
            
            # If still no features, try with single characters
            if X.shape[1] == 0:
                logger.warning("No character n-gram features, trying single characters...")
                vec = TfidfVectorizer(
                    analyzer='char',
                    ngram_range=(1, 1),
                    min_df=1
                )
                X = vec.fit_transform(processed_texts)
                logger.info(f"TF-IDF features computed successfully with single characters: {X.shape}")
        
        if X.shape[1] == 0:
            logger.error("Failed to extract any features from the texts")
            return None, None
            
        logger.info(f"TF-IDF features computed: {X.shape}")
        logger.info(f"Vocabulary size: {len(vec.vocabulary_)}")
        logger.info(f"Sample features: {list(vec.vocabulary_.items())[:10]}")
        
        return vec, X
        
    except Exception as e:
        logger.error(f"Error in TF-IDF vectorization: {str(e)}")
        logger.error(f"Sample text: {texts[0][:200]}..." if texts else "No texts provided")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Exception details: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None, None

def get_cluster_texts(X, embeddings, k: int):
    if embeddings is not None:
        logger.info(f"Clustering {len(embeddings)} texts into {k} clusters using embeddings...")
        try:
            km = KMeans(n_clusters=min(k, len(embeddings)), n_init=10, random_state=42)
            labels = km.fit_predict(embeddings)
            centers = km.cluster_centers_
            logger.info(f"Clustering completed using embeddings, cluster sizes: {[list(labels).count(i) for i in range(k)]}")
            return labels, centers, None
        except Exception as e:
            logger.error(f"Error in embedding-based clustering: {str(e)}")
            # Fall through to TF-IDF if available
    
    if X is not None and hasattr(X, 'shape') and X.shape[0] > 0 and X.shape[1] > 0:
        logger.info(f"Clustering {X.shape[0]} texts into {k} clusters using TF-IDF...")
        try:
            km = KMeans(n_clusters=min(k, X.shape[0]), n_init=10, random_state=42)
            labels = km.fit_predict(X)
            logger.info(f"Clustering completed using TF-IDF, cluster sizes: {[list(labels).count(i) for i in range(k)]}")
            return labels, None, km
        except Exception as e:
            logger.error(f"Error in TF-IDF-based clustering: {str(e)}")
    
    # If we get here, both methods failed
    logger.warning("Both embedding and TF-IDF clustering failed. Using simple clustering...")
    # Return dummy labels (all zeros) as a fallback
    num_texts = len(embeddings) if embeddings is not None else (X.shape[0] if X is not None and hasattr(X, 'shape') else 0)
    if num_texts > 0:
        return [0] * num_texts, None, None
    
    logger.error("No valid data available for clustering")
    return None, None, None

def label_clusters(texts: List[str], labels, k: int, keyword_dict: dict = None) -> Dict[int, Dict[str, Any]]:
    logger.info(f"Labeling {k} clusters based on keywords...")
    df = pd.DataFrame({"text": texts, "label": labels})
    cluster_info = {}
    for c in range(k):
        texts_c = df[df["label"]==c]["text"].tolist()
        joined = " ".join(texts_c)
        if keyword_dict:
            hits = keyword_hits(joined, keyword_dict)
        else:
            hits = {}
        top = sorted(hits.items(), key=lambda x: x[1], reverse=True)[:3]
        label = ", ".join([t[0] for t in top if t[1]>0]) or "General"
        cluster_info[c] = {"label": label, "size": len(texts_c), "keywords": top, "texts": texts_c, "avg_length": sum(len(t) for t in texts_c) / len(texts_c) if texts_c else 0}
        logger.info(f"Cluster {c}: {label} (n={len(texts_c)}, keywords: {top})")
    return cluster_info

def rule_codes(text: str, policies: dict, keyword_fallback: bool, strict: bool, keyword_dict: dict = None) -> List[str]:
    rules = apply_policies(text, policies)
    if strict:
        return rules or ["unspecified"]
    kw: List[str] = []
    if keyword_fallback and keyword_dict is not None:
        hits = keyword_hits(text, keyword_dict)
        kw = [k for k, v in hits.items() if v>0]
        if not rules and not kw:
            return ["unspecified"]
    return sorted(set(rules + kw)) or ["unspecified"]

def split_sentences(text: str) -> List[str]:
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]

def summarize_textrank(texts: List[str], max_sentences: int = 3) -> str:
    logger.info(f"Generating TextRank summary for {len(texts)} texts (max {max_sentences} sentences)")
    sentences = []
    for t in texts:
        sentences.extend(split_sentences(t))
    if not sentences:
        logger.warning("No sentences found for summarization")
        return ""
    
    # Filter out empty or very short sentences
    sentences = [s for s in sentences if len(s.strip()) > 10]
    if not sentences:
        logger.warning("No valid sentences found for summarization after filtering")
        return ""
    
    try:
        vec = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=1, stop_words=GERMAN_STOPWORDS)
        X = vec.fit_transform(sentences)
        
        # Check if we have a valid vocabulary
        if X.shape[1] == 0:
            logger.warning("Empty vocabulary after TF-IDF, returning first few sentences")
            return " ".join(sentences[:max_sentences])
        
        sim = cosine_similarity(X)
        np.fill_diagonal(sim, 0.0)
        g = nx.from_numpy_array(sim)
        scores = nx.pagerank(g, max_iter=200, tol=1.0e-6)
        ranked = sorted(range(len(sentences)), key=lambda i: -scores.get(i,0.0))
        pick = sorted(ranked[:max_sentences])
        summary = " ".join(sentences[i] for i in pick)
        logger.info(f"TextRank summary generated: {len(summary)} characters")
        return summary
        
    except Exception as e:
        logger.warning(f"TextRank summarization failed: {e}, falling back to simple summary")
        # Fallback: return first few sentences
        return " ".join(sentences[:max_sentences])

def format_text_for_html(text: str) -> str:
    """Format text with proper HTML formatting for better readability"""
    if not text:
        return ""
    
    # Replace line breaks with <br> tags
    text = text.replace('\n', '<br>')
    
    # Handle markdown-style formatting
    # Bold: **text** -> <strong>text</strong>
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    
    # Italic: *text* -> <em>text</em>
    text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)
    
    # Headers: # Header -> <h4>Header</h4>
    text = re.sub(r'^#\s+(.+)$', r'<h4>\1</h4>', text, flags=re.MULTILINE)
    text = re.sub(r'^##\s+(.+)$', r'<h5>\1</h5>', text, flags=re.MULTILINE)
    text = re.sub(r'^###\s+(.+)$', r'<h6>\1</h6>', text, flags=re.MULTILINE)
    
    # Lists: - item -> <li>item</li>
    text = re.sub(r'^-\s+(.+)$', r'<li>\1</li>', text, flags=re.MULTILINE)
    
    # Wrap lists in <ul> tags
    if '<li>' in text:
        # Find consecutive <li> tags and wrap them
        lines = text.split('<br>')
        formatted_lines = []
        in_list = False
        
        for line in lines:
            if line.strip().startswith('<li>'):
                if not in_list:
                    formatted_lines.append('<ul>')
                    in_list = True
                formatted_lines.append(line)
            else:
                if in_list:
                    formatted_lines.append('</ul>')
                    in_list = False
                formatted_lines.append(line)
        
        if in_list:
            formatted_lines.append('</ul>')
        
        text = '<br>'.join(formatted_lines)
    
    # Code blocks: `code` -> <code>code</code>
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
    
    # URLs: http://... -> <a href="...">...</a>
    text = re.sub(r'https?://[^\s<>]+', r'<a href="\g<0>" target="_blank">\g<0></a>', text)
    
    return text

# Note: The following functions are now imported from dynamics.py:
# - generate_stance_patterns_from_texts
# - generate_sentiment_based_stance_patterns  
# - generate_comprehensive_stance_patterns
# - analyze_stances_with_dynamic_patterns

# Note: HTML report generation is now imported from html_report.py:
# - generate_html_report

def run_pipeline(input_path: str, out_dir: str, k_clusters: int = 6, cfg_dir: str = "./config", 
                clear_cache_flag: bool = False,
                use_dynamic_keywords: bool = True,
                use_dynamic_stance_patterns: bool = True,
                base_stance_patterns: dict = None,
                no_dynamic_policies: bool = False,
                llm_policies: bool = False,
                no_dynamic_keywords: bool = False,
                use_fixed_keywords: bool = False,
                no_dynamic_stances: bool = False,
                use_fixed_stances: bool = False):
    """Pipeline mit optionaler dynamischer Keyword- und Stance-Pattern-Generierung"""
    logger.info("1"*60)    
    
    logger.info(f"Starting QDA pipeline - input: {input_path}, output: {out_dir}, clusters: {k_clusters}")
    os.system("ls -l data")
    # Setup timeout
    setup_timeout()
    os.system("ls -l /")
    os.system("ls -l /cache")
    try:
        logger.info("2"*60)    
        # Handle cache management
        if clear_cache_flag:
            # Clear cache by resetting to empty dict
            global NLP_CACHE, CACHE_SAVE_COUNTER
            NLP_CACHE = {}
            CACHE_SAVE_COUNTER = 0
            logger.info("Cache cleared as requested")
        else:
            load_persistent_cache()
        logger.info("X"*60)    
        logger.info(f"Using cache with {len(NLP_CACHE)} existing entries")
        logger.info("X"*60)    
        cfg = pathlib.Path(cfg_dir)
        outp = pathlib.Path(out_dir)
        outp.mkdir(parents=True, exist_ok=True)
        (outp/"exports").mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directories created: {outp}")

        logger.info(f"Loading input data from: {input_path}")
        try:
            # Load CSV with enhanced handling
            df = load_csv_with_fallback(input_path)
            logger.info(f"Input data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Segment rows with improved text processing
            seg = segment_rows(df, min_text_length=7)
            
            if seg.empty:
                raise ValueError("No valid text segments found in the input data")
                
            logger.info(f"Processing {len(seg)} text segments")
            logger.info(f"Segment distribution by type:\n{seg['text_type'].value_counts()}")
            
            # Prepare texts for further processing
            texts = seg["text"].tolist()
            
        except Exception as e:
            logger.error(f"Error loading or processing input data: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise

        logger.info("Loading NLP models...")
        nlp = load_ner()
        senti = load_sentiment()
        embedder = load_embedder()

        # Policy handling logic
        if no_dynamic_policies:
            logger.info("[POLICY] Using only policies.yaml (no dynamic policies)")
            policies = load_policies(cfg)
        else:
            logger.info("[POLICY] Generating all policies dynamically (policies.yaml will NOT be used)")
            if llm_policies:
                logger.info("[POLICY] Using local LLM for policy generation")
                policies = generate_policies_with_llm(texts)
            else:
                policies = generate_dynamic_policies(texts)
        mode = (policies.get("policy", {}) or {}).get("mode", "augment").lower()
        strict = (mode == "strict")
        keyword_fallback = not strict
        logger.info(f"Policy mode: {mode} (strict: {strict}, keyword_fallback: {keyword_fallback})")

        logger.info("Computing text features...")
        vec, X = tfidf_features(texts)
        embeddings = compute_embeddings(texts, embedder) if embedder else None
        logger.info("Computing text features finished")

        logger.info(f"About to calculate cluster count... {k_clusters} {len(texts)}")
        k = min(max(2, len(texts)//4), max(2, k_clusters))
        logger.info(f"Adjusted cluster count: {k} (requested: {k_clusters}, texts: {len(texts)})")
        
        logger.info("About to start clustering...")
        labels, centers, km = get_cluster_texts(X, embeddings, k)
        logger.info("Clustering completed successfully")
        seg["cluster"] = labels
 
        # Determine static/dynamic usage per user request
        use_static_keywords = (no_dynamic_keywords or use_fixed_keywords)
        enable_dynamic_keywords = (not no_dynamic_keywords) and use_dynamic_keywords
        use_static_stances = (no_dynamic_stances or use_fixed_stances)
        enable_dynamic_stances = (not no_dynamic_stances) and use_dynamic_stance_patterns

        # Build keyword dicts for labeling and codes
        def build_dynamic_keyword_dict(dyn_kw: dict) -> Dict[str, List[str]]:
            dyn_dict: Dict[str, List[str]] = {}
            try:
                # Technical
                tech = (dyn_kw.get("Technical") or {}).get("keywords") if isinstance(dyn_kw.get("Technical"), dict) else None
                if tech:
                    dyn_dict["Technical"] = list(tech)
                # Thematic: dict of topics
                thematic = (dyn_kw.get("Thematic") or {}).get("keywords") if isinstance(dyn_kw.get("Thematic"), dict) else None
                if isinstance(thematic, dict):
                    for topic_name, words in thematic.items():
                        if words:
                            dyn_dict[str(topic_name)] = list(words)
                # Cluster_Specific: aggregate all
                cluster_spec = (dyn_kw.get("Cluster_Specific") or {}).get("keywords") if isinstance(dyn_kw.get("Cluster_Specific"), dict) else None
                if not cluster_spec and isinstance(dyn_kw.get("Cluster_Specific"), dict):
                    cluster_spec = dyn_kw.get("Cluster_Specific")
                if isinstance(cluster_spec, dict):
                    agg = []
                    for _, words in cluster_spec.items():
                        if words:
                            agg.extend(list(words))
                    if agg:
                        dyn_dict["ClusterKeywords"] = list(sorted(set(agg)))
            except Exception as e:
                logger.warning(f"Failed to build dynamic keyword dict: {e}")
            return dyn_dict

        keyword_dict_for_labeling: Dict[str, List[str]] = {}
        keyword_dict_for_codes: Dict[str, List[str]] = {}

        if enable_dynamic_keywords and 'dynamic_keywords' in locals():
            dyn_dict = build_dynamic_keyword_dict(dynamic_keywords)
            if dyn_dict:
                keyword_dict_for_labeling.update(dyn_dict)
                keyword_dict_for_codes.update(dyn_dict)
        # Include static TECH_KEYWORDS only if fixed keywords requested
        if use_fixed_keywords:
            for k, v in TECH_KEYWORDS.items():
                keyword_dict_for_labeling.setdefault(k, list(v))
                keyword_dict_for_codes.setdefault(k, list(v))
        # If user explicitly disabled dynamic keywords and did not request fixed, leave dicts empty

        # Dynamische Keyword-Generierung
        if enable_dynamic_keywords:
            logger.info("[START] Generating dynamic keywords from document content...")
            try:
                # Hybrid-Keywords generieren
                dynamic_keywords = hybrid_keyword_generation(texts)
                
                # Keywords in den Cluster-Info integrieren (nach der Erstellung)
                logger.info(f"[END] Generated dynamic keywords: {len(dynamic_keywords)} categories")
                
            except Exception as e:
                logger.warning(f"Dynamic keyword generation failed: {e}")
 
        # Dynamische Policy-Generierung
        # Default: Do not use policies.yaml at all. Only use it when --no-dynamic-policies is set.
        logger.info("[START] Generating dynamic policies from document content...")
        try:
            if no_dynamic_policies:
                # Augment YAML policies with dynamic ones (explicitly requested)
                dynamic_policies = generate_dynamic_policies(texts, policies)
                if dynamic_policies and dynamic_policies.get("codes"):
                    if "codes" not in policies:
                        policies["codes"] = []
                    policies["codes"].extend(dynamic_policies["codes"])
                    (outp / "dynamic_policies.json").write_text(
                        json.dumps(dynamic_policies, ensure_ascii=False, indent=2), 
                        encoding="utf-8"
                    )
                    logger.info(f"[END] Generated dynamic policies: {len(dynamic_policies['codes'])} new codes (merged into YAML)")
            else:
                # Policies are already dynamic (created above). Do not use YAML here.
                dynamic_policies = policies
                (outp / "dynamic_policies.json").write_text(
                    json.dumps(dynamic_policies, ensure_ascii=False, indent=2), 
                    encoding="utf-8"
                )
                logger.info("[END] Dynamic policies already in use (no YAML). Wrote dynamic_policies.json")
        except Exception as e:
            logger.warning(f"Dynamic policy generation failed: {e}")
 
        cluster_info = label_clusters(
            texts, labels, k,
            keyword_dict=keyword_dict_for_labeling if keyword_dict_for_labeling else None
        )

        # Dynamische Keywords in Cluster-Info integrieren (nach der Erstellung)
        if enable_dynamic_keywords and 'dynamic_keywords' in locals():
            try:
                for cluster_id in range(k):
                    if cluster_id in cluster_info:
                        cluster_texts = [texts[i] for i in range(len(texts)) if labels[i] == cluster_id]
                        if cluster_texts:
                            # Cluster-spezifische Keywords extrahieren
                            cluster_keywords = (
                                dynamic_keywords.get("Cluster_Specific", {}).get(f"Cluster_{cluster_id}", [])
                                if isinstance(dynamic_keywords.get("Cluster_Specific", {}), dict)
                                else dynamic_keywords.get("Cluster_Specific", {})
                            )
                            if cluster_keywords:
                                cluster_info[cluster_id]["dynamic_keywords"] = cluster_keywords
                
                # Dynamische Keywords speichern
                (outp / "dynamic_keywords.json").write_text(
                    json.dumps(dynamic_keywords, ensure_ascii=False, indent=2), 
                    encoding="utf-8"
                )
            except Exception as e:
                logger.warning(f"Failed to integrate dynamic keywords into cluster info: {e}")

        seg["cluster_label"] = seg["cluster"].map(lambda c: cluster_info.get(c,{}).get("label",""))

        # Now that keyword_dict_for_codes is constructed, process individual text segments
        logger.info("[START] Processing individual text segments with caching (sentiment, stances, codes)...")
        sentiments, persons, orgs, stances, codes_all = process_texts_batch(
            texts, nlp, senti, policies, keyword_fallback, strict,
            keyword_dict=keyword_dict_for_codes if keyword_dict_for_codes else None,
            use_static_stances=use_static_stances
        )
        seg["sentiment"] = sentiments
        seg["codes"] = codes_all
        seg["stance"] = stances
        seg["persons"] = persons
        seg["orgs"] = orgs
        logger.info("[END] Processing individual text segments with caching.")

        # If dynamic stances are enabled, generate and apply them now so they feed into subsequent steps
        stance_patterns_used = {}
        if enable_dynamic_stances:
            logger.info("Generating dynamic stance patterns from document content...")
            dynamic_stance_patterns = dynamics_generate_comprehensive_stance_patterns(
                texts=texts,
                base_stance_patterns=base_stance_patterns
            )
            stances = dynamics_analyze_stances_with_dynamic_patterns(texts, dynamic_stance_patterns)
            logger.info(f"Generated {len(dynamic_stance_patterns)} stance patterns")
            logger.info(f"Detected stances: {list(set([s for stance_list in stances for s in stance_list]))}")
            stance_patterns_used = dynamic_stance_patterns
            seg["stance"] = stances
            stance_report = {
                "patterns_used": stance_patterns_used,
                "stance_distribution": {},
                "generation_method": "dynamic"
            }
            for stance_list in stances:
                for s in stance_list:
                    stance_report["stance_distribution"][s] = stance_report["stance_distribution"].get(s, 0) + 1
            (outp / "stance_analysis.json").write_text(
                json.dumps(stance_report, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            logger.info("Stance analysis report saved")

        logger.info("Building codebook and analyzing code co-occurrences...")
        codebook = {}
        
        # Analyze code co-occurrences
        code_cooccurrence = analyze_code_cooccurrence(seg["codes"])
        
        # Save co-occurrence analysis results
        cooccurrence_dir = outp / "cooccurrence_analysis"
        cooccurrence_dir.mkdir(exist_ok=True)
        
        # Save co-occurrence matrix
        cooccurrence_matrix = pd.DataFrame(code_cooccurrence["matrix"], 
                                         index=code_cooccurrence["codes"], 
                                         columns=code_cooccurrence["codes"])
        cooccurrence_matrix.to_csv(cooccurrence_dir / "cooccurrence_matrix.csv")
        
        # Save code relationships
        relationships = []
        for (code1, code2), count in code_cooccurrence["pairs"].items():
            relationships.append({
                "code1": code1,
                "code2": code2,
                "co_occurrence_count": count,
                "co_occurrence_strength": code_cooccurrence["matrix"][
                    code_cooccurrence["codes"].index(code1)
                ][
                    code_cooccurrence["codes"].index(code2)
                ]
            })
        
        pd.DataFrame(relationships).sort_values("co_occurrence_strength", ascending=False).to_csv(
            cooccurrence_dir / "code_relationships.csv", index=False
        )
        
        # Generate and save network visualization
        generate_code_network(code_cooccurrence, cooccurrence_dir)
        
        # Generate word clouds for codes and themes
        logger.info("Generating word clouds for codes and themes...")
        try:
            # Generate word cloud for all codes
            code_freq = {code: count for code, count in zip(code_cooccurrence["codes"], np.diag(code_cooccurrence["matrix"]))}
            generate_word_cloud(
                code_freq,
                output_path=cooccurrence_dir / "code_wordcloud.png",
                title="Code Frequency Word Cloud",
                max_words=100
            )
            
            # Generate word cloud for code co-occurrence strengths
            code_strengths = {}
            for i, code1 in enumerate(code_cooccurrence["codes"]):
                total_strength = sum(code_cooccurrence["matrix"][i][j] for j in range(len(code_cooccurrence["codes"])) if i != j)
                if total_strength > 0:  # Only include codes that co-occur with others
                    code_strengths[code1] = total_strength
            
            if code_strengths:
                generate_word_cloud(
                    code_strengths,
                    output_path=cooccurrence_dir / "code_cooccurrence_wordcloud.png",
                    title="Code Co-occurrence Strength Word Cloud",
                    max_words=100
                )
            
            logger.info("Word clouds generated successfully")
            
        except Exception as e:
            logger.error(f"Error generating word clouds: {e}")
            logger.error(traceback.format_exc())
        
        # Continue with codebook generation
        for codes in seg["codes"]:
            for c in codes:
                codebook.setdefault(c, 0); codebook[c]+=1
        codebook_sorted = {k:int(v) for k,v in sorted(codebook.items(), key=lambda x: (-x[1], x[0]))}
        logger.info(f"Codebook built: {len(codebook_sorted)} unique codes")

        # Generate structured reports
        logger.info("Generating structured reports...")
        try:
            # Extract question texts if available
            question_texts = {}
            if 'question_text' in seg.columns:
                question_texts = seg[['question_idx', 'question_text']].drop_duplicates().set_index('question_idx')['question_text'].to_dict()
            
            # Generate all reports
            generate_structured_report(
                seg=seg,
                codebook=codebook_sorted,
                code_cooccurrence=code_cooccurrence,
                output_dir=outp,
                question_texts=question_texts if question_texts else None
            )
        except Exception as e:
            logger.error(f"Error generating structured reports: {e}")
            logger.error(traceback.format_exc())
        
        # Generate export files
        logger.info("Generating export files...")
        exp_rows = []
        for _, r in seg.iterrows():
            codes_str = "; ".join(r["codes"]) if isinstance(r["codes"], list) else str(r["codes"])
            doc_name = f"Set{int(r['set_id'])}_Q{int(r['question_idx'])}"
            exp_rows.append({
                "Document": doc_name,
                "ParagraphID": 1,
                "Segment": r["text"],
                "Codes": codes_str,
                "ClusterLabel": r.get("cluster_label",""),
                "Bio": r.get("bio",""),
                "Coder": "AutoQDA",
                "Start": 0,
                "End": len(r["text"]),
            })
        
        maxqda_path = outp/"exports"/"maxqda_import.csv"
        atlasti_path = outp/"exports"/"atlasti_import.csv"
        pd.DataFrame(exp_rows).to_csv(maxqda_path, index=False)
        pd.DataFrame(exp_rows).to_csv(atlasti_path, index=False)
        logger.info(f"Export files created: {maxqda_path}, {atlasti_path}")

        seg_csv = outp / "coded_segments.csv"
        seg.to_csv(seg_csv, index=False)
        logger.info(f"Segments saved to: {seg_csv}")
        
        (outp / "codebook.json").write_text(json.dumps(codebook_sorted, ensure_ascii=False, indent=2), encoding="utf-8")
        (outp / "themes.json").write_text(json.dumps(cluster_info, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Codebook and themes saved")

        logger.info("Generating summaries...")
        cluster_summaries = {}
        for cid in range(k):
            try:
                tx = seg[seg["cluster"]==cid]["text"].tolist()
                cluster_summaries[cid] = summarize_textrank(tx, max_sentences=6)
                logger.info(f"Generated summary for cluster {cid}: {len(cluster_summaries[cid])} characters")
            except Exception as e:
                logger.warning(f"Failed to generate summary for cluster {cid}: {e}")
                cluster_summaries[cid] = f"Summary generation failed: {e}"
        
        try:
            global_summary = summarize_textrank(texts, max_sentences=12)
            logger.info(f"Generated global summary: {len(global_summary)} characters")
        except Exception as e:
            logger.warning(f"Failed to generate global summary: {e}")
            global_summary = f"Global summary generation failed: {e}"
        
        (outp / "summaries.json").write_text(json.dumps({"global": global_summary, "clusters": cluster_summaries}, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Summaries generated and saved")

        logger.info("Generating HTML report...")
        
        # Generate the improved HTML report
        html_content = generate_html_report(seg, cluster_info, cluster_summaries, codebook_sorted, global_summary, input_path, k)
        
        (outp / "report.html").write_text(html_content, encoding="utf-8")
        logger.info("HTML report generated")

        # Generate the policies/keywords/stances summary HTML
        from html_report import generate_policies_html
        # Collect used policies, keywords, stances from the run
        used_policies = set()
        used_keywords = set()
        used_stances = set()
        # Collect from all codes assigned to segments (guard for missing column)
        try:
            if isinstance(seg, pd.DataFrame) and ("codes" in seg.columns):
                for codes in seg["codes"]:
                    for c in (codes or []):
                        used_policies.add(c)
            else:
                # Fallback: use codebook keys if available
                for c in (codebook_sorted.keys() if isinstance(codebook_sorted, dict) else []):
                    used_policies.add(c)
        except Exception:
            for c in (codebook_sorted.keys() if isinstance(codebook_sorted, dict) else []):
                used_policies.add(c)
        # Collect from all keywords in cluster_info and dynamic_keywords
        if 'dynamic_keywords' in locals():
            for cat, catinfo in dynamic_keywords.items():
                if cat == "Base_Keywords" or cat == "summary":
                    continue
                kws = catinfo.get("keywords", {})
                if isinstance(kws, dict):
                    for subcat, subkws in kws.items():
                        for kw in subkws:
                            used_keywords.add(kw)
                elif isinstance(kws, list):
                    for kw in kws:
                        used_keywords.add(kw)
        # Collect from all stances assigned to segments
        if 'stances' in locals():
            for stance_list in stances:
                for s in stance_list:
                    used_stances.add(s)
        # Compose the HTML
        policies_html = generate_policies_html(
            policies=policies,
            dynamic_policies=dynamic_policies if 'dynamic_policies' in locals() else {},
            keywords=dynamic_keywords if 'dynamic_keywords' in locals() else {},
            dynamic_keywords=dynamic_keywords if 'dynamic_keywords' in locals() else {},
            stance_patterns=stance_patterns_used if 'stance_patterns_used' in locals() else {},
            used_policies=used_policies,
            used_keywords=used_keywords,
            used_stances=used_stances,
            input_path=input_path
        )
        (outp / "policies.html").write_text(policies_html, encoding="utf-8")
        logger.info("Policies/keywords/stances summary HTML generated as policies.html")

        # REFI-QDA export
        try:
            logger.info("Starting REFI-QDA export...")
            from refiqda import build_qdpx
            qdpx_path = outp / "exports" / "qda_export.qdpx"
            build_qdpx(str(seg_csv), str(qdpx_path), project_name="Brandschutz QDA")
            logger.info(f"REFI-QDA export completed: {qdpx_path}")
        except Exception as e:
            logger.error(f"REFI-QDA export failed: {e}")

        logger.info(f"Pipeline completed successfully. Outputs in {outp}")
        
        # Save cache before exiting
        save_persistent_cache()
        
    except TimeoutError as e:
        logger.error(f"Pipeline timed out: {e}")
        # Save cache even on timeout
        try:
            save_persistent_cache()
        except:
            pass
        import traceback
        logger.error(f"Traceback (TimeoutError):\n{traceback.format_exc()}")
        raise
    except Exception as e:
        logger.error(f"Pipeline failed with exception: {e}")
        import traceback
        logger.error(f"Traceback (Exception):\n{traceback.format_exc()}")
        # Save cache even on error
        try:
            save_persistent_cache()
        except:
            pass
        raise
    finally:
        # Always clear timeout
        logger.info("Clearing timeout at finally")
        clear_timeout()

# Placeholder for LLM-based policy generation
def analyze_code_cooccurrence(code_lists: List[List[str]]) -> Dict:
    """
    Analyze co-occurrence of codes across text segments.
    
    Args:
        code_lists: List of lists, where each inner list contains codes for a text segment
        
    Returns:
        Dictionary containing co-occurrence matrix, code list, and pair counts
    """
    # Get all unique codes
    all_codes = sorted(list({code for codes in code_lists for code in codes}))
    code_to_idx = {code: i for i, code in enumerate(all_codes)}
    n_codes = len(all_codes)
    
    # Initialize co-occurrence matrix
    cooccurrence_matrix = np.zeros((n_codes, n_codes), dtype=int)
    pair_counts = defaultdict(int)
    
    # Count co-occurrences
    for codes in code_lists:
        # Only consider segments with at least 2 codes
        if len(codes) >= 2:
            # Update co-occurrence counts for each pair of codes in this segment
            for code1, code2 in combinations(sorted(codes), 2):
                i, j = code_to_idx[code1], code_to_idx[code2]
                cooccurrence_matrix[i][j] += 1
                cooccurrence_matrix[j][i] += 1  # Matrix is symmetric
                pair_counts[(code1, code2)] += 1
    
    # Convert to normalized co-occurrence strength (Jaccard similarity)
    code_counts = np.diag(cooccurrence_matrix)
    for i in range(n_codes):
        for j in range(i+1, n_codes):
            if code_counts[i] > 0 and code_counts[j] > 0:
                # Jaccard similarity: intersection / union
                intersection = cooccurrence_matrix[i][j]
                union = code_counts[i] + code_counts[j] - intersection
                similarity = intersection / union if union > 0 else 0
                cooccurrence_matrix[i][j] = similarity
                cooccurrence_matrix[j][i] = similarity
    
    return {
        "matrix": cooccurrence_matrix.tolist(),
        "codes": all_codes,
        "pairs": pair_counts,
        "code_counts": {code: code_counts[i] for i, code in enumerate(all_codes)}
    }

def generate_code_network(cooccurrence_data: Dict, output_dir: pathlib.Path, 
                        min_strength: float = 0.1, max_nodes: int = 50):
    """
    Generate and save a network visualization of code co-occurrences.
    
    Args:
        cooccurrence_data: Output from analyze_code_cooccurrence
        output_dir: Directory to save the visualization
        min_strength: Minimum co-occurrence strength to include an edge
        max_nodes: Maximum number of nodes to include (top N by frequency)
    """
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
        
        # Prepare data
        matrix = np.array(cooccurrence_data["matrix"])
        codes = cooccurrence_data["codes"]
        code_counts = cooccurrence_data["code_counts"]
        
        # Limit to top N most frequent codes if needed
        if len(codes) > max_nodes:
            top_codes = sorted(codes, key=lambda x: code_counts[x], reverse=True)[:max_nodes]
            code_indices = [codes.index(code) for code in top_codes]
            matrix = matrix[np.ix_(code_indices, code_indices)]
            codes = top_codes
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes with size based on code frequency
        for code in codes:
            G.add_node(code, size=code_counts[code])
        
        # Add edges based on co-occurrence strength
        n = len(codes)
        for i in range(n):
            for j in range(i+1, n):
                strength = matrix[i][j]
                if strength >= min_strength:
                    G.add_edge(codes[i], codes[j], weight=strength)
        
        # Skip if no edges meet the threshold
        if not G.edges():
            logger.warning(f"No code co-occurrences found with strength >= {min_strength}")
            return
        
        # Draw the graph
        plt.figure(figsize=(16, 12))
        
        # Node sizes based on code frequency (scaled for visibility)
        node_sizes = [G.nodes[code]['size'] * 20 + 100 for code in G.nodes()]
        
        # Edge widths based on co-occurrence strength
        edge_widths = [G[u][v]['weight'] * 5 for u, v in G.edges()]
        
        # Use spring layout for better node distribution
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        
        # Draw nodes and edges
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                             node_color='lightblue', alpha=0.7)
        nx.draw_networkx_edges(G, pos, width=edge_widths, 
                             edge_color='gray', alpha=0.5)
        
        # Draw node labels with adjusted positions to avoid overlap
        label_pos = {k: (v[0], v[1] + 0.03) for k, v in pos.items()}
        nx.draw_networkx_labels(G, label_pos, font_size=8, 
                              font_family='sans-serif')
        
        # Add title and save
        plt.title("Code Co-occurrence Network")
        plt.axis('off')
        
        # Save the figure
        output_path = output_dir / "code_network.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Code co-occurrence network saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to generate code network: {e}")
        logger.error(traceback.format_exc())

def generate_word_cloud(
    word_freq: Dict[str, float], 
    output_path: pathlib.Path, 
    title: str = "Word Cloud",
    max_words: int = 200,
    width: int = 1600,
    height: int = 800,
    background_color: str = "white",
    colormap: str = "viridis"
) -> None:
    """
    Generate and save a word cloud from word frequencies.
    
    Args:
        word_freq: Dictionary of words and their frequencies/weights
        output_path: Path to save the word cloud image
        title: Title for the word cloud
        max_words: Maximum number of words to include
        width: Width of the output image
        height: Height of the output image
        background_color: Background color of the word cloud
        colormap: Matplotlib colormap name for coloring the words
    """
    try:
        if not word_freq:
            logger.warning("No word frequencies provided for word cloud")
            return
            
        # Create a color function based on the colormap
        cmap = plt.get_cmap(colormap)
        color_func = lambda *args, **kwargs: tuple(int(x * 255) for x in cmap(random.random())[:3])
        
        # Create and generate the word cloud
        wc = WordCloud(
            width=width,
            height=height,
            background_color=background_color,
            max_words=max_words,
            color_func=color_func,
            prefer_horizontal=0.9,
            min_font_size=8,
            max_font_size=200,
            relative_scaling=0.5,
            colormap=colormap
        )
        
        # Generate the word cloud
        wc.generate_from_frequencies(word_freq)
        
        # Create a figure and axis
        plt.figure(figsize=(16, 9))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, fontsize=20, pad=20)
        
        # Save the figure
        plt.tight_layout(pad=0)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
        logger.info(f"Word cloud saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error generating word cloud: {e}")
        logger.error(traceback.format_exc())

def generate_structured_report(
    seg: pd.DataFrame,
    codebook: Dict[str, int],
    code_cooccurrence: Dict,
    output_dir: pathlib.Path,
    question_texts: Optional[Dict[int, str]] = None
) -> None:
    """
    Generate structured reports for qualitative analysis results.
    
    This function creates a comprehensive set of reports including:
    - An overall summary of the analysis
    - Individual reports for each question (if question_texts is provided)
    - Detailed code-specific reports
    - Visualizations including code networks and word clouds
    
    Args:
        seg: DataFrame containing segmented text data with codes and metadata
        codebook: Dictionary of codes and their frequencies
        code_cooccurrence: Output from analyze_code_cooccurrence
        output_dir: Directory to save the reports
        question_texts: Optional dictionary mapping question indices to question texts
    """
    try:
        # Create reports directory
        reports_dir = output_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure the output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting report generation in {reports_dir}")
        
        # 1. Overall Summary Report
        logger.info("Generating overall summary report...")
        generate_overall_summary(
            seg=seg,
            codebook=codebook,
            code_cooccurrence=code_cooccurrence,
            output_path=reports_dir / "overall_summary.md"
        )
        
        # 2. Per-Question Reports (if question_texts is provided)
        if question_texts and len(question_texts) > 0:
            logger.info(f"Generating reports for {len(question_texts)} questions...")
            
            # Create a subdirectory for question reports
            question_reports_dir = reports_dir / "question_reports"
            question_reports_dir.mkdir(exist_ok=True)
            
            for q_idx, q_text in question_texts.items():
                try:
                    logger.info(f"Processing question {q_idx}...")
                    
                    # Filter segments for this question
                    if 'question_idx' in seg.columns:
                        q_seg = seg[seg['question_idx'] == q_idx].copy()
                    else:
                        # If no question_idx column, use all segments for each question
                        # This is a fallback and might not be ideal - we should log a warning
                        logger.warning("No 'question_idx' column found in segments. Using all segments for each question.")
                        q_seg = seg.copy()
                    
                    # Skip if no segments for this question
                    if len(q_seg) == 0:
                        logger.warning(f"No segments found for question {q_idx}")
                        continue
                    
                    logger.info(f"Found {len(q_seg)} segments for question {q_idx}")
                    
                    # Generate question-specific co-occurrence data if we have codes
                    q_code_lists = [codes for codes in q_seg['codes'] if codes]  # Filter out empty code lists
                    
                    if q_code_lists and len(q_code_lists) > 0:
                        logger.debug(f"Analyzing code co-occurrence for question {q_idx}...")
                        q_cooccurrence = analyze_code_cooccurrence(q_code_lists)
                        
                        # Generate and save code network visualization
                        try:
                            network_dir = question_reports_dir / f"q{q_idx}_network"
                            network_dir.mkdir(exist_ok=True)
                            generate_code_network(
                                cooccurrence_data=q_cooccurrence,
                                output_dir=network_dir,
                                min_strength=0.1,
                                max_nodes=30  # Limit nodes for question-specific networks
                            )
                        except Exception as e:
                            logger.error(f"Error generating network for question {q_idx}: {e}")
                    else:
                        q_cooccurrence = None
                    
                    # Generate report for this question
                    logger.debug(f"Generating report for question {q_idx}...")
                    generate_question_report(
                        seg=q_seg,
                        question_text=q_text,
                        question_idx=q_idx,
                        output_path=question_reports_dir / f"question_{q_idx}_report.md",
                        code_cooccurrence=q_cooccurrence
                    )
                    
                    logger.info(f"Completed report for question {q_idx}")
                    
                except Exception as e:
                    logger.error(f"Error processing question {q_idx}: {e}")
                    logger.error(traceback.format_exc())
                    continue  # Continue with next question even if one fails
        
        # 3. Code-Specific Reports
        logger.info("Generating code-specific reports...")
        code_reports_dir = reports_dir / "code_reports"
        code_reports_dir.mkdir(exist_ok=True)
        
        generate_code_reports(
            seg=seg,
            codebook=codebook,
            code_cooccurrence=code_cooccurrence,
            output_dir=code_reports_dir
        )
        
        # 4. Generate a README with an overview of all reports
        logger.info("Generating README...")
        readme_path = reports_dir / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write("# Qualitative Analysis Reports\n\n")
            f.write("This directory contains the following reports and visualizations:\n\n")
            
            f.write("## Main Reports\n")
            f.write("- [Overall Summary](overall_summary.md) - High-level analysis of all data\n")
            
            if question_texts and len(question_texts) > 0:
                f.write("\n## Question-Specific Reports\n")
                for q_idx in sorted(question_texts.keys()):
                    f.write(f"- [Question {q_idx} Report](question_reports/question_{q_idx}_report.md) - Analysis of responses to question {q_idx}\n")
            
            f.write("\n## Code Analysis\n")
            f.write("- [Code Reports](code_reports/) - Detailed reports for each code\n")
            if question_texts and len(question_texts) > 0:
                f.write("- [Question-Specific Code Networks](question_reports/) - Network visualizations for each question\n")
            
            f.write("\n## How to Use These Reports\n")
            f.write("1. Start with the **Overall Summary** for a high-level view\n")
            f.write("2. Review **Question-Specific Reports** for detailed analysis of each question\n")
            f.write("3. Use **Code Reports** to explore specific codes in depth\n")
            f.write("4. Check the network visualizations to understand code relationships\n")
        
        logger.info(f"All reports have been generated in {reports_dir}")
        logger.info(f"Open {reports_dir}/README.md for an overview of all available reports")
        
    except Exception as e:
        logger.error(f"Error generating structured reports: {e}")
        logger.error(traceback.format_exc())
        raise  # Re-raise the exception to be handled by the caller

def generate_overall_summary(
    seg: pd.DataFrame,
    codebook: Dict[str, int],
    code_cooccurrence: Dict,
    output_path: pathlib.Path
) -> None:
    """Generate an overall summary report of the analysis."""
    try:
        # Calculate basic statistics
        total_segments = len(seg)
        total_codes = sum(codebook.values())
        unique_codes = len(codebook)
        avg_codes_per_segment = total_codes / total_segments if total_segments > 0 else 0
        
        # Get top codes
        top_codes = sorted(codebook.items(), key=lambda x: -x[1])[:10]
        
        # Get top code pairs
        relationships = []
        for (code1, code2), count in code_cooccurrence["pairs"].items():
            i, j = code_cooccurrence["codes"].index(code1), code_cooccurrence["codes"].index(code2)
            strength = code_cooccurrence["matrix"][i][j]
            relationships.append((code1, code2, strength, count))
        
        top_pairs = sorted(relationships, key=lambda x: -x[2])[:5]
        
        # Generate markdown report
        report = f"# Qualitative Analysis Summary Report\n\n"
        report += f"## Overview\n"
        report += f"- **Total Segments Analyzed**: {total_segments:,}\n"
        report += f"- **Total Code Applications**: {total_codes:,}\n"
        report += f"- **Unique Codes**: {unique_codes:,}\n"
        report += f"- **Average Codes per Segment**: {avg_codes_per_segment:.2f}\n\n"
        
        report += f"## Most Frequent Codes\n"
        for code, count in top_codes:
            report += f"- **{code}**: {count:,} applications\n"
        report += "\n## Strongest Code Relationships\n"
        for code1, code2, strength, count in top_pairs:
            report += f"- **{code1}** ↔ **{code2}**: Strength={strength:.2f} (co-occurred {count:,} times)\n"
        # Save the report
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding='utf-8')
        logger.info(f"Overall summary report saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error generating overall summary: {e}")
        logger.error(traceback.format_exc())

def generate_question_report(
    seg: pd.DataFrame,
    question_text: str,
    question_idx: int,
    output_path: pathlib.Path,
    code_cooccurrence: Optional[Dict] = None
) -> None:
    """
    Generate a comprehensive report for a specific question with detailed analysis.
    
    Args:
        seg: DataFrame containing segmented text data for this question
        question_text: The full text of the question
        question_idx: Index of the question
        output_path: Path to save the report
        code_cooccurrence: Optional co-occurrence data for this question
    """
    try:
        # Create output directory if it doesn't exist
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate question-specific statistics
        total_segments = len(seg)
        if total_segments == 0:
            logger.warning(f"No segments found for question {question_idx}")
            return
            
        # Calculate code frequencies
        code_freq = {}
        all_codes = []
        for codes in seg['codes']:
            all_codes.append(codes)
            for code in codes:
                code_freq[code] = code_freq.get(code, 0) + 1
        
        # Get top codes (up to 15)
        top_codes = sorted(code_freq.items(), key=lambda x: -x[1])[:15]
        
        # Calculate sentiment statistics if available
        sentiment_stats = {}
        if 'sentiment' in seg.columns:
            sentiments = [s for s in seg['sentiment'] if pd.notna(s)]
            if sentiments:
                sentiment_stats = {
                    'positive': sum(1 for s in sentiments if s > 0.2),
                    'neutral': sum(1 for s in sentiments if -0.2 <= s <= 0.2),
                    'negative': sum(1 for s in sentiments if s < -0.2)
                }
        
        # Generate markdown report
        report = f"# Question {question_idx} Analysis\n\n"
        report += f"## Question Text\n{question_text}\n\n"
        
        # Summary section
        report += f"## 📊 Summary\n"
        report += f"- **Total Responses**: {total_segments:,}\n"
        report += f"- **Unique Codes Applied**: {len(code_freq):,}\n"
        
        # Add sentiment summary if available
        if sentiment_stats:
            total = sum(sentiment_stats.values())
            if total > 0:
                report += f"- **Sentiment Distribution**:\n"
                def sentiment_bar(count, total, width=30):
                    filled = '█' * int((count / total) * width) if total > 0 else ''
                    return f"`{filled.ljust(width)}` {count} ({count/total:.1%})"
                
                report += f"  - 😊 Positive: {sentiment_bar(sentiment_stats['positive'], total)}\n"
                report += f"  - 😐 Neutral: {sentiment_bar(sentiment_stats['neutral'], total)}\n"
                report += f"  - 😟 Negative: {sentiment_bar(sentiment_stats['negative'], total)}\n"
        
        # Code frequency section
        report += f"\n## 🔍 Code Frequency\n"
        if top_codes:
            max_count = max(count for _, count in top_codes)
            for code, count in top_codes:
                bar_length = int((count / max_count) * 30)
                bar = '█' * bar_length
                report += f"- **{code}**: {bar} {count:,} ({count/total_segments:.0%})\n"
        # Add co-occurrence analysis if data is available
        if code_cooccurrence and 'pairs' in code_cooccurrence and code_cooccurrence['pairs']:
            report += f"\n## 🔗 Code Relationships\n"
            # Get top co-occurring code pairs for this question
            relationships = []
            for (code1, code2), count in code_cooccurrence["pairs"].items():
                if code1 in code_freq and code2 in code_freq:  # Only include codes that exist in this question
                    i = code_cooccurrence["codes"].index(code1)
                    j = code_cooccurrence["codes"].index(code2)
                    strength = code_cooccurrence["matrix"][i][j]
                    relationships.append((code1, code2, strength, count))
            
            top_pairs = sorted(relationships, key=lambda x: -x[2])[:5]  # Top 5 by strength
            
            if top_pairs:
                report += "Most strongly related code pairs:\n\n"
                for code1, code2, strength, count in top_pairs:
                    report += f"- **{code1}** ↔ **{code2}**: Strength={strength:.2f} (co-occurred {count} times)\n"
            
                # Generate and save a word cloud of co-occurring codes
                try:
                    cooccurrence_freq = {}
                    for code1, code2, strength, count in relationships:
                        pair = f"{code1} + {code2}"
                        cooccurrence_freq[pair] = strength * 100  # Scale for better visualization
                    
                    if cooccurrence_freq:
                        wc_path = output_dir / f"q{question_idx}_code_relationships_wordcloud.png"
                        generate_word_cloud(
                            word_freq=cooccurrence_freq,
                            output_path=wc_path,
                            title=f"Code Relationships - Question {question_idx}",
                            max_words=50
                        )
                        report += f"\n![Code Relationships Word Cloud]({wc_path.name})\n"
                except Exception as e:
                    logger.error(f"Error generating code relationship word cloud: {e}")
        
        # Example segments section
        report += "\n## 📝 Example Responses\n"
        sample_size = min(5, len(seg))
        sample_segments = seg.sample(sample_size) if len(seg) > sample_size else seg
        
        for idx, (_, row) in enumerate(sample_segments.iterrows(), 1):
            codes = row.get('codes', [])
            sentiment = row.get('sentiment', None)
            text = row.get('text', '').strip()
            
            if not text:
                continue
                
            report += f"\n### Response {idx}\n"
            
            # Add sentiment emoji if available
            if pd.notna(sentiment):
                if sentiment > 0.2:
                    sentiment_emoji = "😊"
                elif sentiment < -0.2:
                    sentiment_emoji = "😟"
                else:
                    sentiment_emoji = "😐"
                report += f"**Sentiment**: {sentiment_emoji} ({sentiment:.2f})\n\n"
            
            # Add codes if available
            if codes:
                report += f"**Codes**: {', '.join(f'`{c}`' for c in codes)}\n\n"
            
            # Add the text with smart truncation
            max_length = 400
            if len(text) > max_length:
                # Try to truncate at sentence boundary
                truncated = text[:max_length]
                last_period = truncated.rfind('.')
                if last_period > max_length // 2:  # Only truncate if we find a reasonable break point
                    truncated = truncated[:last_period + 1]
                report += f"> {truncated}... [response truncated]\n"
            else:
                report += f"> {text}\n"
        
        # Add a word cloud of frequent terms if there's enough text
        try:
            from collections import Counter
            from wordcloud import STOPWORDS
            import string
            
            # Combine all text for this question
            all_text = ' '.join(str(t) for t in seg['text'] if pd.notna(t))
            
            # Basic text cleaning and tokenization
            words = []
            for word in all_text.lower().split():
                # Remove punctuation and numbers
                word = word.translate(str.maketrans('', '', string.punctuation + '0123456789'))
                if (len(word) > 2 and  # At least 3 characters
                    word not in STOPWORDS and  # Not in stopwords
                    not word.isnumeric()):  # Not a number
                    words.append(word)
            
            # Count word frequencies
            word_freq = Counter(words)
            
            # Generate word cloud if we have enough words
            if len(word_freq) >= 5:  # At least 5 unique words
                wc_path = output_dir / f"q{question_idx}_wordcloud.png"
                generate_word_cloud(
                    word_freq=word_freq,
                    output_path=wc_path,
                    title=f"Frequent Terms - Question {question_idx}",
                    max_words=100
                )
                report += f"\n## 📊 Frequent Terms\n"
                report += f"![Word Cloud]({wc_path.name})\n"
        except Exception as e:
            logger.error(f"Error generating word cloud: {e}")
        
        # Add a section for detailed code analysis if there are codes
        if code_freq:
            report += "\n## 🔍 Detailed Code Analysis\n"
            report += "### Code Frequencies\n"
            # Sort codes by frequency (descending)
            sorted_codes = sorted(code_freq.items(), key=lambda x: -x[1])
            
            for code, count in sorted_codes:
                percentage = (count / total_segments) * 100
                report += f"- **{code}**: {count:,} responses ({percentage:.1f}%)\n"
            
            # Add a section for code co-occurrence matrix if available
            if code_cooccurrence and 'matrix' in code_cooccurrence and 'codes' in code_cooccurrence:
                # Get codes that appear in this question
                question_codes = set(code_freq.keys())
                if len(question_codes) > 1:  # Need at least 2 codes for co-occurrence
                    report += "\n### Code Co-occurrence Matrix\n"
                    report += "(Values indicate the Jaccard similarity between code pairs, from 0 to 1)\n\n"
                    
                    # Create a matrix of just the codes in this question
                    matrix = []
                    code_list = sorted(question_codes)
                    
                    # Build the header row
                    header = ["Code"] + code_list
                    rows = []
                    
                    # Build each row of the matrix
                    for i, code1 in enumerate(code_list):
                        row = [code1]
                        for j, code2 in enumerate(code_list):
                            if i == j:
                                row.append("-")
                            else:
                                # Find the indices in the full co-occurrence matrix
                                try:
                                    idx1 = code_cooccurrence["codes"].index(code1)
                                    idx2 = code_cooccurrence["codes"].index(code2)
                                    similarity = code_cooccurrence["matrix"][idx1][idx2]
                                    row.append(f"{similarity:.2f}")
                                except (ValueError, IndexError):
                                    row.append("0.00")
                        rows.append(row)
                    
                    # Format as a markdown table
                    # Header
                    report += "| " + " | ".join(header) + " |\n"
                    report += "|" + "|-" * len(header) + "|\n"
                    
                    # Rows
                    for row in rows:
                        report += "| " + " | ".join(str(cell) for cell in row) + " |\n"
                    
                    report += "\n*Note: The diagonal is marked with '-' as it represents self-similarity.*\n"
        
        # Add a section for potential themes or patterns
        report += "\n## 🎯 Potential Themes and Patterns\n"
        
        # Try to identify potential themes based on code co-occurrence
        if code_cooccurrence and 'pairs' in code_cooccurrence and code_cooccurrence['pairs']:
            # Look for groups of codes that frequently co-occur
            code_groups = {}
            
            # Simple heuristic: if two codes co-occur frequently, group them
            for (code1, code2), count in code_cooccurrence["pairs"].items():
                if code1 in code_freq and code2 in code_freq:  # Only include codes in this question
                    i = code_cooccurrence["codes"].index(code1)
                    j = code_cooccurrence["codes"].index(code2)
                    strength = code_cooccurrence["matrix"][i][j]
                    
                    if strength > 0.3:  # Only consider strong relationships
                        # Try to find an existing group for either code
                        found = False
                        for group in code_groups.values():
                            if code1 in group or code2 in group:
                                group.add(code1)
                                group.add(code2)
                                found = True
                                break
                        
                        if not found:
                            # Create a new group
                            group_id = f"theme_{len(code_groups) + 1}"
                            code_groups[group_id] = {code1, code2}
            
            # Report the identified themes
            if code_groups:
                report += "The following potential themes were identified based on code co-occurrence:\n\n"
                for i, (group_id, codes) in enumerate(code_groups.items(), 1):
                    if len(codes) >= 2:  # Only include groups with at least 2 codes
                        report += f"{i}. **Theme {i}**: {', '.join(f'`{c}`' for c in sorted(codes))}\n"
                        
                        # Add example segments that contain most of these codes
                        relevant_segments = []
                        for _, row in seg.iterrows():
                            segment_codes = set(row.get('codes', []))
                            overlap = segment_codes.intersection(codes)
                            if len(overlap) >= 2:  # At least 2 codes from this theme
                                relevant_segments.append((len(overlap), row))
                        
                        # Sort by number of matching codes (descending)
                        relevant_segments.sort(reverse=True, key=lambda x: x[0])
                        
                        # Add top example if available
                        if relevant_segments:
                            _, example = relevant_segments[0]
                            text = example.get('text', '').strip()
                            if text:
                                if len(text) > 200:
                                    text = text[:200] + "..."
                                report += f"   > *Example*: {text}\n"
            else:
                report += "No strong thematic patterns were automatically identified in the code co-occurrence. " \
                         "This could indicate that responses are diverse or that codes don't strongly co-occur.\n"
        else:
            report += "No code co-occurrence data available for theme analysis.\n"
        # Add a section for recommendations or follow-up questions
        report += "\n## 💡 Recommendations for Follow-up\n"
        
        # Generate some basic recommendations based on the analysis
        if code_freq:
            top_code = max(code_freq.items(), key=lambda x: x[1])
            report += f"- The most frequent code was `{top_code[0]}` which appeared in {top_code[1]} responses. " \
                     f"Consider exploring this topic in more depth.\n"
            
            # If there's a code that appears in the majority of responses, suggest it as a key theme
            majority_codes = [code for code, count in code_freq.items() if count / total_segments > 0.5]
            if majority_codes:
                report += f"- The code(s) {', '.join(f'`{c}`' for c in majority_codes)} appear in the majority of responses, " \
                         f"suggesting they represent key themes for this question.\n"
            
            # If there are many unique codes, suggest potential for categorization
            if len(code_freq) > 10:
                report += "- The high number of unique codes suggests that responses are diverse. " \
                         "Consider if some codes could be consolidated into broader categories.\n"
        
        # If sentiment data is available, add sentiment-based recommendations
        if sentiment_stats:
            total = sum(sentiment_stats.values())
            if total > 0:
                pos_ratio = sentiment_stats['positive'] / total
                neg_ratio = sentiment_stats['negative'] / total
                
                if pos_ratio > 0.7:
                    report += "- The overwhelmingly positive sentiment suggests this aspect is working well. " \
                             "Consider documenting best practices from these responses.\n"
                elif neg_ratio > 0.7:
                    report += "- The strongly negative sentiment indicates a significant concern. " \
                             "This area may require immediate attention and follow-up.\n"
                elif abs(pos_ratio - neg_ratio) < 0.2:  # Roughly balanced
                    report += "- The mixed sentiment suggests diverse perspectives on this topic. " \
                             "Further qualitative analysis may be needed to understand the range of experiences.\n"
        
        # Add a section for limitations
        report += "\n## ⚠️ Limitations\n"
        report += "- This analysis is based on automated coding and may not capture all nuances.\n"
        if total_segments < 10:
            report += f"- The small number of responses ({total_segments}) may limit the reliability of patterns identified.\n"
        if not code_freq:
            report += "- No codes were applied to these responses, which may indicate issues with the coding process.\n"
        # Save the final report
        output_path.write_text(report, encoding='utf-8')
        logger.info(f"Question {question_idx} report saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error generating report for question {question_idx}: {e}")
        logger.error(traceback.format_exc())

def generate_code_reports(
    seg: pd.DataFrame,
    codebook: Dict[str, int],
    code_cooccurrence: Dict,
    output_dir: pathlib.Path
) -> None:
    """Generate individual reports for each code."""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for code in codebook.keys():
            code_segments = seg[seg['codes'].apply(lambda x: code in x)]
            
            # Get co-occurring codes
            co_occurring = []
            if code in code_cooccurrence["codes"]:
                idx = code_cooccurrence["codes"].index(code)
                for j, other_code in enumerate(code_cooccurrence["codes"]):
                    if idx != j and code_cooccurrence["matrix"][idx][j] > 0:
                        co_occurring.append((other_code, code_cooccurrence["matrix"][idx][j]))
                
                co_occurring = sorted(co_occurring, key=lambda x: -x[1])[:5]
            
            # Generate markdown report
            report = f"# Code Report: {code}\n\n"
            report += f"## Usage Statistics\n"
            report += f"- **Total Applications**: {len(code_segments):,}\n"
            
            if co_occurring:
                report += "\n## Frequently Co-occurs With\n"
                for other_code, strength in co_occurring:
                    report += f"- **{other_code}**: Strength={strength:.2f}\n"
            
            # Example segments
            report += "\n## Example Segments\n"
            sample_segments = code_segments.sample(min(3, len(code_segments)))
            for _, row in sample_segments.iterrows():
                report += f"\n**Question {row['question_idx']}**\n\n"
                report += f"> {row['text']}\n\n"
            
            # Save the report
            safe_code = "".join(c if c.isalnum() else "_" for c in code)
            report_path = output_dir / f"{safe_code}.md"
            report_path.write_text(report, encoding='utf-8')
        
        logger.info(f"Code reports saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error generating code reports: {e}")
        logger.error(traceback.format_exc())

def generate_policies_with_llm(texts):
    # TODO: Implement LLM-based policy generation
    logger.warning("LLM-based policy generation not implemented yet")
    return {} #Implement your LLM-based policy generation here
    return generate_dynamic_policies(texts)

if __name__ == "__main__":
    # Configure logging for command line execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("=" * 60)
    logger.info("PIPELINE MODULE STARTING EXECUTION")
    logger.info("=" * 60)
    
    try:
        ap = argparse.ArgumentParser()
        ap.add_argument("--input", required=True)
        ap.add_argument("--out", required=True)
        ap.add_argument("--k", type=int, default=6)
        ap.add_argument("--cfg", default="./config")
        ap.add_argument("--clear-cache", action="store_true", help="Clear the persistent cache before running")
        ap.add_argument("--use-dynamic-keywords", action="store_true", 
                   help="Generate keywords dynamically from document content")
        ap.add_argument("--use-dynamic-stance-patterns", action="store_true", 
                   help="Generate stance patterns dynamically from document content")
        ap.add_argument("--base-stance-patterns", type=str, default=None,
                   help="Path to JSON file with base stance patterns")
        ap.add_argument("--no-dynamic-policies", action="store_true", help="Use only policies.yaml, do not generate dynamic policies")
        ap.add_argument("--llm-policies", action="store_true", help="Use local LLM to generate policies dynamically")
        # New flags for controlling static/dynamic sources
        ap.add_argument("--no-dynamic-keywords", action="store_true", help="Use only fixed TECH_KEYWORDS, disable dynamic keywords")
        ap.add_argument("--use-fixed-keywords", action="store_true", help="Use fixed TECH_KEYWORDS in addition to dynamic keywords")
        ap.add_argument("--no-dynamic-stances", action="store_true", help="Use only fixed STANCE_AXES, disable dynamic stances")
        ap.add_argument("--use-fixed-stances", action="store_true", help="Use fixed STANCE_AXES in addition to dynamic stances")
        
        args = ap.parse_args()
        
        # Basis-Stance-Patterns laden (falls angegeben)
        base_stance_patterns = None
        if args.base_stance_patterns and os.path.exists(args.base_stance_patterns):
            try:
                with open(args.base_stance_patterns, 'r', encoding='utf-8') as f:
                    base_stance_patterns = json.load(f)
                logger.info(f"Loaded base stance patterns from {args.base_stance_patterns}")
            except Exception as e:
                logger.warning(f"Failed to load base stance patterns: {e}")
        logger.info("Pipeline arguments parsed successfully")
        # Pipeline mit neuen Optionen ausführen
        run_pipeline(
            input_path=args.input,
            out_dir=args.out,
            k_clusters=args.k,
            cfg_dir=args.cfg,
            clear_cache_flag=args.clear_cache,
            use_dynamic_keywords=args.use_dynamic_keywords,
            use_dynamic_stance_patterns=args.use_dynamic_stance_patterns,
            base_stance_patterns=base_stance_patterns,
            no_dynamic_policies=args.no_dynamic_policies,
            llm_policies=args.llm_policies,
            no_dynamic_keywords=args.no_dynamic_keywords,
            use_fixed_keywords=args.use_fixed_keywords,
            no_dynamic_stances=args.no_dynamic_stances,
            use_fixed_stances=args.use_fixed_stances
        )
        logger.info("Pipeline execution completed successfully")
        
    except TimeoutError as e:
        logger.error(f"Pipeline execution timed out: {e}")
        logger.error("Pipeline was running for 5 hours and was automatically stopped")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline execution failed with exception: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        # Still try to save cache even on error
        try:
            save_persistent_cache()
        except:
            pass
        sys.exit(1)
