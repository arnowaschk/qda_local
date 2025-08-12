# All comments in English.
import argparse, json, os, pathlib, re, sys
import logging
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import yaml
import networkx as nx
import pickle
import hashlib
import signal
import datetime

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

# Configure logging
logger = logging.getLogger(__name__)

# Log the cache directories being used (these are set by environment variables in Dockerfile)
logger.info("Using global cache directories:")
logger.info(f"  HF_HOME: {os.environ.get('HF_HOME', 'default')}")
logger.info(f"  SPACY_DATA: {os.environ.get('SPACY_DATA', 'default')}")
logger.info(f"  TORCH_HOME: {os.environ.get('TORCH_HOME', 'default')}")

HAS_SPACY = False
HAS_SENTENCE_TRANSFORMERS = False
HAS_TRANSFORMERS = False
try:
    import spacy
    HAS_SPACY = True
    logger.info("spaCy successfully imported")
except Exception as e:
    logger.warning(f"spaCy import failed: {e}")
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
    logger.info("sentence-transformers successfully imported")
except Exception as e:
    logger.warning(f"sentence-transformers import failed: {e}")
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline as hf_pipeline
    HAS_TRANSFORMERS = True
    logger.info("transformers successfully imported")
except Exception as e:
    logger.warning(f"transformers import failed: {e}")

from util import keyword_hits, TECH_KEYWORDS, stance, apply_policies

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
    if not HAS_SPACY:
        logger.warning("spaCy not available, NER functionality disabled")
        return None
    try:
        logger.info("Loading spaCy German language model...")
        logger.info(f"Using spaCy cache directory: {os.environ.get('SPACY_DATA')}")
        
        # Try to load from cache first
        try:
            nlp = spacy.load("de_core_news_lg")
            logger.info("German NER model loaded successfully from cache")
            return nlp
        except OSError as e:
            logger.info(f"Model not found in cache, downloading... {e}")
            # Download the model
            import subprocess
            result = subprocess.run([
                sys.executable, "-m", "spacy", "download", "de_core_news_lg"
            ], capture_output=True, text=True)
            if result.returncode == 0:
                nlp = spacy.load("de_core_news_lg")
                logger.info("German NER model downloaded and loaded successfully")
                return nlp
            else:
                logger.error(f"Failed to download spaCy model: {result.stderr}")
                return None
                
    except Exception as e:
        logger.error(f"Failed to load German NER model: {e}")
        return None

def load_sentiment():
    if not HAS_TRANSFORMERS:
        logger.warning("transformers not available, sentiment analysis disabled")
        return None
    try:
        logger.info("Loading German sentiment BERT model...")
        logger.info(f"Using transformers cache directory: {os.environ.get('HF_HOME')}")
        
        model_name = "oliverguhr/german-sentiment-bert"
        
        # Check if model exists in cache
        cache_path = pathlib.Path(os.environ.get('HF_HOME')) / "models--oliverguhr--german-sentiment-bert"
        logger.info(f"HF_HOME: {os.environ.get('HF_HOME')}")
        logger.info(f"Cache path: {cache_path}")
        logger.info(f"Cache path exists: {cache_path.exists()}")
        if cache_path.exists():
            logger.info(f"Model found in cache: {cache_path}")
        else:
            logger.info(f"Model not found in cache *{cache_path}*, will download on first use")
        
        tok = AutoTokenizer.from_pretrained(model_name, local_files_only=False, cache_dir=str(pathlib.Path(os.environ.get('HF_HOME'))))
        mdl = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=False, cache_dir=str(pathlib.Path(os.environ.get('HF_HOME'))))
        pipeline = hf_pipeline("sentiment-analysis", model=mdl, tokenizer=tok)
        logger.info("German sentiment model loaded successfully")
        return pipeline
    except Exception as e:
        logger.error(f"Failed to load sentiment model: {e}")
        return None

def load_embedder():
    if not HAS_SENTENCE_TRANSFORMERS:
        logger.warning("sentence-transformers not available, embeddings disabled")
        return None
    
    # Get cache directory with fallback
    cache_dir = os.environ.get('SENTENCE_TRANSFORMERS_CACHE')
    if not cache_dir:
        # Fallback to HF_HOME if SENTENCE_TRANSFORMERS_CACHE is not set
        cache_dir = os.environ.get('HF_HOME', './cache')
        logger.warning(f"SENTENCE_TRANSFORMERS_CACHE not set, using fallback: {cache_dir}")
    
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

def segment_rows(df: pd.DataFrame) -> pd.DataFrame:
    logger.info(f"Segmenting {len(df)} rows into individual text segments")
    cols = list(df.columns)
    segments = []
    for i, row in df.iterrows():
        expert_bio = row[cols[-1]]
        for qi, col in enumerate(cols[:-1], start=1):
            text = str(row[col]).strip()
            if not text:
                continue
            segments.append({
                "set_id": int(i)+1,
                "question_idx": qi,
                "question": col,
                "text": text,
                "bio": expert_bio,
            })
    logger.info(f"Created {len(segments)} text segments from {len(df)} rows")
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
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=1, stop_words=GERMAN_STOPWORDS)
    X = vec.fit_transform(texts)
    logger.info(f"TF-IDF features computed: {X.shape}")
    return vec, X

def get_cluster_texts(X, embeddings, k: int):
    logger.info(f"Clustering {X.shape[0]} texts into {k} clusters...")
    if embeddings is not None:
        logger.info("Using embeddings for clustering")
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(embeddings)
        centers = km.cluster_centers_
        logger.info(f"Clustering completed using embeddings, cluster sizes: {[list(labels).count(i) for i in range(k)]}")
        return labels, centers, None
    else:
        logger.info("Using TF-IDF features for clustering")
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)
        logger.info(f"Clustering completed using TF-IDF, cluster sizes: {[list(labels).count(i) for i in range(k)]}")
        return labels, None, km

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
    
    # Setup timeout
    setup_timeout()
    
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
        df = pd.read_csv(input_path)
        logger.info(f"Input data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        min_text_len=7
        seg = segment_rows(df)
        seg = seg[seg["text"].str.len()>=min_text_len]
        logger.info(f"Wir verwenden {seg.shape[0]} Texte mit mindestens {min_text_len} Zeichen")
        texts = seg["text"].tolist()

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

        logger.info("Building codebook...")
        codebook = {}
        for codes in seg["codes"]:
            for c in codes:
                codebook.setdefault(c, 0); codebook[c]+=1
        codebook_sorted = {k:int(v) for k,v in sorted(codebook.items(), key=lambda x: (-x[1], x[0]))}
        logger.info(f"Codebook built: {len(codebook_sorted)} unique codes")

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
def generate_policies_with_llm(texts):
    logger.info("[LLM] Placeholder: Generating policies with local LLM (not implemented)")
    # Implement your LLM-based policy generation here
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
