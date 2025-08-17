from typing import List, Dict, Any, Optional
import pandas as pd
from util import logger
import numpy as np
from util import clean_text
from util import apply_policies, keyword_hits
import re
from collections import Counter
import torch
import hashlib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from cache import get_cached_result
import networkx as nx
from sklearn.cluster import KMeans

from dynamics import (  
    extract_document_themes as _extract_document_themes,
)
from dynamics import GERMAN_STOPWORDS


def extract_document_themes(texts: List[str], n_themes: int = 8) -> Dict[str, List[str]]:
    from pipeline import NLP_CACHE

    """Cached version of extract_document_themes that uses the global caching mechanism.
    
    Args:
        texts: List of input texts
        n_themes: Number of themes to extract
        
    Returns:
        Dictionary mapping theme names to lists of keywords
    """
    if not texts:
        return {}
        
    # For caching, we use a deterministic hash of the texts
    texts_tuple = tuple(texts) if isinstance(texts, list) else (texts,)
    texts_str = str(texts_tuple).encode('utf-8')
    texts_hash = hashlib.md5(texts_str).hexdigest()
    cache_key = f"document_themes_{n_themes}_{texts_hash}"
    try:
        all_cache_ext_doc_keys=[x for x in NLP_CACHE if x.startswith("document_themes_")]
        logger.info(f"Found {len(all_cache_ext_doc_keys)} cached document themes")
        logger.info(f"Cache keys: {all_cache_ext_doc_keys}")
    except AssertionError as e:
        logger.error(f"Maybe Cache empty yet? Error getting cache keys: {e}")
    # Try to get from cache first
    cached = NLP_CACHE.get(cache_key)
    if cached is not None:
        logger.info(f"Using cached themes (n={n_themes})")
        return cached
        
    # If not in cache, compute and store
    logger.info(f"Computing themes (n={n_themes}) - this might take a while...")
    themes = _extract_document_themes(texts, n_themes)
    
    # Cache the results
    NLP_CACHE.set(cache_key, themes, expire=None)
    logger.info(f"Cached themes (n={n_themes}), cache key: {cache_key}")
    return themes

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
        if unique_ratio > 1.1:  # TODO XXX DISABLED was 0.9 # Mostly unique values (likely IDs or metadata)
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

def _compute_embeddings_impl(texts: List[str], embedder) -> np.ndarray:
    """Internal implementation of embedding computation without caching.
    
    Args:
        texts: List of text strings to embed
        embedder: The sentence transformer model for generating embeddings
        
    Returns:
        numpy.ndarray: Array of embeddings, one per input text
    """
    if not texts:
        return np.array([])
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        #logger.info("Using GPU for embeddings")
        embedder = embedder.to(device)
    else:
        pass #logger.info("Using CPU for embeddings")
    
    # Process in batches
    batch_size = 32
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = embedder.encode(
            batch,
            normalize_embeddings=True,
            show_progress_bar=False,
            device=device,
            convert_to_numpy=True
        )
        all_embeddings.append(batch_embeddings)
    
    return np.vstack(all_embeddings) if all_embeddings else np.array([])

def compute_embeddings(texts: List[str], embedder, batch_size: int = 32) -> Optional[np.ndarray]:
    """Compute embeddings for texts with caching using the central caching mechanism.
    
    Args:
        texts: List of text strings to embed
        embedder: The sentence transformer model for generating embeddings
        batch_size: Number of texts to process in each batch
        
    Returns:
        numpy.ndarray: Array of embeddings, one per input text, or None if failed
    """
    if not texts:
        return None
        
    if embedder is None:
        logger.warning("No embedder available, skipping embeddings")
        return None
    
    # Use the batch processing capability of get_cached_result
    embeddings, hit_count = get_cached_result(
        texts=texts,
        function_name="embedding",
        func=lambda batch: _compute_embeddings_impl(batch, embedder),
        batch_size=batch_size
    )
    
    logger.info(f"Computed embeddings for {len(texts)} texts with {hit_count} hits")
    if embeddings and all(e is not None for e in embeddings):
        return np.stack(embeddings)
    return None
    

def tfidf_features(texts: List[str]) -> tuple:
    """Compute TF-IDF features for a list of texts.
    
    Args:
        texts: List of text strings to process
        
    Returns:
        tuple: (vectorizer, X) where vectorizer is the fitted TfidfVectorizer 
              and X is the feature matrix
    """
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

def get_cluster_texts(X, embeddings, k: int) -> tuple:
    """Cluster texts using either embeddings or TF-IDF features.
    
    Args:
        X: TF-IDF feature matrix (can be None if using embeddings)
        embeddings: Text embeddings (can be None if using TF-IDF)
        k: Number of clusters to generate
        
    Returns:
        tuple: (labels, centers, km) where:
            - labels: Cluster labels for each text
            - centers: Cluster centers (None if using TF-IDF)
            - km: Fitted KMeans model (None if using embeddings)
    """
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
    """Generate labels and metadata for text clusters based on keywords.
    
    Args:
        texts: List of text strings in the clusters
        labels: Cluster labels for each text
        k: Number of clusters
        keyword_dict: Optional dictionary of keywords for labeling clusters
        
    Returns:
        Dict mapping cluster indices to metadata including:
            - label: Generated label for the cluster
            - size: Number of texts in the cluster
            - keywords: Top keywords for the cluster
            - texts: List of texts in the cluster
            - avg_length: Average text length in the cluster
    """
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
    """Apply coding rules and keyword matching to a text.
    
    Args:
        text: Text to analyze
        policies: Dictionary of coding policies/rules
        keyword_fallback: Whether to use keyword matching as fallback
        strict: If True, only return rules that strictly match
        keyword_dict: Optional dictionary of keywords for fallback matching
        
    Returns:
        List of matching rule codes (or ['unspecified'] if no matches)
    """
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
    """Split text into sentences using basic punctuation rules.
    
    Args:
        text: Input text to split
        
    Returns:
        List of sentence strings with whitespace stripped
    """
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]

def summarize_textrank(texts: List[str], max_sentences: int = 3) -> str:
    """Generate a summary of texts using the TextRank algorithm.
    
    Args:
        texts: List of text strings to summarize
        max_sentences: Maximum number of sentences in the summary
        
    Returns:
        String containing the generated summary
    """
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

def build_dynamic_keyword_dict(dyn_kw: dict) -> Dict[str, List[str]]:
    """
    Build a dictionary of dynamic keywords for labeling and coding.
    
    Args:
        dyn_kw: Dictionary containing dynamic keywords
        
    Returns:
        Dictionary mapping keyword categories to lists of keywords
    """
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

def generate_dynamic_policies(texts: List[str], base_policies: Optional[Dict] = None) -> Dict[str, Any]:
    """Generiert Policies dynamisch aus dem Dokumentinhalt"""
    
    logger.info(f"[START] generate_dynamic_policies: {len(texts)} texts")

    # 1. Hauptthemen extrahieren
    themes = extract_document_themes(texts)
    
    # 2. Häufige Begriffe und Konzepte identifizieren
    # Alle Wörter sammeln
    all_words = []
    stopset = set(GERMAN_STOPWORDS)
    for text in texts:
        words = re.findall(r'\b\w+\b', text.lower())
        all_words.extend([w for w in words if len(w) > 2 and w not in stopset and not w.isdigit()])
    
    # Häufigste Begriffe
    word_counts = Counter(all_words)
    top_terms = [word for word, count in word_counts.most_common(50)]
    logger.info(str([word for word, count in word_counts.most_common(10)])+" most common 10")

    # 3. Policy-Codes aus Themen generieren
    dynamic_codes = []
    used_names: Dict[str, int] = {}

    def slugify(token: str) -> str:
        token = token.strip().lower()
        token = re.sub(r"[^a-z0-9äöüß]+", "_", token)
        token = re.sub(r"_+", "_", token).strip("_")
        return token or "x"

    def unique_name(base: str) -> str:
        if base not in used_names:
            used_names[base] = 1
            return base
        used_names[base] += 1
        return f"{base}_{used_names[base]}"
 
    # Themen-basierte Codes
    for theme_name, theme_words in themes.items():
        # Stopwords aus Theme-Wörtern entfernen
        filtered_theme_words = [w for w in theme_words if isinstance(w, str) and w.lower() not in stopset and len(w) > 2]
        # Nimm die 1-2 signifikantesten Wörter als Basis
        base_tokens = [slugify(w) for w in filtered_theme_words[:2] if w]
        base_tokens = [t for t in base_tokens if t]
        base_part = "_".join(base_tokens) if base_tokens else slugify(theme_name)
        # Menschlich lesbarer Anzeigename
        disp_tokens = [str(w).strip().capitalize() for w in filtered_theme_words[:2] if isinstance(w, str)]
        display_name = (" ".join(disp_tokens) if disp_tokens else str(theme_name).strip()) + " Core"
        name = unique_name(f"{base_part}_Core")
        dynamic_codes.append({
            "name": name,
            "display_name": display_name,
            "any": filtered_theme_words[:5]  # Top 5 Wörter pro Thema (ohne Stopwörter)
        })
        logger.info("Dynamic Codes Core: "+str(dynamic_codes[-1]))

    # Häufige Begriffe als Codes
    for i in range(0, len(top_terms), 5):
        batch = top_terms[i:i+5]
        # Nimm 1-2 führende Terme als Basis
        base_tokens = [slugify(w) for w in batch[:2] if w]
        base_tokens = [t for t in base_tokens if t]
        base_part = "_".join(base_tokens) if base_tokens else f"term_{i//5+1}"
        disp_tokens = [str(w).strip().capitalize() for w in batch[:2] if isinstance(w, str)]
        display_name = (" ".join(disp_tokens) if disp_tokens else f"Terms {i//5+1}") + " frequent term"
        name = unique_name(f"{base_part}_frequent_term")
        dynamic_codes.append({
            "name": name,
            "display_name": display_name,
            "any": batch
        })
        logger.info("Dynamic Codes Freq: "+str(dynamic_codes[-1]))
    
    # 4. Mit Basis-Policies kombinieren (falls vorhanden)
    if base_policies and base_policies.get("codes"):
        dynamic_codes.extend(base_policies["codes"])
    
    # 5. Dynamische Policies erstellen
    dynamic_policies = {
        "policy": {
            "mode": "augment",
            "generation_method": "dynamic",
            "themes_extracted": len(themes)
        },
        "codes": dynamic_codes,
        "themes": themes,
        "top_terms": top_terms[:20]
    }
    
    logger.info(f"Generated {len(dynamic_codes)} dynamic policy codes from {len(themes)} themes")
    logger.info(f"[END] generate_dynamic_policies: returning {len(dynamic_policies.get('codes', []))} codes")
    return dynamic_policies



def generate_policies_with_llm(texts: List[str]) -> Dict[str, Any]:
    """Generate coding policies using a language model.
    
    Note: This is a placeholder function that currently returns an empty dict.
    
    Args:
        texts: List of text strings to analyze for policy generation
        
    Returns:
        Dictionary containing generated policies (currently empty)
        
    TODO: Implement actual LLM-based policy generation
    """
    logger.warning("LLM-based policy generation not implemented yet")
    return {}
    return generate_dynamic_policies(texts)

