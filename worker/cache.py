from typing import Union, List, Tuple, Any, Optional, Dict
import hashlib
import os
from util import logger
import traceback
from util import stance


CACHE_DIR = os.environ.get('CACHE_DIR', '/cache')
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_PATH = os.path.join(CACHE_DIR, 'nlp_cache.dc')

def load_persistent_cache() -> None:
    """Initialize the persistent cache system.
    
    Note:
        This is now a no-op since DiskCache handles persistence automatically.
        Kept for backward compatibility with existing code.
        
    Side Effects:
        - Creates cache directory if it doesn't exist
        - Initializes logging for cache operations
    """
    logger.info(f"Initialized DiskCache at {CACHE_PATH}")

def save_persistent_cache() -> None:
    """Synchronize in-memory cache with disk storage.
    
    Note:
        This is a no-op for FanoutCache as it handles persistence automatically.
        Kept for backward compatibility.
        
    Side Effects:
        - May write cache data to disk
        - Logs cache synchronization status
    """
    logger.info("Cache persistence is handled automatically by FanoutCache")

def get_cached_result(texts: Union[str, List[str]], function_name: str, func, batch_size: int = 32, 
    progress: bool = False, *args, **kwargs) -> Tuple[Union[Any, List[Any]], bool]:
    """Get cached results or compute and cache them in batches.
    
    Args:
        texts: Single text string or list of text strings to process
        function_name: Name of the function for cache key generation
        func: Function to call for computation if not in cache
        batch_size: Number of texts to process in each batch
        progress: Whether to show progress logs
        *args, **kwargs: Additional arguments to pass to the function
        
    Returns:
        Tuple of (results, hit_rate) where results is either a single result or a list of results,
        and hit_rate is the fraction of cache hits (0.0 to 1.0)
    """
    from pipeline import NLP_CACHE    
    single_input = isinstance(texts, str)
    if single_input:
        texts = [texts]
        
    results = []
    total_hits = 0
    
    if progress:
        logger.info(f"Processing {len(texts)} texts with batch processing... (Batch size: {batch_size})")
    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_hashes = [hashlib.md5(text.encode('utf-8')).hexdigest() for text in batch]
        cache_keys = [f"{function_name}:{text_hash}" for text_hash in batch_hashes]
        
        # Try to get from cache
        batch_results = []
        batch_to_compute = []
        batch_indices = []
        
        for idx, (cache_key, text) in enumerate(zip(cache_keys, batch)):
            cached = NLP_CACHE.get(cache_key)
            if cached is not None:
                batch_results.append((idx, cached))
                total_hits += 1
            else:
                batch_to_compute.append(text)
                batch_indices.append(idx)
        
        if progress:
            logger.info(f"Batch {i//batch_size}: {len(batch_to_compute)} texts to compute")
        # Compute results for texts not in cache
        if batch_to_compute:
            if progress:
                logger.info(f"Batch to compute:{batch_to_compute} /Batch")
            try:
                if progress:
                    logger.info(f"Computing results for batch {i//batch_size}")
                computed_results = func(batch_to_compute, *args, **kwargs) if batch_to_compute else []
                if progress:
                    logger.info(f"Computed results for batch {i//batch_size}")
                # Cache the computed results
                for idx, result, text_hash in zip(batch_indices, computed_results, [batch_hashes[i] for i in batch_indices]):
                    if result is not None:  # Only cache non-None results
                        NLP_CACHE.set(f"{function_name}:{text_hash}", result, expire=None)
                    batch_results.append((idx, result))
            except Exception as e:
                logger.error(f"Error in get_cached_result batch {i//batch_size} for {function_name}: {e}")
                logger.error(traceback.format_exc())
                # Fallback to neutral for failed computations
                for idx in batch_indices:
                    if function_name == "sentiment" or function_name == "stance":
                        batch_results.append((idx, "neutral"))
                    else:
                        batch_results.append((idx, None))
        
        # Sort results by original index and extract just the values
        batch_results.sort(key=lambda x: x[0])
        results.extend([result for _, result in batch_results])
    logger.info(f"Cache hits: {total_hits} von {len(texts)} für {function_name}")
    return results, total_hits

def process_texts_in_batch(texts: List[str], function_name: str, func, *args, **kwargs) -> List[Any]:
    """Process a list of texts in batches with caching support.
    
    Args:
        texts: List of text strings to process
        function_name: Name of the function for cache key generation
        func: Callable that processes a batch of texts
        *args: Positional arguments to pass to func
        **kwargs: Keyword arguments including:
            - batch_size: Number of texts to process per batch (default: 32)
            Additional kwargs are passed to func
            
    Returns:
        List of processed results in the same order as input texts
        
    Note:
        - Automatically handles caching of results
        - Processes texts in batches for memory efficiency
        - Periodically syncs cache during long operations
    """
    from pipeline import NLP_CACHE
    results = []
    batch_size = kwargs.pop('batch_size', 32)
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_results = []
        
        # First check cache for all texts in batch
        cache_keys = []
        texts_to_process = []
        
        for idx, text in enumerate(batch):
            text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
            cache_key = f"{function_name}:{text_hash}"
            cache_keys.append(cache_key)
            
            # Try to get from cache
            cached = NLP_CACHE.get(cache_key)
            if cached is not None:
                batch_results.append(cached)
            else:
                texts_to_process.append((idx, text))
        
        # Process texts not in cache
        if texts_to_process:
            # Prepare batch for processing
            process_texts = [text for _, text in texts_to_process]
            
            # Process batch
            processed = func(process_texts, *args, **kwargs)
            
            # Store results in cache and collect for output
            for (idx, text), result in zip(texts_to_process, processed):
                text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
                cache_key = f"{function_name}:{text_hash}"
                if result is not None:  # Only cache non-None results
                    NLP_CACHE.set(cache_key, result, expire=None)
                batch_results.insert(idx, result)
        
        results.extend(batch_results)
        
        # Periodically sync cache during long operations
        if i > 0 and i % (batch_size * 10) == 0:
            NLP_CACHE.sync()
            
    return results

def process_texts_batch(
    texts: List[str],
    nlp: Any,
    senti: Any,
    policies: Dict[str, Any],
    keyword_fallback: bool,
    strict: bool,
    keyword_dict: Optional[Dict[str, List[str]]] = None,
    use_static_stances: bool = True
) -> Tuple[List[str], List[List[str]], List[List[str]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Process texts through multiple NLP pipelines with batch processing and caching.
    
    Args:
        texts: List of text strings to process
        nlp: Initialized spaCy NLP pipeline for NER processing
        senti: Initialized sentiment analysis model
        policies: Dictionary containing policy definitions for rule-based coding
        keyword_fallback: If True, fall back to keyword matching when other methods fail
        strict: If True, use strict matching for rule application
        keyword_dict: Optional dictionary of keywords for rule-based coding
        use_static_stances: If True, compute and include static stance analysis
        
    Returns:
        Tuple containing:
            - List of sentiment labels (one per text)
            - List of person entities (list per text)
            - List of organization entities (list per text)
            - List of stance analysis results (dict per text)
            - List of applied rule codes (dict per text)
            
    Note:
        - Processes texts in optimized batch sizes for each NLP task
        - Uses caching to avoid redundant computations
        - Handles memory management for large text collections
    """
    from pipeline import NLP_CACHE
    batch_size_sentiment = 32  # Good for transformer models
    batch_size_stance = 32     # Similar to sentiment
    batch_size_rule_codes = 64 # Lighter processing
    batch_size_ner = 16         # More memory-intensive
    
    logger.info(f"Processing {len(texts)} texts with batch processing...")
    
    # Initialize result lists
    n = len(texts)
    sentiments = ["neutral"] * n
    persons = [[] for _ in range(n)]
    orgs = [[] for _ in range(n)]
    stances = [{} for _ in range(n)]
    codes_all = [{} for _ in range(n)]
    
    # Process sentiment in batches
    if senti:
        logger.info("Processing sentiment in batches...")
        sentiment_results, _ = get_cached_result(
            texts,
            function_name="sentiment",
            func=lambda batch: [senti(t)[0]["label"] for t in batch],
            batch_size=batch_size_sentiment
        )
        for i, sentiment in enumerate(sentiment_results):
            sentiments[i] = sentiment
    
    # Process stances in batches if requested
    if use_static_stances:
        logger.info("Processing stances in batches...")
        stance_results, _ = get_cached_result(
            texts,
            function_name="stance",
            func=lambda batch: [stance(t) for t in batch],
            batch_size=batch_size_stance
        )
        for i, stance_result in enumerate(stance_results):
            stances[i] = stance_result
    
    # Process rule codes in batches
    logger.info("Processing rule codes in batches...")
    # Lazy import to avoid circular import (analyze.py imports cache.py)
    from analyze import rule_codes
    rule_code_results, _ = get_cached_result(
        texts,
        function_name="rule_codes",
        func=lambda batch: [rule_codes(t, policies, keyword_fallback, strict, keyword_dict) for t in batch],
        batch_size=batch_size_rule_codes
    )
    for i, codes in enumerate(rule_code_results):
        if codes:
            codes_all[i] = codes
    
    # Process NER in batches if nlp is available
    if nlp:
        logger.info("Processing NER in batches...")
        
        def process_ner_batch(batch):
            # Disable unnecessary pipeline components for NER
            with nlp.select_pipes(enable=["ner"]):
                results = []
                for doc in nlp.pipe(batch, n_process=1):  # n_process=1 to avoid memory issues
                    # Extract only the entity information we need
                    entities = []
                    for ent in doc.ents:
                        if ent.label_ in ("PER", "PERSON", "ORG"):
                            entities.append({
                                'text': ent.text,
                                'label_': ent.label_,
                                'start': ent.start_char,
                                'end': ent.end_char
                            })
                    results.append(entities)
                return results
        
        # Get cached or processed results
        ner_results, _ = get_cached_result(
            texts,
            function_name="ner_entities",  # Changed cache key to indicate new format
            func=process_ner_batch,
            batch_size=min(10, batch_size_ner),  # Smaller batch size for memory safety
            progress=True
        )
        
        # Process the cached results
        for i, entities in enumerate(ner_results):
            if entities:
                persons[i] = [e['text'] for e in entities if e['label_'] in ("PER", "PERSON")]
                orgs[i] = [e['text'] for e in entities if e['label_'] == "ORG"]
    
    logger.info(f"Text processing completed. Cache size: {len(NLP_CACHE)}")
    
    # Save cache after processing
    if len(NLP_CACHE) > 0:
        logger.info(f"Saving cache after batch processing: {len(NLP_CACHE)} entries")
        save_persistent_cache()
    
    return sentiments, persons, orgs, stances, codes_all

def clear_ner_cache() -> int:
    """Remove all NER-related entries from the cache.
    
    Returns:
        Number of cache entries removed
        
    Side Effects:
        - Modifies the global NLP_CACHE by removing NER entries
        - Logs the number of entries cleared
    """
    from pipeline import NLP_CACHE    
    if 'NLP_CACHE' in globals():
        keys_to_remove = [k for k in NLP_CACHE if str(k).startswith('ner_entities:')]
        for key in keys_to_remove:
            del NLP_CACHE[key]
        logger.info(f"Cleared {len(keys_to_remove)} NER cache entries")
        return len(keys_to_remove)
    return 0

