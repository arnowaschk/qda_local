# All comments in English.
import argparse, json, os, sys, traceback
import logging
from pathlib import Path
from typing import Dict
import pandas as pd
import os, re
from diskcache import FanoutCache

from analyze import generate_policies_with_llm, generate_dynamic_policies
from dynamics import hybrid_keyword_generation

from util import (
    TimeoutError, setup_timeout, DirectLogger
)
from util import create_safe_dirname
# Global timeout configuration (5 hours = 18000 seconds)
TIMEOUT_SECONDS = 18000

# Simple direct logging to stderr for immediate output
import sys

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



# Initialize DiskCache for persistent storage
CACHE_DIR = os.environ.get('CACHE_DIR', '/cache')
os.makedirs(CACHE_DIR, exist_ok=True)
global CACHE_PATH
CACHE_PATH = os.path.join(CACHE_DIR, 'nlp_cache.dc')

# Global cache for NLP results with persistence
global NLP_CACHE
NLP_CACHE = FanoutCache(
    directory=CACHE_PATH,
    timeout=1,
    size_limit=10**11,  # 100GB size limit
    shards=64,         # More shards for better concurrency
    eviction_policy='least-recently-used'
)

from cache import (
    load_persistent_cache, 
    save_persistent_cache,
)

from loaders import (
    load_policies, load_embedder,
    load_csv_with_fallback,
)

from analyze import (
    segment_rows, _compute_embeddings_impl, compute_embeddings,
    tfidf_features, get_cluster_texts, label_clusters, rule_codes,
    split_sentences, summarize_textrank, build_dynamic_keyword_dict
)

from networks import (
    analyze_code_cooccurrence, generate_code_network, generate_word_cloud
)

from reports import generate_structured_report, generate_overall_summary
from reports import generate_question_report, generate_code_reports
from reports import generate_reports

def analyze_data(df: pd.DataFrame, output_dir: Path, report_name: str = None, 
                k_clusters: int = 6, cfg_dir: str = "./config",
                use_dynamic_keywords: bool = True,
                use_dynamic_stance_patterns: bool = True,
                stances_biased_by_keywords: bool = True,
                no_dynamic_policies: bool = False,
                llm_policies: bool = False,
                no_dynamic_keywords: bool = False,
                no_dynamic_stances: bool = False) -> dict:
    """Core analysis function that processes either full dataset or a single column.
    
    Args:
        df: Input DataFrame to analyze
        output_dir: Directory to save outputs
        report_name: Optional name for the report
        k_clusters: Number of clusters to generate
        cfg_dir: Directory containing configuration files
        use_dynamic_keywords: Whether to generate dynamic keywords
        use_dynamic_stance_patterns: Whether to use dynamic stance patterns
        stances_biased_by_keywords: If True, bias stance patterns with generated keywords (comprehensive patterns)
        no_dynamic_policies: If True, skip dynamic policy generation
        no_dynamic_keywords: If True, skip dynamic keyword generation
        no_dynamic_stances: If True, skip dynamic stance extraction
        
    Returns:
        Dictionary containing analysis results and metadata
    """
    try:
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Segment the data
        seg = segment_rows(df)
        if len(seg) == 0:
            logger.warning(f"No valid text segments found in {'column ' + report_name if report_name else 'dataset'}")
            return {}
        
        # Extract texts for analysis
        texts = [str(t) for t in seg['text'].tolist() if pd.notna(t) and str(t).strip()]
        if not texts:
            logger.warning(f"No valid text found in {'column ' + report_name if report_name else 'dataset'}")
            return {}
        
        # Load NLP models
        logger.info("Loading NLP models...")
        #nlp = load_ner(HAS_SPACY)
        #senti = load_sentiment(HAS_TRANSFORMERS)
        embedder = load_embedder(HAS_SENTENCE_TRANSFORMERS)

        # Policy handling logic
        if no_dynamic_policies:
            logger.info("[POLICY] Using only policies.yaml (no dynamic policies)")
            policies = load_policies(Path(cfg_dir))
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
        logger.info(f"Policy mode: {mode} (strict: {strict}, keyword_fallback: {keyword_fallback}")

        # Compute features and cluster
        logger.info("Computing text features...")
        vec, X = tfidf_features(texts)
        embeddings = compute_embeddings(texts, embedder) if embedder else None

            # Determine number of clusters
        k = min(max(2, len(texts)//4), max(2, k_clusters))
        logger.info(f"Clustering {len(texts)} texts into {k} clusters...")
        
            # Perform clustering
        labels, centers, km = get_cluster_texts(X, embeddings, k)
        seg["cluster"] = labels

        # Generate dynamic content if enabled
        dynamic_policies = policies if not no_dynamic_policies else {}
        dynamic_keywords = hybrid_keyword_generation(texts) if not no_dynamic_keywords else {}

        # Assign codes to each segment (ensure 'codes' column exists)
        try:
            kw_dict = build_dynamic_keyword_dict(dynamic_keywords) if dynamic_keywords else {}
        except Exception:
            kw_dict = {}
        try:
            seg['codes'] = [
                rule_codes(text=str(t), policies=policies, keyword_fallback=keyword_fallback, strict=strict, keyword_dict=kw_dict if keyword_fallback else None)
                for t in seg['text']
            ]
        except Exception as e:
            logger.warning(f"Failed to assign codes to segments: {e}")
            seg['codes'] = [[] for _ in range(len(seg))]

        # Process stances if enabled
        stance_patterns_used = {}
        if use_dynamic_stance_patterns and not no_dynamic_stances:
            try:
                if stances_biased_by_keywords:
                    # Map dynamic keywords to base stance patterns dict[str, list[str]]
                    base_patterns = {}
                    try:
                        tech_kw = (dynamic_keywords.get("Technical") or {}).get("keywords") or []
                        if isinstance(tech_kw, list) and tech_kw:
                            base_patterns["Technical"] = list(tech_kw)
                    except Exception:
                        pass
                    try:
                        thematic = (dynamic_keywords.get("Thematic") or {}).get("keywords") or {}
                        if isinstance(thematic, dict):
                            base_patterns["Thematic"] = sorted({w for lst in thematic.values() for w in (lst or [])})
                    except Exception:
                        pass
                    try:
                        cluster_spec = (dynamic_keywords.get("Cluster_Specific") or {}).get("keywords") or {}
                        if isinstance(cluster_spec, dict):
                            base_patterns["Cluster_Specific"] = sorted({w for lst in cluster_spec.values() for w in (lst or [])})
                    except Exception:
                        pass
                    try:
                        base_kw = (dynamic_keywords.get("Base_Keywords") or {}).get("keywords")
                        if isinstance(base_kw, dict):
                            base_patterns["Base_Keywords"] = sorted({w for lst in base_kw.values() for w in (lst or [])})
                        elif isinstance(base_kw, list):
                            if base_kw:
                                base_patterns["Base_Keywords"] = list(base_kw)
                    except Exception:
                        pass

                    stance_patterns_used = dynamics_generate_comprehensive_stance_patterns(
                        texts,
                        base_stance_patterns=base_patterns if base_patterns else None
                    )
                else:
                    # Purely text-driven stance patterns (no keyword bias)
                    stance_patterns_used = dynamics_generate_stance_patterns_from_texts(texts)
            except Exception as e:
                logger.warning(f"Could not generate stance patterns: {e}")
    
        # Generate cluster summaries and global summary
        cluster_summaries = {}
        for cid in range(k):
            cluster_texts = [t for t, l in zip(texts, labels) if l == cid]
            try:
                summary = summarize_textrank(cluster_texts, max_sentences=3)
                cluster_summaries[cid] = summary
            except Exception as e:
                logger.warning(f"Failed to generate summary for cluster {cid}: {e}")
                cluster_summaries[cid] = f"Summary generation failed: {e}"
        
        try:
            global_summary = summarize_textrank(texts, max_sentences=12)
        except Exception as e:
            logger.warning(f"Failed to generate global summary: {e}")
            global_summary = f"Global summary generation failed: {e}"
            
        # Generate and save reports
        results = {
            'segment_df': seg,
            'policies': policies,
            'dynamic_policies': dynamic_policies,
            'dynamic_keywords': dynamic_keywords,
            'stance_patterns': stance_patterns_used,
            'cluster_summaries': cluster_summaries,
            'global_summary': global_summary,
            'output_dir': output_dir,
            'vectorizer': vec,
            'embeddings': embeddings,
            'cluster_model': km,
            'cluster_centers': centers
        }
        
        return results

    except Exception as e:
        logger.error(f"Error in analyze_data: {e}")
        logger.error(traceback.format_exc())
        raise

def _build_question_texts(seg: pd.DataFrame) -> Dict[int, str]:
    """Build a mapping of question indices to question texts from a segment DataFrame.
    
    Processes the input DataFrame to extract unique question index-text pairs.
    Handles missing or invalid data gracefully by returning an empty dict.
    
    Args:
        seg: Input DataFrame containing 'question_idx' and 'question' columns
        
    Returns:
        Dict[int, str]: Mapping of question indices to question texts
        
    Example:
        >>> df = pd.DataFrame({
        ...     'question_idx': [1, 1, 2, 2],
        ...     'question': ['Q1', 'Q1', 'Q2', 'Q2']
        ... })
        >>> _build_question_texts(df)
        {1: 'Q1', 2: 'Q2'}
        
    Note:
        - Returns an empty dict if input is None, empty, or missing required columns
        - Drops duplicate question indices, keeping the first occurrence
        - Converts indices to integers and texts to strings
    """
    try:
        if seg is None or len(seg) == 0:
            return {}
        if 'question_idx' in seg.columns and 'question' in seg.columns:
            # Drop duplicates and NaNs, ensure ints and strings
            pairs = (
                seg[['question_idx', 'question']]
                .dropna()
                .drop_duplicates()
                .sort_values('question_idx')
            )
            return {int(row['question_idx']): str(row['question']) for _, row in pairs.iterrows()}
    except Exception as e:
        logger.warning(f"Failed to build question_texts: {e}")
    return {}

def run_pipeline(input_path: str, out_dir: str, k_clusters: int = 6, cfg_dir: str = "./config", 
                clear_cache_flag: bool = False,
                use_dynamic_keywords: bool = True,
                use_dynamic_stance_patterns: bool = True,
                stances_biased_by_keywords: bool = True,
                base_stance_patterns: dict = None,
                no_dynamic_policies: bool = False,
                llm_policies: bool = False,
                no_dynamic_keywords: bool = False,
                use_fixed_keywords: bool = False,
                no_dynamic_stances: bool = False,
                use_fixed_stances: bool = False):
    """Pipeline with optional dynamic keyword and stance pattern generation.
    
    Args:
        input_path: Path to input CSV file
        out_dir: Output directory for results
        k_clusters: Number of clusters to generate
        cfg_dir: Directory containing configuration files
        clear_cache_flag: If True, clear the cache before starting
        use_dynamic_keywords: Whether to generate dynamic keywords
        use_dynamic_stance_patterns: Whether to use dynamic stance patterns
        base_stance_patterns: Base stance patterns to use
        no_dynamic_policies: If True, skip dynamic policy generation
        llm_policies: If True, use LLM for policy generation
        no_dynamic_keywords: If True, skip dynamic keyword generation
        use_fixed_keywords: If True, use fixed keywords from config
        no_dynamic_stances: If True, skip dynamic stance extraction
        use_fixed_stances: If True, use fixed stances from config
        
    Returns:
        Path to the output directory containing all results
    """
    logger.info(f"Starting QDA pipeline - input: {input_path}, output: {out_dir}, clusters: {k_clusters}")
    
    # Setup timeout and cache
    setup_timeout()
    
    try:
        # Handle cache management
        if clear_cache_flag:
            global NLP_CACHE, CACHE_SAVE_COUNTER
            NLP_CACHE = {}
            CACHE_SAVE_COUNTER = 0
            logger.info("Cache cleared as requested")
        else:
            load_persistent_cache()
            
        logger.info(f"Using cache with {len(NLP_CACHE)} existing entries")
        
        # Setup output directories
        outp = Path(out_dir)
        outp.mkdir(parents=True, exist_ok=True)
        (outp/"exports").mkdir(parents=True, exist_ok=True)
        
        # Load input data
        logger.info(f"Loading input data from: {input_path}")
        df = load_csv_with_fallback(input_path)
        logger.info(f"Input data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Analyze full dataset first
        logger.info("Analyzing full dataset...")
        full_results = analyze_data(
            df=df,
            output_dir=outp,
            k_clusters=k_clusters,
            cfg_dir=cfg_dir,
            use_dynamic_keywords=use_dynamic_keywords,
            use_dynamic_stance_patterns=use_dynamic_stance_patterns,
            stances_biased_by_keywords=stances_biased_by_keywords,
            no_dynamic_policies=no_dynamic_policies,
            llm_policies=llm_policies,
            no_dynamic_keywords=no_dynamic_keywords,
            no_dynamic_stances=no_dynamic_stances
        )
        
        if not full_results:
            raise ValueError("No valid text segments found in the input data")
        
        # Generate reports for full dataset
        full_question_texts = _build_question_texts(full_results.get('segment_df'))
        generate_reports(
            segment_df=full_results['segment_df'],
            output_dir=outp,
            dynamic_policies=full_results.get('dynamic_policies', {}),
            dynamic_keywords=full_results.get('dynamic_keywords', {}),
            used_policies=set(),
            used_keywords=set(),
            used_stances=set(),
            codebook=full_results.get('policies', {}),
            cluster_info=full_results.get('cluster_summaries', {}),
            cluster_summaries=full_results.get('cluster_summaries', {}),
            global_summary=full_results.get('global_summary', ''),
            input_path=input_path,
            k_clusters=k_clusters,
            policies=full_results.get('policies', {}),
            stance_patterns=full_results.get('stance_patterns', {}),
            question_texts=full_question_texts,
        )
        
        # Process individual columns if there are multiple columns
        if len(df.columns) > 1:
            logger.info("Processing individual columns...")
            for col_idx, col_name in enumerate(df.columns):
                # Skip non-text columns or empty columns
                if pd.api.types.is_numeric_dtype(df[col_name]) or df[col_name].isna().all():
                    continue
                
                logger.info(f"Analyzing column: {col_name}")
                
                # Create column-specific output directory
                col_dirname = create_safe_dirname(str(col_name), col_idx)
                col_outp = outp / col_dirname
                
                try:
                    # Create a single-column DataFrame for analysis
                    col_df = df[[col_name]].rename(columns={col_name: 'text'})
                
                    # Analyze the column
                    col_results = analyze_data(
                        df=col_df,
                        output_dir=col_outp,
                    report_name=col_name,
                    k_clusters=k_clusters,
                    cfg_dir=cfg_dir,
                    use_dynamic_keywords=use_dynamic_keywords,
                    use_dynamic_stance_patterns=use_dynamic_stance_patterns,
                    stances_biased_by_keywords=stances_biased_by_keywords,
                    no_dynamic_policies=no_dynamic_policies,
                    llm_policies=llm_policies,
                    no_dynamic_keywords=no_dynamic_keywords,
                    no_dynamic_stances=no_dynamic_stances
                )
                
                    if not col_results:
                        logger.warning(f"No valid text segments found in column: {col_name}")
                        continue
                    
                    # Generate reports for this column
                    col_question_texts = _build_question_texts(col_results.get('segment_df'))
                    generate_reports(
                        segment_df=col_results['segment_df'],
                        output_dir=col_outp,
                        report_name=col_name,
                        dynamic_policies=col_results.get('dynamic_policies', {}),
                        dynamic_keywords=col_results.get('dynamic_keywords', {}),
                        used_policies=set(),
                        used_keywords=set(),
                        used_stances=set(),
                        codebook=col_results.get('policies', {}),
                        cluster_info=col_results.get('cluster_summaries', {}),
                        cluster_summaries=col_results.get('cluster_summaries', {}),
                        global_summary=col_results.get('global_summary', ''),
                        input_path=input_path,
                        k_clusters=k_clusters,
                        policies=col_results.get('policies', {}),
                        stance_patterns=col_results.get('stance_patterns', {}),
                        question_texts=col_question_texts,
                )
                
                except Exception as e:
                    logger.error(f"Error processing column {col_name}: {e}")
                    logger.error(traceback.format_exc())
                    continue
        
        logger.info("Pipeline completed successfully")
        return outp
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        logger.error(traceback.format_exc())
        raise
    finally:
        # Always save the cache before exiting
        try:
            save_persistent_cache()
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
        logger.info("All analyses and reports completed successfully")
        
        # Save cache before exiting
        save_persistent_cache()
        
        # Return the output directory path
        return outp


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
        ap.add_argument("--no-stance-bias-by-keywords", action="store_true",
                   help="Disable biasing stance patterns with generated keywords (enabled by default)")
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
            stances_biased_by_keywords=not args.no_stance_bias_by_keywords,
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
