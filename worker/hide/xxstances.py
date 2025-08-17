"""
Stance Analysis Module for QDA Pipeline

This module contains functions for analyzing and generating stance patterns
from document content using various NLP techniques.
"""

# All comments in English.
import logging
import re
from collections import Counter
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

# Optional imports with fallback
try:
    import textblob_de
    from textblob_de import TextBlobDE
    HAS_TEXTBLOB = True
    logger.info("textblob_de successfully imported")
except ImportError as e:
    HAS_TEXTBLOB = False
    logger.warning(f"textblob_de not available: {e}, sentiment analysis will be limited")
except Exception as e:
    HAS_TEXTBLOB = False
    logger.warning(f"textblob_de import failed: {e}, sentiment analysis will be limited")


def generate_stance_patterns_from_texts(texts: List[str], n_patterns: int = 10) -> Dict[str, List[str]]:
    """Generate stance patterns from texts using TF-IDF and KMeans clustering.
    
    This function identifies common patterns of language use that indicate different
    stances or perspectives in the input texts. It uses TF-IDF vectorization to
    extract important n-grams and KMeans clustering to group similar expressions.
    
    Args:
        texts: List of input text documents to analyze
        n_patterns: Maximum number of stance patterns to generate (default: 10)
        
    Returns:
        Dict[str, List[str]]: Dictionary mapping stance pattern names to lists of 
            characteristic n-grams. Pattern names are in the format "Cluster_N_Stance".
            
    Example:
        >>> texts = ["I love this product!", "This product is terrible"]
        >>> patterns = generate_stance_patterns_from_texts(texts)
        >>> print(patterns.keys())
        dict_keys(['Cluster_1_Stance', 'Cluster_2_Stance'])
        
    Note:
        - Uses German stop words by default
        - Processes both unigrams and bigrams (1-3 word sequences)
        - Requires scikit-learn for TF-IDF and KMeans
    """
    
    try:
        # TF-IDF Vektorisierung
        vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.8,
            stop_words='german'
        )
        
        X = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        
        # KMeans Clustering für Polarisation
        n_clusters = min(n_patterns, len(texts) // 2)
        if n_clusters < 2:
            n_clusters = 2
            
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X)
        
        # Cluster-spezifische Patterns extrahieren
        stance_patterns = {}
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_texts = [texts[i] for i in range(len(texts)) if cluster_mask[i]]
            
            if len(cluster_texts) > 0:
                # TF-IDF für diesen Cluster
                cluster_vectorizer = TfidfVectorizer(
                    max_features=100,
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.9,
                    stop_words='german'
                )
                
                cluster_X = cluster_vectorizer.fit_transform(cluster_texts)
                cluster_features = cluster_vectorizer.get_feature_names_out()
                
                # Top Features als Patterns
                cluster_scores = cluster_X.sum(axis=0).A1
                top_features = [cluster_features[i] for i in cluster_scores.argsort()[-10:][::-1]]
                
                stance_patterns[f"Cluster_{cluster_id+1}_Stance"] = top_features
        
        logger.info(f"Generated {len(stance_patterns)} stance patterns from {len(texts)} texts")
        return stance_patterns
        
    except Exception as e:
        logger.warning(f"Failed to generate stance patterns from texts: {e}")
        return {}

def generate_sentiment_based_stance_patterns(texts: List[str]) -> Dict[str, List[str]]:
    """Generate stance patterns based on sentiment analysis of input texts.
    
    Analyzes the sentiment of input texts using TextBlobDE (German language)
    and extracts characteristic terms for positive, negative, and neutral stances.
    
    Args:
        texts: List of input text documents to analyze
        
    Returns:
        Dict[str, List[str]]: Dictionary with keys 'Positive_Stance', 'Negative_Stance',
            and/or 'Neutral_Stance', each mapping to a list of characteristic terms.
            
    Example:
        >>> texts = ["Das ist großartig!", "Das ist schrecklich"]
        >>> patterns = generate_sentiment_based_stance_patterns(texts)
        >>> 'Positive_Stance' in patterns
        True
        
    Note:
        - Requires textblob_de package for German language support
        - Only includes terms with length > 3 characters
        - Returns empty dict if textblob_de is not available
    """
    
    try:
        if not HAS_TEXTBLOB:
            logger.warning("Skipping sentiment-based stance patterns generation due to missing textblob_de.")
            return {}
        
        # Sentiment für jeden Text berechnen
        text_sentiments = []
        for text in texts:
            try:
                blob = TextBlobDE(text)
                sentiment = blob.sentiment.polarity
                text_sentiments.append((text, sentiment))
            except:
                text_sentiments.append((text, 0.0))
        
        # Texte nach Sentiment gruppieren
        positive_texts = [text for text, sent in text_sentiments if sent > 0.1]
        negative_texts = [text for text, sent in text_sentiments if sent < -0.1]
        neutral_texts = [text for text, sent in text_sentiments if -0.1 <= sent <= 0.1]
        
        def extract_common_terms(text_list: List[str], top_k: int = 20) -> List[str]:
            """Extract most frequent terms from a list of texts.
            
            Args:
                text_list: List of input texts
                top_k: Number of top terms to return (default: 20)
                
            Returns:
                List[str]: Top k most frequent terms across all texts
                
            Note:
                - Converts all text to lowercase
                - Only includes words longer than 3 characters
                - Uses simple whitespace tokenization
            """
            if not text_list:
                return []
            
            all_words = []
            for text in text_list:
                words = re.findall(r'\b\w+\b', text.lower())
                all_words.extend([w for w in words if len(w) > 3])
            
            word_counts = Counter(all_words)
            return [word for word, count in word_counts.most_common(top_k)]
        
        # Stance-Patterns basierend auf Sentiment
        stance_patterns = {}
        
        if positive_texts:
            positive_terms = extract_common_terms(positive_texts)
            stance_patterns["Positive_Stance"] = positive_terms
        
        if negative_texts:
            negative_terms = extract_common_terms(negative_texts)
            stance_patterns["Negative_Stance"] = negative_terms
        
        if neutral_texts:
            neutral_terms = extract_common_terms(neutral_texts)
            stance_patterns["Neutral_Stance"] = neutral_terms
        
        logger.info(f"Generated {len(stance_patterns)} sentiment-based stance patterns")
        return stance_patterns
        
    except Exception as e:
        logger.warning(f"Failed to generate sentiment-based stance patterns: {e}")
        return {}

def generate_contextual_stance_patterns(texts: List[str]) -> Dict[str, List[str]]:
    """Generate stance patterns by analyzing context around known stance indicators.
    
    Identifies terms that frequently appear near known stance indicators (e.g.,
    "unterstützt", "gegen") to discover context-specific stance expressions.
    
    Args:
        texts: List of input text documents to analyze
        
    Returns:
        Dict[str, List[str]]: Dictionary with keys like 'Pro_Context', 'Contra_Context',
            and 'Neutral_Context', each mapping to a list of context terms.
            
    Example:
        >>> texts = ["Wir unterstützen diese Initiative", "Wir sind gegen diesen Vorschlag"]
        >>> patterns = generate_contextual_stance_patterns(texts)
        >>> 'Pro_Context' in patterns or 'Contra_Context' in patterns
        True
        
    Note:
        - Uses a predefined set of German stance indicators
        - Analyzes a 7-word window around each indicator (3 words before and after)
        - Only includes terms with length > 3 characters
    """
    
    try:
        # Bekannte Stance-Indikatoren
        stance_indicators = {
            "pro": ["unterstützt", "befürwortet", "positiv", "gut", "sinnvoll", "richtig"],
            "contra": ["gegen", "dagegen", "negativ", "schlecht", "falsch", "problematisch"],
            "neutral": ["neutral", "unparteiisch", "objektiv", "sachlich", "unvoreingenommen"]
        }
        
        contextual_patterns = {}
        
        for stance_type, indicators in stance_indicators.items():
            contextual_terms = []
            
            for text in texts:
                for indicator in indicators:
                    if indicator.lower() in text.lower():
                        # Kontext um den Indikator extrahieren
                        words = re.findall(r'\b\w+\b', text.lower())
                        indicator_pos = -1
                        
                        for i, word in enumerate(words):
                            if indicator.lower() in word:
                                indicator_pos = i
                                break
                        
                        if indicator_pos >= 0:
                            # Wörter vor und nach dem Indikator
                            start = max(0, indicator_pos - 3)
                            end = min(len(words), indicator_pos + 4)
                            context_words = words[start:end]
                            
                            # Relevante Wörter hinzufügen
                            for word in context_words:
                                if len(word) > 3 and word not in contextual_terms:
                                    contextual_terms.append(word)
            
            if contextual_terms:
                contextual_patterns[f"{stance_type.capitalize()}_Context"] = contextual_terms[:15]
        
        logger.info(f"Generated {len(contextual_patterns)} contextual stance patterns")
        return contextual_patterns
        
    except Exception as e:
        logger.warning(f"Failed to generate contextual stance patterns: {e}")
        return {}

# Note: The following functions are now imported from dynamics.py:
# - generate_comprehensive_stance_patterns
# - analyze_stances_with_dynamic_patterns
# 
# Please import them from the dynamics module instead. 