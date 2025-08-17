"""
Dynamic Content Generation Module for QDA Pipeline

This module contains functions for dynamically generating:
- Technical keywords from document content
- Topics and keywords using LDA
- Cluster-specific keyword lists
- Hybrid keyword generation combining static and dynamic
- Document themes extraction
- Dynamic policy generation
- Enhanced stance pattern analysis
"""

# All comments in English.
import logging
import hashlib
import re
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True
)
logger = logging.getLogger(__name__)

# German stopwords support (spaCy -> NLTK -> default list)
try:
    from spacy.lang.de.stop_words import STOP_WORDS as SPACY_DE_STOPWORDS
    _SPACY_DE_STOPWORDS = set(SPACY_DE_STOPWORDS)
except Exception:
    _SPACY_DE_STOPWORDS = None

try:
    from nltk.corpus import stopwords as _nltk_stopwords
    try:
        _NLTK_DE_STOPWORDS = set(_nltk_stopwords.words('german'))
    except Exception:
        _NLTK_DE_STOPWORDS = None
except Exception:
    _NLTK_DE_STOPWORDS = None

"""
_DEFAULT_DE_STOPWORDS = set(["'n'a'n",
    "der","die","das","und","oder","aber","doch","denn","weil","wenn","dass","da","wie","wer","was",
    "in","im","ins","am","an","auf","aus","bei","für","mit","nach","von","vor","zu","zum","zur","über","unter","ohne","um",
    "ein","eine","einer","einem","einen","kein","keine","keiner","keinem","keinen",
    "ist","sind","war","waren","wird","werden","wurde","wurden","hat","haben","habe","hatte","hatten",
    "ich","du","er","sie","es","wir","ihr","nicht","nur","auch","schon","noch","so","sehr","mehr","weniger",
    "dies","diese","dieser","dieses","jener","jene","jenes","man","sich","sein","seine","seiner","seinem","seinen","ihr","ihre","nein","nichts"
])
"""
_DEFAULT_DE_STOPWORD=set([])

# Default German stopwords as fallback
_DEFAULT_DE_STOPWORDS = set([
    "der", "die", "das", "und", "oder", "aber", "doch", "denn", "weil", "wenn", "dass", "da", "wie", "wer", "was",
    "in", "im", "ins", "am", "an", "auf", "aus", "bei", "für", "mit", "nach", "von", "vor", "zu", "zum", "zur", "über", "unter", "ohne", "um",
    "ein", "eine", "einer", "einem", "einen", "kein", "keine", "keiner", "keinem", "keinen",
    "ist", "sind", "war", "waren", "wird", "werden", "wurde", "wurden", "hat", "haben", "habe", "hatte", "hatten",
    "ich", "du", "er", "sie", "es", "wir", "ihr", "nicht", "nur", "auch", "schon", "noch", "so", "sehr", "mehr", "weniger",
    "dies", "diese", "dieser", "dieses", "jener", "jene", "jenes", "man", "sich", "sein", "seine", "seiner", "seinem", "seinen", "ihr", "ihre", "nein", "nichts"
])

GERMAN_STOPWORDS = list(_SPACY_DE_STOPWORDS or _NLTK_DE_STOPWORDS or _DEFAULT_DE_STOPWORDS)
logger.info(f"Wir verwenden eine Liste mit {len(GERMAN_STOPWORDS)} Stopwords")
# Optional imports with fallback
try:
    from textblob_de import TextBlobDE
    HAS_TEXTBLOB = True
    logger.info("textblob_de successfully imported")
except ImportError as e:
    HAS_TEXTBLOB = False
    logger.warning(f"textblob_de not available: {e}, sentiment analysis will be limited")
except Exception as e:
    HAS_TEXTBLOB = False
    logger.warning(f"textblob_de import failed: {e}, sentiment analysis will be limited")

def extract_tech_keywords_from_texts(texts: List[str], top_k: int = 50) -> List[str]:
    """Extract technical keywords from texts using TF-IDF.
    
    Args:
        texts: List of input texts to analyze
        top_k: Number of top keywords to return (default: 50)
        
    Returns:
        List of top technical keywords ordered by TF-IDF score
        
    Note:
        - Uses German stopwords for filtering
        - Processes n-grams (1-3 words)
        - Filters out terms that appear in too few or too many documents
    """
    
    logger.info(f"[START] extract_tech_keywords_from_texts: {len(texts)} texts, top_k={top_k}")
    try:
        vectorizer = TfidfVectorizer(
            max_features=2000,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.8,
            stop_words=GERMAN_STOPWORDS
        )
        
        X = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        
        # TF-IDF Scores extrahieren
        tfidf_scores = X.sum(axis=0).A1
        feature_scores = list(zip(feature_names, tfidf_scores))
        
        # Nach Score sortieren und Top-K auswählen
        sorted_features = sorted(feature_scores, key=lambda x: x[1], reverse=True)
        top_keywords = [word for word, score in sorted_features[:top_k]]
        
        logger.info("Extracted %d technical keywords from texts", len(top_keywords))
        logger.info("[END] extract_tech_keywords_from_texts: returning %d keywords", len(top_keywords))
        return top_keywords
        
    except Exception as e:
        logger.warning("Failed to extract technical keywords: %s", str(e))
        logger.info("[END] extract_tech_keywords_from_texts: returning []")
        return []

def _extract_topics_and_keywords(texts: List[str], n_topics: int = 10) -> Dict[str, List[str]]:
    """Extract topics and keywords using Latent Dirichlet Allocation (LDA).
    
    Args:
        texts: List of input texts to analyze
        n_topics: Number of topics to extract (default: 10)
        
    Returns:
        Dictionary mapping topic names to lists of top keywords
        
    Note:
        - Uses German stopwords for filtering
        - Processes n-grams (1-3 words)
        - Returns 15 top keywords per topic
    """
    
    logger.info(f"[START] extract_topics_and_keywords: {len(texts)} texts, n_topics={n_topics}")
    try:
        vectorizer = CountVectorizer(
            max_features=2000,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.8,
            stop_words=GERMAN_STOPWORDS
        )
        
        X = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        
        # LDA für Themenmodellierung
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=100
        )
        
        lda.fit(X)
        
        # Themen und Keywords extrahieren
        topics_keywords = {}
        for topic_idx, topic in enumerate(lda.components_):
            top_keywords = [feature_names[i] for i in topic.argsort()[-15:][::-1]]
            topics_keywords[f"Topic_{topic_idx+1}"] = top_keywords
        
        logger.info("Extracted %d topics with keywords", len(topics_keywords))
        logger.info("[END] extract_topics_and_keywords: returning %d topics", len(topics_keywords))
        return topics_keywords
        
    except Exception as e:
        logger.warning("Failed to extract topics and keywords: %s", str(e))
        logger.info("[END] extract_topics_and_keywords: returning {}")
        return {}

def extract_topics_and_keywords(texts: List[str], n_topics: int = 10) -> Dict[str, List[str]]:
    from pipeline import NLP_CACHE

    """Cached version of extract_topics_and_keywords that uses the global caching mechanism.
    
    Args:
        texts: List of input texts
        n_topics: Number of LDA topics to extract
        
    Returns:
        Dictionary mapping topic names to lists of keywords
    """
    if not texts:
        return {}
        
    # Create deterministic cache key from inputs
    texts_tuple = tuple(texts) if isinstance(texts, list) else (texts,)
    texts_str = str(texts_tuple).encode('utf-8')
    texts_hash = hashlib.md5(texts_str).hexdigest()
    cache_key = f"topics_keywords_{n_topics}_{texts_hash}"
    try:
        all_cache_ext_doc_keys=[x for x in NLP_CACHE if x.startswith("topics_keywords_")]
        logger.info(f"Found {len(all_cache_ext_doc_keys)} cached topics/keywords entries")
        logger.info(f"Cache keys: {all_cache_ext_doc_keys}")
    except AssertionError as e:
        logger.error(f"Maybe Cache empty yet? Error getting cache keys: {e}")
    
    # Try cache first
    cached = NLP_CACHE.get(cache_key)
    if cached is not None:
        logger.info(f"Using cached topics/keywords (n={n_topics})")
        return cached
    
    # Compute and store
    logger.info(f"Computing topics/keywords (n={n_topics}) - this might take a while...")
    topics = _extract_topics_and_keywords(texts, n_topics)
    NLP_CACHE.set(cache_key, topics, expire=None)
    logger.info(f"Cached topics/keywords (n={n_topics}), cache key: {cache_key}")
    return topics

def create_dynamic_keyword_lists(texts: List[str], cluster_labels: List[int]) -> Dict[str, List[str]]:
    """Generate cluster-specific keyword lists using TF-IDF.
    
    Args:
        texts: List of input texts
        cluster_labels: Cluster assignments for each text
        
    Returns:
        Dictionary mapping cluster names to lists of top keywords
        
    Note:
        - Creates separate keyword lists for each cluster
        - Processes n-grams (1-2 words)
        - Returns up to 20 top keywords per cluster
    """
    
    logger.info(f"[START] create_dynamic_keyword_lists: {len(texts)} texts, {len(set(cluster_labels))} clusters")
    try:
        # Cluster-spezifische Texte sammeln
        cluster_texts = defaultdict(list)
        for text, label in zip(texts, cluster_labels):
            cluster_texts[label].append(text)
        
        # Pro Cluster Keywords extrahieren
        cluster_keywords = {}
        for cluster_id, cluster_text_list in cluster_texts.items():
            if len(cluster_text_list) > 0:
                # TF-IDF für diesen Cluster
                vectorizer = TfidfVectorizer(
                    max_features=100,
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.9,
                    stop_words=GERMAN_STOPWORDS
                )
                
                X = vectorizer.fit_transform(cluster_text_list)
                feature_names = vectorizer.get_feature_names_out()
                
                # Top Keywords für diesen Cluster
                tfidf_scores = X.sum(axis=0).A1
                feature_scores = list(zip(feature_names, tfidf_scores))
                sorted_features = sorted(feature_scores, key=lambda x: x[1], reverse=True)
                
                top_keywords = [word for word, score in sorted_features[:20]]
                cluster_keywords[f"Cluster_{cluster_id}"] = top_keywords
        
        logger.info("Created dynamic keyword lists for %d clusters", len(cluster_keywords))
        logger.info("[END] create_dynamic_keyword_lists: returning %d clusters", len(cluster_keywords))
        return cluster_keywords
        
    except Exception as e:
        logger.warning("Failed to create dynamic keyword lists: %s", str(e))
        logger.info("[END] create_dynamic_keyword_lists: returning {}")
        return {}

def hybrid_keyword_generation(texts: List[str], base_keywords: Optional[Dict] = None) -> Dict[str, Any]:
    """Combine predefined and dynamically generated keywords with scoring.
    
    Args:
        texts: List of input texts to analyze
        base_keywords: Optional dictionary of predefined keywords
        
    Returns:
        Nested dictionary containing:
        - Technical: TF-IDF based keywords
        - Thematic: LDA-based topics and keywords
        - Cluster_Specific: Cluster-based keywords
        - Base_Keywords: Original base keywords (if provided)
        - summary: Generation statistics
        
    Note:
        - Performs clustering if enough texts are available
        - Combines multiple keyword generation methods
        - Includes source and count metadata
    """
    
    logger.info(f"[START] hybrid_keyword_generation: {len(texts)} texts")
    
    # 1. Dynamische Keywords extrahieren
    dynamic_tech_keywords = extract_tech_keywords_from_texts(texts, top_k=100)
    dynamic_topics = extract_topics_and_keywords(texts, n_topics=8)
    
    # 2. Cluster-spezifische Keywords (falls Cluster vorhanden)
    cluster_keywords = {}
    if len(texts) > 1:
        # Einfache Clusterung für Keywords
        vectorizer = TfidfVectorizer(max_features=1000, min_df=2, max_df=0.8, stop_words=GERMAN_STOPWORDS)
        X = vectorizer.fit_transform(texts)
        
        if X.shape[0] > 1:
            n_clusters = min(4, len(texts) // 2)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(X)
            cluster_keywords = create_dynamic_keyword_lists(texts, cluster_labels)
    
    # 3. Hybrid-Keywords erstellen
    hybrid_keywords = {}
    
    # Technische Keywords
    if dynamic_tech_keywords:
        hybrid_keywords["Technical"] = {
            "keywords": dynamic_tech_keywords[:30],
            "source": "dynamic_tfidf",
            "count": len(dynamic_tech_keywords)
        }
    
    # Themen-basierte Keywords
    if dynamic_topics:
        hybrid_keywords["Thematic"] = {
            "keywords": dynamic_topics,
            "source": "dynamic_lda",
            "count": sum(len(kw) for kw in dynamic_topics.values())
        }
    
    # Cluster-spezifische Keywords
    if cluster_keywords:
        hybrid_keywords["Cluster_Specific"] = {
            "keywords": cluster_keywords,
            "source": "dynamic_clustering",
            "count": sum(len(kw) for kw in cluster_keywords.values())
        }
    
    # 4. Mit Basis-Keywords kombinieren (falls vorhanden)
    if base_keywords:
        hybrid_keywords["Base_Keywords"] = {
            "keywords": base_keywords,
            "source": "static_config",
            "count": sum(len(kw) for kw in base_keywords.values()) if isinstance(base_keywords, dict) else len(base_keywords)
        }
    
    # 5. Scoring und Ranking
    total_keywords = sum(cat["count"] for cat in hybrid_keywords.values())
    hybrid_keywords["summary"] = {
        "total_categories": len(hybrid_keywords),
        "total_keywords": total_keywords,
        "generation_method": "hybrid_dynamic_static"
    }
    
    logger.info("Generated hybrid keywords: %d total keywords across %d categories", total_keywords, len(hybrid_keywords))
    logger.info("[END] hybrid_keyword_generation: returning %d categories", len(hybrid_keywords))
    return hybrid_keywords

def extract_document_themes(texts: List[str], n_themes: int = 8) -> Dict[str, List[str]]:
    """Extract main themes from documents using LDA.
    
    Args:
        texts: List of input texts to analyze
        n_themes: Number of themes to extract (default: 8)
        
    Returns:
        Dictionary mapping theme names to lists of top keywords
        
    Note:
        - Uses German stopwords for filtering
        - Processes n-grams (1-3 words)
        - Returns 10 top keywords per theme
    """
    
    logger.info(f"[START] extract_document_themes: {len(texts)} texts, n_themes={n_themes}")
    try:
        vectorizer = CountVectorizer(
            max_features=2000,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.8,
            stop_words=GERMAN_STOPWORDS
        )
        
        X = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        
        # LDA für Themenmodellierung
        lda = LatentDirichletAllocation(
            n_components=n_themes,
            random_state=42,
            max_iter=100
        )
        
        lda.fit(X)
        
        # Themen extrahieren
        themes = {}
        for topic_idx, topic in enumerate(lda.components_):
            top_words = [feature_names[i] for i in topic.argsort()[-10:][::-1]]
            themes[f"Theme_{topic_idx+1}"] = top_words
            logger.info("Thema %d: %s", topic_idx+1, top_words)
        
        logger.info("Extracted %d themes", len(themes))
        logger.info("[END] extract_document_themes: returning %d themes", len(themes))
        return themes
        
    except Exception as e:
        logger.warning("Failed to extract themes: %s", str(e))
        logger.info("[END] extract_document_themes: returning {}")
        return {}

def analyze_stances_with_dynamic_patterns(texts: List[str], stance_patterns: Dict[str, List[str]]) -> List[List[str]]:
    """Analyze text stances using dynamic patterns.
    
    Args:
        texts: List of input texts to analyze
        stance_patterns: Dictionary mapping stance names to lists of regex patterns
        
    Returns:
        List of stance labels for each text
        
    Note:
        - Returns 'neutral' if no patterns match
        - Uses case-insensitive matching
        - Handles invalid regex patterns gracefully
    """
    
    logger.info(f"[START] analyze_stances_with_dynamic_patterns: {len(texts)} texts, {len(stance_patterns) if stance_patterns else 0} patterns")
    stance_results = []
    
    for text in texts:
        text_stances = []
        for stance_name, patterns in stance_patterns.items():
            for pattern in patterns:
                try:
                    if re.search(pattern, text, re.IGNORECASE):
                        text_stances.append(stance_name)
                        break  # Ein Match pro Stance reicht
                except re.error:
                    logger.warning(f"Invalid regex pattern: {pattern}")
                    continue
        
        if text_stances:
            stance_results.append(text_stances)
        else:
            stance_results.append(["neutral"])
    
    logger.info(f"[END] analyze_stances_with_dynamic_patterns: returning {len(stance_results)} stances")
    return stance_results

def generate_stance_patterns_from_texts(texts: List[str], n_patterns: int = 10) -> Dict[str, List[str]]:
    """Generate stance patterns from texts using TF-IDF and KMeans clustering.
    
    Args:
        texts: List of input texts to analyze
        n_patterns: Target number of patterns to generate (default: 10)
        
    Returns:
        Dictionary mapping cluster-based stance names to lists of keywords
        
    Note:
        - Automatically determines optimal number of clusters
        - Processes n-grams (1-3 words)
        - Returns up to 10 top keywords per cluster
    """
    
    logger.info(f"[START] generate_stance_patterns_from_texts: {len(texts)} texts, n_patterns={n_patterns}")
    try:
        vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.8,
            stop_words=GERMAN_STOPWORDS
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
                    stop_words=GERMAN_STOPWORDS
                )
                
                cluster_X = cluster_vectorizer.fit_transform(cluster_texts)
                cluster_features = cluster_vectorizer.get_feature_names_out()
                
                # Top Features als Patterns
                cluster_scores = cluster_X.sum(axis=0).A1
                top_features = [cluster_features[i] for i in cluster_scores.argsort()[-10:][::-1]]
                
                stance_patterns[f"Cluster_{cluster_id+1}_Stance"] = top_features
        
        logger.info("Generated %d stance patterns from %d texts", len(stance_patterns), len(texts))
        logger.info("[END] generate_stance_patterns_from_texts: returning %d patterns", len(stance_patterns))
        return stance_patterns
        
    except Exception as e:
        logger.warning("Failed to generate stance patterns from texts: %s", str(e))
        logger.info("[END] generate_stance_patterns_from_texts: returning {}")
        return {}

def generate_sentiment_based_stance_patterns(texts: List[str]) -> Dict[str, List[str]]:
    """Generate stance patterns based on sentiment analysis.
    
    Args:
        texts: List of input texts to analyze
        
    Returns:
        Dictionary mapping sentiment-based stance names to lists of keywords
        
    Note:
        - Requires textblob_de package for German sentiment analysis
        - Categorizes texts as positive, negative, or neutral
        - Returns up to 20 top keywords per sentiment category
    """
    
    logger.info(f"[START] generate_sentiment_based_stance_patterns: {len(texts)} texts")
    try:
        if not HAS_TEXTBLOB:
            logger.warning("Skipping sentiment-based stance patterns due to missing textblob_de.")
            logger.info("[END] generate_sentiment_based_stance_patterns: returning {}")
            return {}

        # Sentiment für jeden Text berechnen
        text_sentiments = []
        for text in texts:
            try:
                blob = TextBlobDE(text)
                sentiment = blob.sentiment.polarity
                text_sentiments.append((text, sentiment))
            except (ValueError, AttributeError, Exception) as e:
                logger.debug("Error in sentiment analysis for text: %s", str(e))
                text_sentiments.append((text, 0.0))
        
        # Texte nach Sentiment gruppieren
        positive_texts = [text for text, sent in text_sentiments if sent > 0.1]
        negative_texts = [text for text, sent in text_sentiments if sent < -0.1]
        neutral_texts = [text for text, sent in text_sentiments if -0.1 <= sent <= 0.1]
        
        def extract_common_terms(text_list, top_k=20):
            """Extrahiert häufige Begriffe aus einer Textliste"""
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
        logger.info(f"[END] generate_sentiment_based_stance_patterns: returning {len(stance_patterns)} patterns")
        return stance_patterns
        
    except Exception as e:
        logger.warning(f"Failed to generate sentiment-based stance patterns: {e}")
        logger.info(f"[END] generate_sentiment_based_stance_patterns: returning {'{}'}")
        return {}

def generate_contextual_stance_patterns(texts: List[str]) -> Dict[str, List[str]]:
    """Generate stance patterns from the context of known stance indicators.
    
    Args:
        texts: List of input texts to analyze
        
    Returns:
        Dictionary mapping context-based stance names to lists of keywords
        
    Note:
        - Looks for known stance indicators (pro/contra/neutral)
        - Extracts surrounding context words
        - Returns up to 15 context words per stance type
    """
    
    logger.info(f"[START] generate_contextual_stance_patterns: {len(texts)} texts")
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
        logger.info(f"[END] generate_contextual_stance_patterns: returning {len(contextual_patterns)} patterns")
        return contextual_patterns
        
    except Exception as e:
        logger.warning("Failed to generate contextual stance patterns: %s", str(e))
        logger.info("[END] generate_contextual_stance_patterns: returning {}")
        return {}

def generate_comprehensive_stance_patterns(texts: List[str], base_stance_patterns: Optional[Dict] = None) -> Dict[str, List[str]]:
    """Combine all stance pattern generation methods into comprehensive patterns.
    
    Args:
        texts: List of input texts to analyze
        base_stance_patterns: Optional dictionary of predefined stance patterns
        
    Returns:
        Dictionary mapping comprehensive stance pattern names to keyword lists
        
    Note:
        - Combines TF-IDF, sentiment, and contextual patterns
        - Removes duplicate terms
        - Limits to 20 keywords per pattern
        - Preserves original base patterns if provided
    """
    
    logger.info("[START] generate_comprehensive_stance_patterns: %d texts", len(texts))
    
    # 1. Alle Methoden anwenden
    tfidf_patterns = generate_stance_patterns_from_texts(texts)
    sentiment_patterns = generate_sentiment_based_stance_patterns(texts)
    contextual_patterns = generate_contextual_stance_patterns(texts)
    
    # 2. Alle Patterns zusammenführen
    comprehensive_patterns = {}
    
    # TF-IDF basierte Patterns
    for name, patterns in tfidf_patterns.items():
        comprehensive_patterns[f"TFIDF_{name}"] = patterns
    
    # Sentiment-basierte Patterns
    for name, patterns in sentiment_patterns.items():
        comprehensive_patterns[f"Sentiment_{name}"] = patterns
    
    # Kontext-basierte Patterns
    for name, patterns in contextual_patterns.items():
        comprehensive_patterns[f"Context_{name}"] = patterns
    
    # 3. Mit Basis-Patterns kombinieren (falls vorhanden)
    if base_stance_patterns:
        for name, patterns in base_stance_patterns.items():
            comprehensive_patterns[f"Base_{name}"] = patterns
    
    # 4. Duplikate entfernen und filtern
    filtered_patterns = {}
    seen_terms = set()
    
    for pattern_name, pattern_terms in comprehensive_patterns.items():
        unique_terms = []
        for term in pattern_terms:
            if term not in seen_terms and len(term) > 2:
                unique_terms.append(term)
                seen_terms.add(term)
        
        if unique_terms:
            filtered_patterns[pattern_name] = unique_terms[:20]  # Max 20 pro Pattern
    
    logger.info(f"Generated {len(filtered_patterns)} comprehensive stance patterns")
    logger.info(f"[END] generate_comprehensive_stance_patterns: returning {len(filtered_patterns)} patterns")
    return filtered_patterns 
