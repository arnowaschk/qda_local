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
import re
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity

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

_DEFAULT_DE_STOPWORDS = set([
    "der","die","das","und","oder","aber","doch","denn","weil","wenn","dass","da","wie","wer","was",
    "in","im","ins","am","an","auf","aus","bei","für","mit","nach","von","vor","zu","zum","zur","über","unter","ohne","um",
    "ein","eine","einer","einem","einen","kein","keine","keiner","keinem","keinen",
    "ist","sind","war","waren","wird","werden","wurde","wurden","hat","haben","habe","hatte","hatten",
    "ich","du","er","sie","es","wir","ihr","nicht","nur","auch","schon","noch","so","sehr","mehr","weniger",
    "dies","diese","dieser","dieses","jener","jene","jenes","man","sich","sein","seine","seiner","seinem","seinen","ihr","ihre","nein","nichts"
])

GERMAN_STOPWORDS = list(_SPACY_DE_STOPWORDS or _NLTK_DE_STOPWORDS or _DEFAULT_DE_STOPWORDS)

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

def extract_tech_keywords_from_texts(texts: List[str], top_k: int = 50) -> List[str]:
    """Extrahiert technische Keywords aus Texten mit TF-IDF"""
    
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
        
        logger.info(f"Extracted {len(top_keywords)} technical keywords from texts")
        logger.info(f"[END] extract_tech_keywords_from_texts: returning {len(top_keywords)} keywords")
        return top_keywords
        
    except Exception as e:
        logger.warning(f"Failed to extract technical keywords: {e}")
        logger.info(f"[END] extract_tech_keywords_from_texts: returning []")
        return []

def extract_topics_and_keywords(texts: List[str], n_topics: int = 10) -> Dict[str, List[str]]:
    """Extrahiert Themen und Keywords mit LDA"""
    
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
        
        logger.info(f"Extracted {len(topics_keywords)} topics with keywords")
        logger.info(f"[END] extract_topics_and_keywords: returning {len(topics_keywords)} topics")
        return topics_keywords
        
    except Exception as e:
        logger.warning(f"Failed to extract topics and keywords: {e}")
        logger.info(f"[END] extract_topics_and_keywords: returning {{}}")
        return {}

def create_dynamic_keyword_lists(texts: List[str], cluster_labels: List[int]) -> Dict[str, List[str]]:
    """Erstellt cluster-spezifische Keyword-Listen"""
    
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
        
        logger.info(f"Created dynamic keyword lists for {len(cluster_keywords)} clusters")
        logger.info(f"[END] create_dynamic_keyword_lists: returning {len(cluster_keywords)} clusters")
        return cluster_keywords
        
    except Exception as e:
        logger.warning(f"Failed to create dynamic keyword lists: {e}")
        logger.info(f"[END] create_dynamic_keyword_lists: returning {{}}")
        return {}

def hybrid_keyword_generation(texts: List[str], base_keywords: Optional[Dict] = None) -> Dict[str, Any]:
    """Kombiniert vordefinierte und dynamisch generierte Keywords mit Scoring"""
    
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
    
    logger.info(f"Generated hybrid keywords: {total_keywords} total keywords across {len(hybrid_keywords)} categories")
    logger.info(f"[END] hybrid_keyword_generation: returning {len(hybrid_keywords)} categories")
    return hybrid_keywords

def extract_document_themes(texts: List[str], n_themes: int = 8) -> Dict[str, List[str]]:
    """Extrahiert Hauptthemen aus den Dokumenten mit LDA"""
    
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
        
        logger.info(f"Extracted {len(themes)} themes")
        logger.info(f"[END] extract_document_themes: returning {len(themes)} themes")
        return themes
        
    except Exception as e:
        logger.warning(f"Failed to extract themes: {e}")
        logger.info(f"[END] extract_document_themes: returning {{}}")
        return {}

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
    
    # 3. Policy-Codes aus Themen generieren
    dynamic_codes = []
    used_names: Dict[str, int] = {}

    def slugify(token: str) -> str:
        token = token.strip().lower()
        token = re.sub(r"[^a-z0-9]+", "_", token)
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

def analyze_stances_with_dynamic_patterns(texts: List[str], stance_patterns: Dict[str, List[str]]) -> List[List[str]]:
    """Analysiert Stances mit dynamisch generierten Patterns"""
    
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
    """Generiert Stance-Patterns aus Texten mit TF-IDF und KMeans"""
    
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
        
        logger.info(f"Generated {len(stance_patterns)} stance patterns from {len(texts)} texts")
        logger.info(f"[END] generate_stance_patterns_from_texts: returning {len(stance_patterns)} patterns")
        return stance_patterns
        
    except Exception as e:
        logger.warning(f"Failed to generate stance patterns from texts: {e}")
        logger.info(f"[END] generate_stance_patterns_from_texts: returning {{}}")
        return {}

def generate_sentiment_based_stance_patterns(texts: List[str]) -> Dict[str, List[str]]:
    """Generiert Stance-Patterns basierend auf Sentiment-Analyse"""
    
    logger.info(f"[START] generate_sentiment_based_stance_patterns: {len(texts)} texts")
    try:
        if not HAS_TEXTBLOB:
            logger.warning("Skipping sentiment-based stance patterns due to missing textblob_de.")
            logger.info(f"[END] generate_sentiment_based_stance_patterns: returning {{}}")
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
        logger.info(f"[END] generate_sentiment_based_stance_patterns: returning {{}}")
        return {}

def generate_contextual_stance_patterns(texts: List[str]) -> Dict[str, List[str]]:
    """Generiert Stance-Patterns aus dem Kontext bekannter Stance-Indikatoren"""
    
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
        logger.warning(f"Failed to generate contextual stance patterns: {e}")
        logger.info(f"[END] generate_contextual_stance_patterns: returning {{}}")
        return {}

def generate_comprehensive_stance_patterns(texts: List[str], base_stance_patterns: Optional[Dict] = None) -> Dict[str, List[str]]:
    """Kombiniert alle Stance-Pattern-Generierungsmethoden"""
    
    logger.info(f"[START] generate_comprehensive_stance_patterns: {len(texts)} texts")
    
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
