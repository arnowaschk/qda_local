from typing import List, Dict, Optional, Any
import pathlib
import numpy as np
from collections import defaultdict
from itertools import combinations
from util import logger
import traceback
import math



def analyze_code_cooccurrence(code_lists: List[List[str]]) -> Dict:
    """Analyze co-occurrence patterns of codes across text segments using Jaccard similarity.
    
    This function calculates how frequently codes appear together in the same text segments,
    normalized by their individual frequencies to provide a measure of association strength.
    
    Args:
        code_lists: A list of code lists, where each inner list contains all codes
                   applied to a single text segment. Example:
                   [
                       ['code1', 'code2'],  # Codes for segment 1
                       ['code2', 'code3'],  # Codes for segment 2
                       ...
                   ]
                   
    Returns:
        Dict: A dictionary containing:
            - 'matrix' (List[List[float]]): Symmetric matrix of Jaccard similarity scores
               between all pairs of codes (0-1 range)
            - 'codes' (List[str]): Sorted list of all unique codes in the analysis
            - 'pairs' (Dict[Tuple[str, str], int]): Dictionary mapping code pairs to their
               raw co-occurrence counts
            - 'code_counts' (Dict[str, int]): Dictionary mapping each code to its total
               frequency across all segments
    
    Raises:
        ValueError: If code_lists is empty or contains invalid data
        
    Example:
        >>> code_lists = [
        ...     ['code1', 'code2'],
        ...     ['code2', 'code3'],
        ...     ['code1', 'code3']
        ... ]
        >>> result = analyze_code_cooccurrence(code_lists)
        >>> print(result['codes'])
        ['code1', 'code2', 'code3']
        >>> # Access Jaccard similarity between code1 and code2
        >>> i, j = result['codes'].index('code1'), result['codes'].index('code2')
        >>> similarity = result['matrix'][i][j]
        
    Note:
        - Jaccard similarity is calculated as |A ∩ B| / |A ∪ B|
        - Only segments with 2+ codes are considered for co-occurrence analysis
        - The output matrix is symmetric (matrix[i][j] == matrix[j][i])
        - Diagonal elements represent code frequencies (not similarity scores)
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

def generate_code_network(
    cooccurrence_data: Dict, 
    output_dir: pathlib.Path, 
    min_strength: float = 0.1, 
    max_nodes: int = 50
) -> None:
    """Generate and save a network visualization of code co-occurrence relationships.
    
    Creates a force-directed network graph where nodes represent codes and edges represent
    co-occurrence relationships. The visualization is saved as a high-resolution PNG file.
    
    Args:
        cooccurrence_data: Dictionary containing co-occurrence analysis results from
                         analyze_code_cooccurrence() with keys:
            - 'matrix' (List[List[float]]): Jaccard similarity matrix
            - 'codes' (List[str]): List of code names
            - 'code_counts' (Dict[str, int]): Frequency of each code
        output_dir: Directory where the visualization will be saved. Will be created
                  if it doesn't exist.
        min_strength: Minimum Jaccard similarity (0-1) to draw an edge between codes.
                     Higher values result in fewer, stronger connections. Default: 0.1
        max_nodes: Maximum number of nodes to include, based on code frequency.
                  Helps prevent overcrowding in large codebooks. Default: 50
        
    Returns:
        None: Saves the visualization as 'code_network.png' in the specified directory.
        
    Raises:
        FileNotFoundError: If output_dir cannot be created or is not writable
        ValueError: If cooccurrence_data is missing required keys or has invalid format
        ImportError: If required packages (networkx, matplotlib) are not available
        
    Example:
        >>> data = analyze_code_cooccurrence(code_lists)
        >>> generate_code_network(
        ...     cooccurrence_data=data,
        ...     output_dir=Path("output/network"),
        ...     min_strength=0.2,
        ...     max_nodes=30
        ... )
        
    Note:
        - Node size is proportional to code frequency
        - Edge thickness represents co-occurrence strength (Jaccard similarity)
        - Uses Fruchterman-Reingold force-directed layout algorithm
        - Saves as 300 DPI PNG with dimensions 16x12 inches (4800x3600 pixels)
        - Logs progress and errors using the module's logger
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

def _get_wordcloud_cache_key(word_freq: Dict[str, float], output_path: pathlib.Path, **kwargs) -> str:
    """Generate a cache key for wordcloud generation.
    
    Args:
        word_freq: Dictionary of word frequencies
        output_path: Output path for the wordcloud image
        **kwargs: Additional parameters that affect wordcloud generation
        
    Returns:
        str: A unique cache key for the NLP_CACHE
    """
    import json
    import hashlib
    # Create a stable string representation of the parameters
    params = {
        'word_freq': {k: float(v) for k, v in sorted(word_freq.items())},
        'output_path': str(output_path),
        'title': kwargs.get('title', ''),
        'max_words': kwargs.get('max_words', 200),
        'width': kwargs.get('width', 2480),
        'height': kwargs.get('height', 3508),
        'background_color': kwargs.get('background_color', 'white'),
        'colormap': kwargs.get('colormap', 'viridis')
    }
    
    # Create a hash of the parameters
    param_str = json.dumps(params, sort_keys=True)
    return f"wordcloud:{hashlib.md5(param_str.encode('utf-8')).hexdigest()}"

def generate_word_cloud(
    word_freq: Dict[str, float], 
    output_path: pathlib.Path, 
    title: str = "Word Cloud",
    max_words: int = 200,
    width: int = 2480,  # A4 at 300dpi (8.27in * 300dpi)
    height: int = 3508,  # A4 at 300dpi (11.69in * 300dpi)
    background_color: str = "white",
    colormap: str = "viridis",
    codebook: Optional[Dict[str, Dict]] = None,
    cache_dir: Optional[pathlib.Path] = None,
    nlp_cache: Any = None
) -> None:
    """Generate a visually appealing word cloud from word frequency data.
    
    Creates a word cloud visualization where word size corresponds to frequency/weight.
    Supports custom styling, color schemes, and optional code display name substitution.
    
    Args:
        word_freq: Dictionary mapping words/terms to their associated weights or frequencies.
                  Example: {'python': 0.8, 'data': 0.6, 'analysis': 0.4}
        output_path: File path where the word cloud image will be saved. Should include
                   .png extension. Parent directories will be created if needed.
        title: Title to display above the word cloud. Set to empty string to omit.
        max_words: Maximum number of words to include in the cloud. Default: 200
        width: Width of the output image in pixels. Default: 2480 (A4 @ 300dpi)
        height: Height of the output image in pixels. Default: 3508 (A4 @ 300dpi)
        background_color: Background color name or hex code. Default: "white"
        colormap: Name of a matplotlib colormap for word colors. Common options:
                 'viridis', 'plasma', 'inferno', 'magma', 'cividis'. Default: 'viridis'
        codebook: Optional dictionary mapping code IDs to metadata. If provided and a key
                in word_freq matches a code ID, the code's display_name will be used
                in the visualization. Example:
                {
                    'code1': {'display_name': 'First Code', 'color': '#ff0000'},
                    'code2': {'display_name': 'Second Code'}
                }
        cache_dir: Optional directory to use for caching wordclouds. If not provided,
                  a default cache directory will be used.
                
    Returns:
        None: The word cloud is saved to the specified output path as a PNG file.
        
    Raises:
        FileNotFoundError: If the output directory cannot be created
        ValueError: If word_freq is empty or contains invalid values
        ImportError: If required packages (wordcloud, matplotlib) are not available
        
    Example:
        >>> word_freq = {
        ...     'code1': 45,
        ...     'code2': 32,
        ...     'code3': 21
        ... }
        >>> generate_word_cloud(
        ...     word_freq=word_freq,
        ...     output_path=Path("output/wordcloud.png"),
        ...     title="Frequent Codes",
        ...     max_words=100,
        ...     colormap="plasma"
        ... )
        
    Note:
        - Words are scaled proportionally to their frequency/weight
        - Layout is optimized to fit words within the specified dimensions
        - Output is saved as a 300 DPI PNG with transparent or colored background
        - Handles special characters and non-ASCII text
        - Logs progress and errors using the module's logger
    """
    try:
        import matplotlib.pyplot as plt
        from wordcloud import WordCloud
        import random
        
        # Generate cache key
        cache_key = _get_wordcloud_cache_key(
            word_freq,
            output_path=output_path,
            title=title,
            max_words=max_words,
            width=width,
            height=height,
            background_color=background_color,
            colormap=colormap,
            codebook=codebook is not None
        )
        
        # Check cache first
        if nlp_cache and cache_key in nlp_cache:
            logger.info(f"Using cached wordcloud from NLP_CACHE with key: {cache_key}")
            return
        
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create a color function based on the colormap
        cmap = plt.get_cmap(colormap)
        def color_func(*args, **kwargs):
            return tuple(int(x * 255) for x in cmap(random.random())[:3])
        
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
        
        # Filter out NaN values and ensure all frequencies are positive numbers
        filtered_freq = {}
        for k, v in word_freq.items():
            if v is not None and not math.isnan(v) and v > 0:
                # Use display_name from codebook if available, otherwise use the code as is
                display_name = k
                if codebook and k in codebook and 'display_name' in codebook[k]:
                    display_name = codebook[k]['display_name']
                filtered_freq[display_name] = float(v)
        
        if not filtered_freq:
            logger.warning("No valid word frequencies found after filtering NaN/zero values")
            return
            
        # Generate the word cloud
        wc.generate_from_frequencies(filtered_freq)
         
        # Create a figure and axis with A4 dimensions
        fig, ax = plt.subplots(figsize=(8.27, 11.69), dpi=300)  # A4 size in inches at 300dpi
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        
        # Add title with some padding
        if title:
            fig.suptitle(title, fontsize=24, y=0.98)
        
        # Save the figure with minimal padding
        plt.tight_layout(pad=0.1)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close()
        
        # Save the word cloud to file
        wc.to_file(str(output_path))
        logger.info(f"Word cloud saved to {output_path}")
        
        # Cache the result if nlp_cache is provided and has set method
        if nlp_cache is not None and hasattr(nlp_cache, 'set'):
            try:
                nlp_cache.set(cache_key, True, expire=864000)  # Cache for ten days
                logger.debug(f"Cached wordcloud with key: {cache_key}")
            except Exception as e:
                logger.warning(f"Warning: Could not cache wordcloud: {e}")
        else:
            logger.warning("No nlp_cache provided, skipping caching")
            
    except Exception as e:
        logger.error(f"Error generating word cloud: {e}")
        logger.error(traceback.format_exc())
        raise
    finally:
        # FanoutCache handles syncing automatically, no need to call sync()
        pass
