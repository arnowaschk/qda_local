"""
HTML Report Generation Module for QDA Pipeline

This module contains functions for generating comprehensive HTML reports
from the QDA analysis results.
"""

import os
import logging
from typing import Any, Dict, List, Set
import re
import datetime

logger = logging.getLogger(__name__)

def generate_html_report(
    seg: List[Dict[str, Any]],
    cluster_info: Dict[int, Dict[str, Any]],
    cluster_summaries: Dict[int, str],
    codebook_sorted: Dict[str, int],
    global_summary: str,
    input_path: str,
    k: int
) -> str:
    """Generate a comprehensive HTML report from QDA analysis results.
    
    Creates a well-formatted HTML document that presents the analysis results
    including cluster information, codebook, and global summary in a visually
    appealing and interactive format.
    
    Args:
        seg: List of text segments/documents that were analyzed
        cluster_info: Dictionary mapping cluster IDs to their metadata and content
        cluster_summaries: Dictionary mapping cluster IDs to their summary text
        codebook_sorted: Dictionary of code names and their frequencies, sorted
        global_summary: Overall summary text of the analysis
        input_path: Path to the input file that was analyzed
        k: Number of clusters used in the analysis
        
    Returns:
        str: Complete HTML document as a string
        
    Example:
        >>> report = generate_html_report(
        ...     segments,
        ...     cluster_data,
        ...     cluster_summaries,
        ...     sorted_codes,
        ...     "Overall analysis summary...",
        ...     "input/survey_data.csv",
        ...     5
        ... )
        >>> with open("report.html", "w") as f:
        ...     f.write(report)
    """
    
    # Start building the HTML content
    html_parts = []
    
    # HTML header with comprehensive styling
    html_parts.append("""<!doctype html>
<html lang="de">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QDA Report - """ + os.path.basename(input_path) + """</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
            color: #333;
            background-color: #f8f9fa;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 30px;
        }
        h2 {
            color: #34495e;
            border-left: 4px solid #3498db;
            padding-left: 15px;
            margin-top: 40px;
            margin-bottom: 20px;
        }
        h3 {
            color: #2c3e50;
            margin-top: 30px;
            margin-bottom: 15px;
        }
        h4, h5, h6 {
            color: #34495e;
            margin-top: 20px;
            margin-bottom: 10px;
        }
        .metadata {
            background: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 30px;
        }
        .metadata ul {
            list-style: none;
            padding: 0;
            margin: 0;
        }
        .metadata li {
            margin: 5px 0;
            padding: 5px 0;
        }
        .cluster-section {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 5px;
            padding: 20px;
            margin: 20px 0;
        }
        .cluster-header {
            background: #3498db;
            color: white;
            padding: 10px 15px;
            margin: -20px -20px 20px -20px;
            border-radius: 5px 5px 0 0;
        }
        .cluster-content {
            margin-left: 20px;
        }
        .text-example {
            background: white;
            border-left: 4px solid #e74c3c;
            padding: 15px;
            margin: 15px 0;
            border-radius: 0 5px 5px 0;
        }
        .text-example .question {
            font-weight: bold;
            color: #e74c3c;
            margin-bottom: 10px;
        }
        .text-example .answer {
            color: #2c3e50;
        }
        .codebook {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 5px;
            padding: 20px;
            margin: 20px 0;
        }
        .codebook h3 {
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        .code-item {
            background: white;
            border: 1px solid #dee2e6;
            border-radius: 5px;
            padding: 15px;
            margin: 10px 0;
        }
        .code-name {
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
        }
        .code-patterns {
            color: #7f8c8d;
            font-family: monospace;
            background: #ecf0f1;
            padding: 5px;
            border-radius: 3px;
        }
        .summary {
            background: #e8f5e8;
            border: 1px solid #27ae60;
            border-radius: 5px;
            padding: 20px;
            margin: 20px 0;
        }
        .summary h3 {
            color: #27ae60;
            margin-top: 0;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .stat-item {
            background: white;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
            border: 1px solid #dee2e6;
        }
        .stat-number {
            font-size: 2em;
            font-weight: bold;
            color: #3498db;
        }
        .stat-label {
            color: #7f8c8d;
            margin-top: 5px;
        }
        .footer {
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #dee2e6;
            color: #7f8c8d;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>QDA Analyse Report</h1>
        
        <div class="metadata">
            <h2>Metadaten</h2>
            <ul>
                <li><strong>Eingabedatei:</strong> """ + os.path.basename(input_path) + """</li>
                <li><strong>Anzahl Cluster:</strong> """ + str(k) + """</li>
                <li><strong>Anzahl Texte:</strong> """ + str(len(seg)) + """</li>
                <li><strong>Generiert am:</strong> """ + (
                    datetime.datetime.fromtimestamp(os.path.getmtime(input_path)).strftime('%Y-%m-%d %H:%M')
                    if input_path and os.path.exists(input_path) else datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
                ) + """</li>
            </ul>
        </div>
        
        <div class="stats">
            <div class="stat-item">
                <div class="stat-number">""" + str(len(seg)) + """</div>
                <div class="stat-label">Texte analysiert</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">""" + str(k) + """</div>
                <div class="stat-label">Cluster erstellt</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">""" + str(len(codebook_sorted)) + """</div>
                <div class="stat-label">Codes gefunden</div>
            </div>
        </div>""")
    
    # Cluster sections
    for cluster_id in range(k):
        if cluster_info and cluster_id in cluster_info:
            cluster_data = cluster_info[cluster_id]
            cluster_summary = cluster_summaries.get(cluster_id, "Keine Zusammenfassung verfügbar")
            logger.info(f"Cluster {cluster_id} is {cluster_data.__repr__()[:200]}")
            logger.info(f"Cluster {cluster_id} has {cluster_data.keys()} keys")
            html_parts.append(f"""
        <div class="cluster-section">
            <div class="cluster-header">
                <h3>Cluster {cluster_id + 1}</h3>
            </div>
            <div class="cluster-content">
                <p><strong>Anzahl Texte:</strong> {len(cluster_data.get('texts', []))}</p>
                <p><strong>Durchschnittliche Länge:</strong> {cluster_data.get('avg_length', 0):.1f} Zeichen</p>
                
                <h4>Beispieltexte:</h4>""")
            
            # Show first 3 texts as examples
            for i, text in enumerate((cluster_data.get('texts') or [])[:3]):
                html_parts.append(f"""
                <div class="text-example">
                    <div class="question">Text {i + 1}:</div>
                    <div class="answer">{text[:200]}{'...' if len(text) > 200 else ''}</div>
                </div>""")
            
            html_parts.append(f"""
                <h4>Zusammenfassung:</h4>
                <p>{cluster_summary}</p>
            </div>
        </div>""")
    
    # Codebook section
    html_parts.append("""
        <div class="codebook">
            <h2>Codebook</h2>""")
    
    for code_name, count in (codebook_sorted or {}).items():
        html_parts.append(f"""
            <div class="code-item">
                <div class="code-name">{code_name}</div>
                <div class="code-patterns">Anzahl: {count}</div>
            </div>""")
    
    html_parts.append("""
        </div>
        
        <div class="summary">
            <h3>Globale Zusammenfassung</h3>
            <p>""")
    
    if global_summary:
        # apply minimal formatting for readability
        html_parts.append(format_text_for_html(str(global_summary)))
    else:
        html_parts.append("Keine globale Zusammenfassung verfügbar.")
    
    html_parts.append("""
            </p>
        </div>
        
        <div class="footer">
            <p>QDA Pipeline Report - Generiert mit Python und spaCy</p>
        </div>
    </div>
</body>
</html>""")
    
    return ''.join(html_parts) 

def generate_policies_html(
    policies: Dict[str, Any],
    dynamic_policies: Dict[str, Any],
    keywords: Dict[str, Any],
    dynamic_keywords: Dict[str, Any],
    stance_patterns: Dict[str, List[str]],
    used_policies: Set[str],
    used_keywords: Set[str],
    used_stances: Set[str],
    input_path: str
) -> str:
    """Generate an HTML report of policies, keywords, and stance patterns.
    
    Creates a comprehensive overview of all analysis policies, keywords, and stances,
    highlighting which items were actually used during analysis. Dynamically generated
    items are marked with an asterisk (*) and unused items with a minus (-).
    
    Args:
        policies: Dictionary of static policy definitions
        dynamic_policies: Dictionary of dynamically generated policies
        keywords: Dictionary of static keyword definitions
        dynamic_keywords: Dictionary of dynamically generated keywords
        stance_patterns: Dictionary of stance patterns and their definitions
        used_policies: Set of policy names that were used in analysis
        used_keywords: Set of keywords that were matched in the text
        used_stances: Set of stance patterns that were detected
        input_path: Path to the input file that was analyzed
        
    Returns:
        str: Complete HTML document as a string
        
    Note:
        - Items marked with * were dynamically generated during analysis
        - Items marked with - were not used in the current analysis
    """
    import datetime
    html_parts = []
    html_parts.append("""<!doctype html>
<html lang=\"de\">
<head>
    <meta charset=\"utf-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
    <title>QDA Policies & Keywords Übersicht</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; line-height: 1.6; color: #333; background-color: #f8f9fa; }
        .container { background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; margin-bottom: 30px; }
        h2 { color: #34495e; border-left: 4px solid #3498db; padding-left: 15px; margin-top: 40px; margin-bottom: 20px; }
        ul { list-style: none; padding: 0; }
        li { margin: 8px 0; padding: 4px 0; }
        .asterisk { color: #e67e22; font-weight: bold; }
        .minus { color: #aaa; font-weight: bold; margin-right: 4px; }
        .section { margin-bottom: 40px; }
        .meta { background: #ecf0f1; padding: 15px; border-radius: 5px; margin-bottom: 30px; }
        .footer { margin-top: 40px; padding-top: 20px; border-top: 1px solid #dee2e6; color: #7f8c8d; text-align: center; }
    </style>
</head>
<body>
    <div class=\"container\">
        <h1>QDA Policies, Keywords & Stances Übersicht</h1>
        <div class=\"meta\">
            <strong>Eingabedatei:</strong> """ + os.path.basename(input_path) + """<br>
            <strong>Generiert am:</strong> """ + datetime.datetime.now().strftime('%Y-%m-%d %H:%M') + """
        </div>
""")

    # --- Policies Section ---
    html_parts.append('<div class="section"><h2>Policies</h2><ul>')
    all_policy_names = set()
    # Static policies
    for item in policies.get("codes", []):
        name = item.get("name")
        if not name:
            continue
        all_policy_names.add(name)
        is_used = name in used_policies
        prefix = '<span class="minus">-</span>' if not is_used else ''
        html_parts.append(f'<li>{prefix}{name}</li>')
    # Dynamic policies
    for item in dynamic_policies.get("codes", []):
        name = item.get("name")
        if not name or name in all_policy_names:
            continue
        all_policy_names.add(name)
        is_used = name in used_policies
        prefix = '<span class="minus">-</span>' if not is_used else ''
        html_parts.append(f'<li>{prefix}{name}<span class="asterisk">*</span></li>')
    html_parts.append('</ul></div>')

    # --- Keywords Section ---
    html_parts.append('<div class="section"><h2>Keywords</h2><ul>')
    # Static keywords
    for cat, val in (keywords.get("Base_Keywords", {}).get("keywords", {}) or {}).items():
        for kw in val:
            is_used = kw in used_keywords
            prefix = '<span class="minus">-</span>' if not is_used else ''
            html_parts.append(f'<li>{prefix}{kw}</li>')
    # Dynamic keywords
    for cat, catinfo in dynamic_keywords.items():
        if cat == "Base_Keywords" or cat == "summary":
            continue
        kws = catinfo.get("keywords", {})
        if isinstance(kws, dict):
            for subcat, subkws in kws.items():
                for kw in subkws:
                    is_used = kw in used_keywords
                    prefix = '<span class="minus">-</span>' if not is_used else ''
                    html_parts.append(f'<li>{prefix}{kw}<span class="asterisk">*</span></li>')
        elif isinstance(kws, list):
            for kw in kws:
                is_used = kw in used_keywords
                prefix = '<span class="minus">-</span>' if not is_used else ''
                html_parts.append(f'<li>{prefix}{kw}<span class="asterisk">*</span></li>')
    html_parts.append('</ul></div>')

    # --- Stances Section ---
    html_parts.append('<div class="section"><h2>Stances</h2><ul>')
    for stance_name, patterns in (stance_patterns or {}).items():
        is_used = stance_name in used_stances
        prefix = '<span class="minus">-</span>' if not is_used else ''
        html_parts.append(f'<li>{prefix}{stance_name}<span class="asterisk">*</span></li>')
    html_parts.append('</ul></div>')

    html_parts.append('<div class="footer">QDA Pipeline Policies/Keywords/Stances Übersicht</div>')
    html_parts.append('</div></body></html>')
    return ''.join(html_parts) 

def format_text_for_html(text: str) -> str:
    """Convert plain text with markdown-style formatting to HTML.
    
    Processes text with markdown-style formatting and converts it to HTML.
    Handles common formatting patterns like bold, italics, headers, lists,
    code blocks, and URLs.
    
    Args:
        text: Input text potentially containing markdown-style formatting
        
    Returns:
        str: Formatted HTML text
        
    Example:
        >>> format_text("**Bold** and *italic* text")
        '&lt;strong&gt;Bold&lt;/strong&gt; and &lt;em&gt;italic&lt;/em&gt; text'
        
    Supported Formatting:
        - **bold** → &lt;strong&gt;bold&lt;/strong&gt;
        - *italic* → &lt;em&gt;italic&lt;/em&gt;
        - # Header → &lt;h4&gt;Header&lt;/h4&gt;
        - - Item → &lt;li&gt;Item&lt;/li&gt;
        - `code` → &lt;code&gt;code&lt;/code&gt;
        - URLs → Clickable links
    """
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

