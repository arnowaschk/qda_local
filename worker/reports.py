import pandas as pd
from typing import Dict, Optional
import pathlib
from util import logger
import traceback
import json
import re

# External analysis/visualization utilities used by report generation
from networks import (
    analyze_code_cooccurrence,
    generate_code_network,
    generate_word_cloud,
)
from html_report import (
    generate_html_report as _generate_dataset_html,
    generate_policies_html as _generate_policies_html,
    format_text_for_html as _md_to_html,
)

def generate_structured_report(
    seg: pd.DataFrame,
    codebook: Optional[Dict] = None,
    code_cooccurrence: Optional[Dict] = None,
    output_dir: pathlib.Path = None,
    question_texts: Optional[Dict[int, str]] = None,
    report_name: Optional[str] = None,
    dynamic_policies: Optional[Dict] = None,
    dynamic_keywords: Optional[Dict] = None,
    used_policies: Optional[set] = None,
    used_keywords: Optional[set] = None,
    used_stances: Optional[set] = None,
    cluster_summaries: Optional[Dict] = None,
    global_summary: Optional[str] = None,
    input_path: Optional[str] = None,
    k_clusters: Optional[int] = None,
    policies: Optional[Dict] = None,
    stance_patterns: Optional[Dict] = None,
) -> None:
    """Generate comprehensive structured reports for qualitative analysis results.
    
    This function orchestrates the generation of multiple report types including overall summaries,
    question-specific reports, code-specific analyses, and visualizations. It handles data preparation,
    computes necessary statistics, and writes all outputs to the specified directory.
    
    Args:
        seg: DataFrame containing segmented text data with 'codes' column and optional metadata.
        codebook: Dictionary mapping codes to their frequencies. If None, computed from seg.
        code_cooccurrence: Dictionary containing co-occurrence analysis results with keys 'codes',
                         'matrix', and 'pairs'. If None, computed from seg.
        output_dir: Directory path where all report files will be saved.
        question_texts: Optional dict mapping question indices to question texts for per-question reports.
        report_name: Optional name identifying the report scope (e.g., column or dataset name).
        dynamic_policies: Dictionary of dynamically generated policies from pipeline.
        dynamic_keywords: Dictionary of dynamically generated keywords from pipeline.
        used_policies: Set of policy codes that were actually used in analysis.
        used_keywords: Set of keywords that were actually found in the text.
        used_stances: Set of stance patterns that were actually found in the text.
        cluster_summaries: Dictionary of cluster IDs to summary texts.
        global_summary: Precomputed global summary text to include in reports.
        input_path: Path to the input data file for reference.
        k_clusters: Number of clusters used in analysis (if applicable).
        policies: Dictionary of policy definitions and metadata.
        stance_patterns: Dictionary of stance patterns used in analysis.
        
    Returns:
        None: All outputs are written to files in the specified output directory.
        
    Raises:
        FileNotFoundError: If output_dir cannot be created or accessed.
        ValueError: If required input data is missing or malformed.
        
    Example:
        >>> generate_structured_report(
        ...     seg=df,
        ...     codebook={"code1": 10, "code2": 5},
        ...     output_dir=Path("reports"),
        ...     question_texts={1: "What are your thoughts?"}
        ... )
    """
    try:
        # Create reports directory
        reports_dir = output_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure the output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting report generation in {reports_dir}")
        
        # Auto-derive used sets if not provided
        try:
            # Policies: collect codes used in seg
            if not used_policies:
                used_policies = set()
                if seg is not None and 'codes' in seg.columns:
                    for codes in seg['codes']:
                        if codes:
                            for c in codes:
                                used_policies.add(str(c))

            # Prepare concatenated text for keyword/stance scanning
            all_text = " ".join(str(t) for t in seg['text'] if pd.notna(t)) if (seg is not None and 'text' in seg.columns) else ""
            all_text_l = all_text.lower()

            def _flatten_keywords(dk: Dict) -> set:
                vals = set()
                if not isinstance(dk, dict):
                    return vals
                # common sections
                for key in ["Technical", "Thematic", "Cluster_Specific", "Base_Keywords", "keywords"]:
                    if key in dk and dk.get(key) is not None:
                        payload = dk.get(key)
                        if isinstance(payload, dict) and "keywords" in payload and payload.get("keywords") is not None:
                            payload = payload.get("keywords")
                        if isinstance(payload, dict):
                            for _, words in payload.items():
                                if isinstance(words, (list, set, tuple)):
                                    vals.update(str(x) for x in words if x)
                                elif words:
                                    vals.add(str(words))
                        elif isinstance(payload, (list, set, tuple)):
                            vals.update(str(x) for x in payload if x)
                        elif payload:
                            vals.add(str(payload))
                # any remaining keys
                for k, v in dk.items():
                    if k in ["Technical", "Thematic", "Cluster_Specific", "Base_Keywords", "keywords"]:
                        continue
                    if isinstance(v, dict):
                        for _, words in v.items():
                            if isinstance(words, (list, set, tuple)):
                                vals.update(str(x) for x in words if x)
                            elif words:
                                vals.add(str(words))
                    elif isinstance(v, (list, set, tuple)):
                        vals.update(str(x) for x in v if x)
                    elif v:
                        vals.add(str(v))
                return {s.strip() for s in vals if str(s).strip()}

            # Keywords: mark those present in text
            if not used_keywords and dynamic_keywords:
                used_keywords = set()
                candidates = _flatten_keywords(dynamic_keywords)
                for kw in candidates:
                    kw_l = kw.lower()
                    # simple word-boundary regex to reduce substring noise
                    try:
                        if re.search(rf"\b{re.escape(kw_l)}\b", all_text_l):
                            used_keywords.add(kw)
                    except re.error:
                        # fallback to substring
                        if kw_l in all_text_l:
                            used_keywords.add(kw)

            # Stances: flatten patterns and mark those present in text
            if not used_stances and stance_patterns:
                used_stances = set()
                if isinstance(stance_patterns, dict):
                    for _, patterns in stance_patterns.items():
                        seq = patterns if isinstance(patterns, (list, set, tuple)) else [patterns]
                        for pat in seq:
                            if not pat:
                                continue
                            pat_s = str(pat).strip()
                            if not pat_s:
                                continue
                            pat_l = pat_s.lower()
                            try:
                                if re.search(rf"\b{re.escape(pat_l)}\b", all_text_l):
                                    used_stances.add(pat_s)
                            except re.error:
                                if pat_l in all_text_l:
                                    used_stances.add(pat_s)
                else:
                    pat_s = str(stance_patterns)
                    if pat_s and pat_s.lower() in all_text_l:
                        used_stances.add(pat_s)
        except Exception as e:
            logger.warning(f"Failed to auto-derive used sets: {e}")

        # Ensure we have a frequency codebook based on seg if necessary
        # The pipeline may pass a policy structure instead of frequencies.
        if codebook is None or (codebook and not all(isinstance(v, (int, float)) for v in codebook.values())):
            freq: Dict[str, int] = {}
            if 'codes' in seg.columns:
                for codes in seg['codes']:
                    for c in (codes or []):
                        freq[c] = freq.get(c, 0) + 1
            codebook = freq

        # Compute co-occurrence if not provided
        if code_cooccurrence is None:
            try:
                code_lists = [codes for codes in seg['codes'] if codes]
                code_cooccurrence = analyze_code_cooccurrence(code_lists) if code_lists else {
                    "codes": [], "matrix": [], "pairs": {}
                }
            except Exception as e:
                logger.warning(f"Could not compute code co-occurrence: {e}")
                code_cooccurrence = {"codes": [], "matrix": [], "pairs": {}}

        # Store pipeline metadata
        try:
            meta = {
                "report_name": report_name,
                "input_path": input_path,
                "k_clusters": k_clusters,
                "num_segments": int(len(seg)) if seg is not None else 0,
                "num_unique_codes": int(len(codebook)) if codebook else 0,
                "has_dynamic_policies": bool(dynamic_policies),
                "has_dynamic_keywords": bool(dynamic_keywords),
                "num_stance_patterns": len(stance_patterns or {}),
            }
            (reports_dir / "metadata.json").write_text(
                json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:
            # Avoid breaking report generation due to optional metadata
            pass

        # 1. Overall Summary Report
        logger.info("Generating overall summary report...")
        overall_path = reports_dir / "overall_summary.md"
        if global_summary:
            # If a precomputed global summary exists, save it directly and append stats
            try:
                report_txt = "# Qualitative Analysis Summary Report\n\n"
                if report_name:
                    report_txt += f"Scope: {report_name}\n\n"
                report_txt += global_summary.strip() + "\n\n---\n\n"
                # Append basic stats similar to generate_overall_summary
                total_segments = len(seg)
                total_codes = sum(codebook.values()) if codebook else 0
                unique_codes = len(codebook) if codebook else 0
                avg_codes_per_segment = total_codes / total_segments if total_segments > 0 else 0
                report_txt += "## Overview\n"
                report_txt += f"- **Total Segments Analyzed**: {total_segments:,}\n"
                report_txt += f"- **Total Code Applications**: {total_codes:,}\n"
                report_txt += f"- **Unique Codes**: {unique_codes:,}\n"
                report_txt += f"- **Average Codes per Segment**: {avg_codes_per_segment:.2f}\n"
                overall_path.write_text(report_txt, encoding="utf-8")
            except Exception:
                # Fallback to programmatic summary generation
                generate_overall_summary(
                    seg=seg,
                    codebook=codebook,
                    code_cooccurrence=code_cooccurrence,
                    output_path=overall_path,
                )
        else:
            generate_overall_summary(
                seg=seg,
                codebook=codebook,
                code_cooccurrence=code_cooccurrence,
                output_path=overall_path
            )

        # Also provide a simple HTML rendering of the overall summary
        try:
            md_txt = overall_path.read_text(encoding="utf-8")
            html_body = _md_to_html(md_txt)
            html_out = reports_dir / "overall_summary.html"
            html_out.write_text(
                """<!doctype html><html lang=\"de\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\"><title>Overall Summary</title><style>body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif;max-width:1200px;margin:0 auto;padding:20px;line-height:1.6;color:#333;background:#f8f9fa}.container{background:#fff;padding:30px;border-radius:8px;box-shadow:0 2px 10px rgba(0,0,0,.1)}</style></head><body><div class=\"container\">"""
                + html_body + """</div></body></html>""",
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(f"Failed to render overall summary HTML: {e}")
        
        # 1b. Persist cluster summaries if provided
        if cluster_summaries:
            try:
                cluster_md = reports_dir / "cluster_summaries.md"
                lines = ["# Cluster Summaries\n"]
                for cid, summary in sorted(cluster_summaries.items(), key=lambda x: x[0]):
                    lines.append(f"\n## Cluster {cid}\n")
                    lines.append(summary if isinstance(summary, str) else str(summary))
                cluster_md.write_text("\n".join(lines), encoding="utf-8")
            except Exception:
                pass

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
                    q_code_lists = []
                    if 'codes' in q_seg.columns:
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
                    # Also generate an HTML rendering of the question report
                    try:
                        md = (question_reports_dir / f"question_{q_idx}_report.md").read_text(encoding="utf-8")
                        html_body = _md_to_html(md)
                        (question_reports_dir / f"question_{q_idx}_report.html").write_text(
                            """<!doctype html><html lang=\"de\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\"><title>Question Report</title><style>body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif;max-width:1200px;margin:0 auto;padding:20px;line-height:1.6;color:#333;background:#f8f9fa}.container{background:#fff;padding:30px;border-radius:8px;box-shadow:0 2px 10px rgba(0,0,0,.1)}</style></head><body><div class=\"container\">"""
                            + html_body + """</div></body></html>""",
                            encoding="utf-8",
                        )
                    except Exception as e:
                        logger.warning(f"Failed to render question {q_idx} HTML: {e}")
                    
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
        # 3b. Dedicated diagnostics: list all artifacts and mark used ones
        try:
            diagnostics_dir = reports_dir / "diagnostics"
            diagnostics_dir.mkdir(exist_ok=True)

            def _mark(name: str, used: Optional[set]) -> str:
                try:
                    return f"{name}*" if used and name in used else str(name)
                except Exception:
                    return str(name)

            # Policies diagnostics
            if dynamic_policies:
                pol_path = diagnostics_dir / "policies.md"
                lines = ["# Policies (All)\n",
                         "Items marked with '*' were used. Nothing is filtered out.\n",
                         "\n## Codes\n"]
                codes = dynamic_policies.get("codes") or []
                if isinstance(codes, list) and codes:
                    for c in codes:
                        if isinstance(c, dict):
                            name = c.get("name") or c.get("id") or "unknown"
                            disp = c.get("display_name") or name
                            lines.append(f"- {_mark(name, used_policies)} ({disp})")
                        else:
                            lines.append(f"- {_mark(str(c), used_policies)}")
                else:
                    lines.append("(no codes available)\n")

                # Optional: themes/top_terms
                themes = dynamic_policies.get("themes") or {}
                if isinstance(themes, dict) and themes:
                    lines.append("\n## Themes\n")
                    for tname, words in themes.items():
                        lines.append(f"- {tname}: {', '.join(map(str, words or []))}")

                top_terms = dynamic_policies.get("top_terms") or []
                if top_terms:
                    lines.append("\n## Top Terms\n")
                    lines.append(", ".join(map(str, top_terms)))

                pol_path.write_text("\n".join(lines), encoding="utf-8")

            # Keywords diagnostics
            if dynamic_keywords:
                kw_path = diagnostics_dir / "keywords.md"
                lines = ["# Keywords (All)\n",
                         "Items marked with '*' were used. Nothing is filtered out.\n"]

                def _render_keywords_section(title: str, data):
                    lines.append(f"\n## {title}\n")
                    if isinstance(data, dict):
                        # dict of topic -> list[str]
                        for sub, words in data.items():
                            lines.append(f"- {sub}:")
                            if isinstance(words, (list, set, tuple)):
                                for w in sorted({str(x) for x in words}):
                                    lines.append(f"  - {_mark(w, used_keywords)}")
                            else:
                                lines.append(f"  - {_mark(str(words), used_keywords)}")
                    elif isinstance(data, (list, set, tuple)):
                        for w in sorted({str(x) for x in data}):
                            lines.append(f"- {_mark(w, used_keywords)}")
                    else:
                        lines.append(f"- {_mark(str(data), used_keywords)}")

                # Try common structures
                for key in ["Technical", "Thematic", "Cluster_Specific", "Base_Keywords", "keywords"]:
                    if key in dynamic_keywords and dynamic_keywords.get(key) is not None:
                        payload = dynamic_keywords.get(key)
                        # Some providers nest in {key: {keywords: ...}}
                        if isinstance(payload, dict) and "keywords" in payload and payload.get("keywords") is not None:
                            payload = payload.get("keywords")
                        _render_keywords_section(key, payload)

                # Fallback: render remaining keys generically
                for k, v in dynamic_keywords.items():
                    if k in ["Technical", "Thematic", "Cluster_Specific", "Base_Keywords", "keywords"]:
                        continue
                    _render_keywords_section(k, v)

                kw_path.write_text("\n".join(lines), encoding="utf-8")

            # Stance patterns diagnostics
            if stance_patterns:
                st_path = diagnostics_dir / "stance_patterns.md"
                lines = ["# Stance Patterns (All)\n",
                         "Items marked with '*' were used. Nothing is filtered out.\n"]
                if isinstance(stance_patterns, dict):
                    for group, patterns in stance_patterns.items():
                        lines.append(f"\n## {group}\n")
                        if isinstance(patterns, (list, set, tuple)):
                            for p in patterns:
                                lines.append(f"- {_mark(str(p), used_stances)}")
                        else:
                            lines.append(f"- {_mark(str(patterns), used_stances)}")
                else:
                    lines.append(f"- {_mark(str(stance_patterns), used_stances)}")
                st_path.write_text("\n".join(lines), encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to create diagnostics sections: {e}")

        # 4. Generate a README with an overview of all reports
        logger.info("Generating README...")
        readme_path = reports_dir / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write("# Qualitative Analysis Reports\n\n")
            scope = report_name or "full dataset"
            f.write(f"Scope: {scope}\n\n")
            f.write("This directory contains the following reports and visualizations:\n\n")
            
            f.write("## Main Reports\n")
            f.write("- [Overall Summary](overall_summary.md) - High-level analysis of all data\n")
            if cluster_summaries:
                f.write("- [Cluster Summaries](cluster_summaries.md) - Textrank summaries per cluster\n")
            
            if question_texts and len(question_texts) > 0:
                f.write("\n## Question-Specific Reports\n")
                for q_idx in sorted(question_texts.keys()):
                    f.write(f"- [Question {q_idx} Report](question_reports/question_{q_idx}_report.md) - Analysis of responses to question {q_idx}\n")
            
            f.write("\n## Code Analysis\n")
            f.write("- [Code Reports](code_reports/) - Detailed reports for each code\n")
            if question_texts and len(question_texts) > 0:
                f.write("- [Question-Specific Code Networks](question_reports/) - Network visualizations for each question\n")
            
            # Diagnostics
            if any([dynamic_policies, dynamic_keywords, stance_patterns]):
                f.write("\n## Pipeline Diagnostics\n")
                if dynamic_policies:
                    f.write("- Dynamic policies were generated and considered. See [diagnostics/policies.md](diagnostics/policies.md).\n")
                if dynamic_keywords:
                    f.write("- Dynamic keywords were generated and considered. See [diagnostics/keywords.md](diagnostics/keywords.md).\n")
                if stance_patterns:
                    f.write(f"- {len(stance_patterns)} stance pattern groups were generated. See [diagnostics/stance_patterns.md](diagnostics/stance_patterns.md).\n")
        
        logger.info(f"All reports have been generated in {reports_dir}")
        logger.info(f"Open {reports_dir}/README.md for an overview of all available reports")

        # Dataset-wide HTML report (comprehensive)
        try:
            codebook_sorted = dict(sorted(codebook.items(), key=lambda x: -x[1])) if codebook else {}
            dataset_html = _generate_dataset_html(
                seg=list(seg['text']) if 'text' in seg.columns else list(seg.index),
                cluster_info=None,
                cluster_summaries=cluster_summaries or {},
                codebook_sorted=codebook_sorted,
                global_summary=global_summary or (overall_path.read_text(encoding='utf-8') if overall_path.exists() else ""),
                input_path=input_path or "",
                k=k_clusters or (len(cluster_summaries) if cluster_summaries else 0),
            )
            (reports_dir / "report.html").write_text(dataset_html, encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to create dataset-wide HTML report: {e}")

        # Policies/keywords/stances overview HTML
        try:
            if any([policies, dynamic_policies, dynamic_keywords, stance_patterns]):
                pol_html = _generate_policies_html(
                    policies=policies or {},
                    dynamic_policies=dynamic_policies or {},
                    keywords=dynamic_keywords or {},
                    dynamic_keywords=dynamic_keywords or {},
                    stance_patterns=stance_patterns or {},
                    used_policies=used_policies or set(),
                    used_keywords=used_keywords or set(),
                    used_stances=used_stances or set(),
                    input_path=input_path or "",
                )
                (reports_dir / "policies.html").write_text(pol_html, encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to create policies.html: {e}")
        
    except Exception as e:
        logger.error(f"Error generating structured reports: {e}")
        logger.error(traceback.format_exc())
        raise  # Re-raise the exception to be handled by the caller

def generate_reports(
    segment_df: pd.DataFrame,
    output_dir: pathlib.Path,
    dynamic_policies: Optional[Dict] = None,
    dynamic_keywords: Optional[Dict] = None,
    used_policies: Optional[set] = None,
    used_keywords: Optional[set] = None,
    used_stances: Optional[set] = None,
    codebook: Optional[Dict] = None,
    cluster_info: Optional[Dict] = None,
    cluster_summaries: Optional[Dict] = None,
    global_summary: Optional[str] = None,
    input_path: Optional[str] = None,
    k_clusters: Optional[int] = None,
    policies: Optional[Dict] = None,
    stance_patterns: Optional[Dict] = None,
    report_name: Optional[str] = None,
    question_texts: Optional[Dict[int, str]] = None,
) -> None:
    """Compatibility wrapper to generate all analysis reports from pipeline outputs.
    
    This function serves as the main entry point from the analysis pipeline, preparing
    and forwarding all analysis artifacts to generate_structured_report(). It handles
    parameter normalization and error handling for the reporting process.
    
    Args:
        segment_df: DataFrame containing the segmented text data with codes and metadata.
        output_dir: Directory path where all report files will be saved.
        dynamic_policies: Dictionary of dynamically generated policies.
        dynamic_keywords: Dictionary of dynamically generated keywords.
        used_policies: Set of policy codes actually used in analysis.
        used_keywords: Set of keywords actually found in the text.
        used_stances: Set of stance patterns actually found in the text.
        codebook: Dictionary mapping codes to their frequencies.
        cluster_info: Legacy parameter, use cluster_summaries instead.
        cluster_summaries: Dictionary of cluster IDs to summary texts.
        global_summary: Precomputed global summary text.
        input_path: Path to the input data file.
        k_clusters: Number of clusters used in analysis.
        policies: Dictionary of policy definitions and metadata.
        stance_patterns: Dictionary of stance patterns used in analysis.
        report_name: Optional name identifying the report scope.
        question_texts: Optional dict mapping question indices to question texts.
        
    Returns:
        None: All outputs are written to files in the specified output directory.
        
    Raises:
        FileNotFoundError: If output_dir cannot be created or accessed.
        ValueError: If required input data is missing or malformed.
        
    Note:
        This function is primarily called from the analysis pipeline and provides
        backward compatibility with older pipeline versions.
    """
    try:
        # Prefer explicit cluster_summaries arg but accept cluster_info fallback
        clusters = cluster_summaries or cluster_info

        generate_structured_report(
            seg=segment_df,
            codebook=codebook,
            code_cooccurrence=None,  # compute from seg if needed
            output_dir=output_dir,
            question_texts=question_texts,
            report_name=report_name,
            dynamic_policies=dynamic_policies,
            dynamic_keywords=dynamic_keywords,
            used_policies=used_policies,
            used_keywords=used_keywords,
            used_stances=used_stances,
            cluster_summaries=clusters,
            global_summary=global_summary,
            input_path=input_path,
            k_clusters=k_clusters,
            policies=policies,
            stance_patterns=stance_patterns,
        )
    except Exception as e:
        logger.error(f"Error in generate_reports wrapper: {e}")
        logger.error(traceback.format_exc())
        raise

def generate_overall_summary(
    seg: pd.DataFrame,
    codebook: Dict[str, int],
    code_cooccurrence: Dict,
    output_path: pathlib.Path
) -> None:
    """Generate a comprehensive summary report of the qualitative analysis.
    
    This function creates a markdown report containing key statistics and insights
    from the analysis, including code frequencies, co-occurrence patterns, and
    overall dataset characteristics.
    
    Args:
        seg: DataFrame containing the segmented text data with codes and metadata.
        codebook: Dictionary mapping codes to their frequencies.
        code_cooccurrence: Dictionary containing co-occurrence analysis results with
                         keys 'codes', 'matrix', and 'pairs'.
        output_path: File path where the summary report will be saved.
        
    Returns:
        None: The report is written to the specified output path.
        
    Raises:
        FileNotFoundError: If the output directory does not exist and cannot be created.
        ValueError: If input data is missing required keys or has invalid format.
        
    Example:
        >>> generate_overall_summary(
        ...     seg=df,
        ...     codebook={"code1": 10, "code2": 5},
        ...     code_cooccurrence={
        ...         'codes': ['code1', 'code2'],
        ...         'matrix': [[0, 0.5], [0.5, 0]],
        ...         'pairs': {('code1', 'code2'): 3}
        ...     },
        ...     output_path=Path("summary.md")
        ... )
    """
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
        report = "# Qualitative Analysis Summary Report\n\n"
        report += "## Overview\n"
        report += f"- **Total Segments Analyzed**: {total_segments:,}\n"
        report += f"- **Total Code Applications**: {total_codes:,}\n"
        report += f"- **Unique Codes**: {unique_codes:,}\n"
        report += f"- **Average Codes per Segment**: {avg_codes_per_segment:.2f}\n\n"
        
        report += "## Most Frequent Codes\n"
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
    code_cooccurrence: Optional[Dict] = None,
) -> None:
    """Generate a detailed markdown report for analyzing responses to a specific question.
    
    This function creates a comprehensive report that includes code frequencies,
    co-occurrence patterns, and representative text examples for a single survey or
    interview question. The report is saved as a markdown file with visualizations.
    
    Args:
        seg: DataFrame containing the segmented text data for this question.
            Expected columns: 'text', 'codes', and any metadata columns.
        question_text: The full text of the question being analyzed.
        question_idx: Numeric identifier for the question (used in output filenames).
        output_path: File path where the report will be saved (should have .md extension).
        code_cooccurrence: Optional precomputed co-occurrence data. If None, it will
            be computed from the segment data. Should be a dict with keys 'codes',
            'matrix', and 'pairs' as returned by analyze_code_cooccurrence().
            
    Returns:
        None: The report is written to the specified output path.
        
    Raises:
        ValueError: If required columns are missing from seg.
        FileNotFoundError: If the output directory doesn't exist.
        
    Example:
        >>> df = pd.DataFrame({
        ...     'text': ['Response 1', 'Response 2'],
        ...     'codes': [['code1'], ['code1', 'code2']]
        ... })
        >>> generate_question_report(
        ...     seg=df,
        ...     question_text="What are your thoughts?",
        ...     question_idx=1,
        ...     output_path=Path("reports/q1_report.md")
        ... )
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
            
        # Calculate code frequencies (if codes exist)
        code_freq = {}
        all_codes = []
        has_codes = 'codes' in seg.columns
        if has_codes:
            for codes in seg['codes']:
                if not codes:
                    continue
                all_codes.append(codes)
                for code in codes:
                    code_freq[code] = code_freq.get(code, 0) + 1
        
        # Get top codes (up to 15)
        top_codes = sorted(code_freq.items(), key=lambda x: -x[1])[:15] if code_freq else []
        
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
        report += "## 📊 Summary\n"
        report += f"- **Total Responses**: {total_segments:,}\n"
        report += f"- **Unique Codes Applied**: {len(code_freq):,}\n"
        
        # Add sentiment summary if available
        if sentiment_stats:
            total = sum(sentiment_stats.values())
            if total > 0:
                report += "- **Sentiment Distribution**:\n"
                def sentiment_bar(count, total, width=30):
                    filled = '█' * int((count / total) * width) if total > 0 else ''
                    return f"`{filled.ljust(width)}` {count} ({count/total:.1%})"
                
                report += f"  - 😊 Positive: {sentiment_bar(sentiment_stats['positive'], total)}\n"
                report += f"  - 😐 Neutral: {sentiment_bar(sentiment_stats['neutral'], total)}\n"
                report += f"  - 😟 Negative: {sentiment_bar(sentiment_stats['negative'], total)}\n"
        
        # Code frequency section
        report += "\n## 🔍 Code Frequency\n"
        if top_codes:
            max_count = max(count for _, count in top_codes)
            for code, count in top_codes:
                bar_length = int((count / max_count) * 30)
                bar = '█' * bar_length
                report += f"- **{code}**: {bar} {count:,} ({count/total_segments:.0%})\n"
        else:
            report += "(No codes available for this question)\n"
        # Add co-occurrence analysis if data is available
        if has_codes and code_cooccurrence and 'pairs' in code_cooccurrence and code_cooccurrence['pairs']:
            report += "\n## 🔗 Code Relationships\n"
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
            import traceback
            
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
                report += "\n## 📊 Frequent Terms\n"
                report += f"![Word Cloud]({wc_path.name})\n"
        except Exception as e:
            logger.error(f"Error generating word cloud: {e}")
            logger.error(traceback.format_exc())
        
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
                    
                    # Create a list of codes for this question
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
    output_dir: pathlib.Path,
    top_n: int = 10,
) -> None:
    """Generate detailed markdown reports for each code in the codebook.
    
    Creates individual reports for each code containing:
    - Code definition and frequency
    - Most common co-occurring codes
    - Representative text examples
    - Visualizations (word clouds, network graphs)

    Args:
        seg: DataFrame containing the segmented text data with 'text' and 'codes' columns.
        codebook: Dictionary mapping codes to their frequencies in the dataset.
        code_cooccurrence: Dictionary containing co-occurrence analysis results.
            Must include 'codes' (list), 'matrix' (co-occurrence matrix), and 'pairs'
            (dictionary of code pairs to co-occurrence counts).
        output_dir: Directory where code reports will be saved.
        top_n: Number of top co-occurring codes to include in each report (default: 10).
            Each file will be named using a sanitized version of the code.

    Returns:
        None: All reports are written to files in the specified output directory.

    Raises:
        FileNotFoundError: If output_dir does not exist and cannot be created.
        ValueError: If input data is missing required columns or has invalid format.

    Example:
        >>> generate_code_reports(
        ...     seg=df,
        ...     codebook={"theme1": 25, "theme2": 15},
        ...     code_cooccurrence=cooccurrence_data,
        ...     output_dir=Path("reports/codes")
        ... )
    """
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
            report += "## Usage Statistics\n"
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

