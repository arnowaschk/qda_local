# QDA Local – Fully-automated, local QDA

Pipeline: Ingestion → Segmentation → Embeddings/TF-IDF → Clustering → Rule-based Auto-Codes → NER/Sentiment → LLM-free summaries → Reports & REFI-QDA export.

Quick start:
  docker compose up --build
  curl -X POST "http://localhost:8000/analyze?input_path=/app/data/input.csv&out_dir=/app/out"

Outputs:
  out/report.html, coded_segments.csv, codebook.json, themes.json, summaries.json
  out/exports/maxqda_import.csv, out/exports/atlasti_import.csv, out/exports/qda_export.qdpx
