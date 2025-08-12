#!/usr/bin/env bash
set -euo pipefail
INPUT=${1:-data/input.csv}
OUT=${2:-out}
python worker/pipeline.py --input "$INPUT" --out "$OUT"
echo "Report: $OUT/report.html"
