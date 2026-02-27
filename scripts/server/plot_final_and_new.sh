#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "$REPO_ROOT/experiments/final_delivery/figures"
mkdir -p "$REPO_ROOT/experiments/new/figures"

"$PYTHON_BIN" experiments/plot_results.py \
  --data-dir "$REPO_ROOT/experiments/final_delivery/data" \
  --out-dir "$REPO_ROOT/experiments/final_delivery/figures" \
  --tag "final_delivery"

"$PYTHON_BIN" experiments/plot_results.py \
  --data-dir "$REPO_ROOT/experiments/new/data" \
  --out-dir "$REPO_ROOT/experiments/new/figures" \
  --tag "new"

echo "[done] figures generated for final_delivery and new"
