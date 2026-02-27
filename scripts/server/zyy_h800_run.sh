#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/server/zyy_h800_run.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MODEL_CACHE_DIR="/data/zhangyuyun/models/models"
MODEL_STABLE_DIR="/data/zhangyuyun/models/models/Qwen/Qwen3-8B"
KV_ROOT="/data/zhangyuyun/kvcache_index"
NEW_OUT_DIR="$REPO_ROOT/experiments/new"

PYTHON_BIN=""
UV_BIN=""

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
    return
  fi

  echo "[setup] uv not found, installing into user home..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
  if ! command -v uv >/dev/null 2>&1; then
    echo "[error] uv installation failed."
    exit 1
  fi
  UV_BIN="$(command -v uv)"
}

setup_venv() {
  local venv_dir="$REPO_ROOT/.venv-zyy"
  if [[ ! -d "$venv_dir" ]]; then
    echo "[setup] creating virtual env at $venv_dir"
    "$UV_BIN" venv --python 3.10 "$venv_dir"
  fi
  # shellcheck disable=SC1091
  source "$venv_dir/bin/activate"
  PYTHON_BIN="$(command -v python)"
  echo "[setup] using python: $PYTHON_BIN"
}

install_deps() {
  echo "[setup] installing project and runtime deps"
  "$UV_BIN" pip install --upgrade pip setuptools wheel

  # Install editable package without forcing heavy build dependencies first.
  "$UV_BIN" pip install -e . --no-deps

  "$UV_BIN" pip install modelscope pandas matplotlib seaborn psutil tqdm termcolor

  # Ensure core runtime packages are available.
  if ! "$PYTHON_BIN" -c "import torch" >/dev/null 2>&1; then
    "$UV_BIN" pip install torch
  fi
  if ! "$PYTHON_BIN" -c "import transformers" >/dev/null 2>&1; then
    "$UV_BIN" pip install transformers
  fi
  if ! "$PYTHON_BIN" -c "import xxhash" >/dev/null 2>&1; then
    "$UV_BIN" pip install xxhash
  fi

  # Optional acceleration dependencies.
  if ! "$PYTHON_BIN" -c "import flash_attn" >/dev/null 2>&1; then
    echo "[warn] flash_attn not detected. Trying installation..."
    "$UV_BIN" pip install flash-attn --no-build-isolation || echo "[warn] flash-attn install failed, continue with existing runtime."
  fi
  if ! "$PYTHON_BIN" -c "import triton" >/dev/null 2>&1; then
    echo "[warn] triton not detected. Trying installation..."
    "$UV_BIN" pip install triton || echo "[warn] triton install failed, continue with existing runtime."
  fi

  # GPUDirect userspace library (optional)
  if ! "$PYTHON_BIN" -c "import kvikio" >/dev/null 2>&1; then
    echo "[warn] kvikio not detected. Trying installation..."
    "$UV_BIN" pip install kvikio || echo "[warn] kvikio install failed, GDS path will fallback automatically."
  fi
}

download_model() {
  echo "[setup] downloading Qwen3-8B from ModelScope"
  mkdir -p "$MODEL_CACHE_DIR"
  "$PYTHON_BIN" scripts/server/download_qwen3_8b_modelscope.py \
    --model-id "Qwen/Qwen3-8B" \
    --cache-dir "$MODEL_CACHE_DIR" \
    --target-dir "$MODEL_STABLE_DIR"
}

run_ablations() {
  echo "[run] running full ablations to $NEW_OUT_DIR"
  mkdir -p "$KV_ROOT"
  mkdir -p "$NEW_OUT_DIR"

  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  export NANOVLLM_MODEL_PATH="$MODEL_STABLE_DIR"
  export NANOVLLM_KV_DIR="$KV_ROOT"

  # Recommended runtime switches for this branch.
  export NANOVLLM_USE_CSR_KV=1
  export NANOVLLM_GPU_HOT_CACHE_ENABLE=1
  export NANOVLLM_GPU_HOT_CACHE_GB="${NANOVLLM_GPU_HOT_CACHE_GB:-2.0}"
  export NANOVLLM_CSR_DATA_ALIGN_BYTES=4096
  export NANOVLLM_CSR_RECORD_ALIGN_BYTES=512
  export NANOVLLM_CSR_COMMIT_INTERVAL=32
  export NANOVLLM_ENABLE_TASK_CALIBRATION=1

  "$PYTHON_BIN" experiments/new/run_all_ablations.py \
    --data-path "$REPO_ROOT/data/imdb.csv" \
    --model-path "$MODEL_STABLE_DIR" \
    --kv-dir "$KV_ROOT" \
    --out-dir "$NEW_OUT_DIR" \
    --limit "${ABLT_LIMIT:-64}" \
    --task-limit "${ABLT_TASK_LIMIT:-48}" \
    --repeats "${ABLT_REPEATS:-2}"
}

plot_results() {
  echo "[plot] generating figures for final_delivery and new"
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
}

main() {
  ensure_uv
  setup_venv
  install_deps
  download_model
  run_ablations
  plot_results
  echo "[done] all ablations and figures completed."
}

main "$@"
