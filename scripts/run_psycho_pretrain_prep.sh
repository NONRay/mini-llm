#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-$ROOT/../models/Qwen3-8B}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/psycho_prep_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "[start] $(date '+%F %T')"
echo "[model] $MODEL_PATH"
echo "[log] $LOG_FILE"

python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA 不可用；请在能访问 GPU 的 SSH 终端运行，任务未启动。")
torch.empty(1, device="cuda:0")
print(f"[gpu] {torch.cuda.get_device_name(0)}", flush=True)
PY

python "$ROOT/scripts/prepare_psycho_data.py" \
  --input-dir "$ROOT/data/psycho/markdown" \
  --output "$ROOT/data/raw/psycho_train.txt"

python "$ROOT/scripts/quality_filter_psycho.py" \
  --input-dir "$ROOT/data/psycho/markdown" \
  --output "$ROOT/data/raw/psycho_train_qc.txt" \
  --audit "$ROOT/data/raw/psycho_qc_audit.jsonl" \
  --model "Qwen3-8B" \
  --hf-model-path "$MODEL_PATH"

echo "[done] $(date '+%F %T')"
