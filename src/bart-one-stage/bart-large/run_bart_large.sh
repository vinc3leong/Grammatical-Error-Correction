#!/usr/bin/env bash
set -euo pipefail

echo "Starting BART GEC Training"
date

DATA_ROOT="$HOME/NLP/BART/data"
MODEL_NAME="facebook/bart-large"
OUT_NAME="bart-large-gec"

EPOCHS=10
LR=2e-5
TRAIN_BATCH_SIZE=16
EVAL_BATCH_SIZE=24
GRAD_ACCUM=4
MAX_LEN=128                
BEAM=5           
DROPOUT=0.3
PATIENCE=5
LR_SCHEDULER="linear"

SCRATCH_DIR="${SLURM_TMPDIR:-/tmp/$USER/${SLURM_JOB_ID:-interactive}}"
mkdir -p "$SCRATCH_DIR"/{tmp,pip-cache,hf,out}
export TMPDIR="$SCRATCH_DIR/tmp"
export PIP_CACHE_DIR="$SCRATCH_DIR/pip-cache"
export HF_HOME="$SCRATCH_DIR/hf"
export TRANSFORMERS_CACHE="$HF_HOME"
export PATH="$SCRATCH_DIR/venv/bin:$PATH"

python3 -m venv "$SCRATCH_DIR/venv"
source "$SCRATCH_DIR/venv/bin/activate"
pip install -U pip wheel setuptools

pip install "torch>=2.2" "transformers>=4.43" "accelerate>=0.33" \
            "datasets>=2.20" "evaluate>=0.4" "sacrebleu>=2.0" \
            "sentencepiece>=0.1.99" "numpy>=1.24"

pip install "errant>=3.0" "spacy>=3.7"
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl

RUNS_DIR="$HOME/NLP/BART/runs"
RUN_STAMP="$(date +'%Y%m%d-%H%M%S')"
OUT_DIR_LOCAL="$SCRATCH_DIR/out/${OUT_NAME}-${RUN_STAMP}"
mkdir -p "$OUT_DIR_LOCAL"

echo "Starting training..."

python "$HOME/NLP/BART/bart_final.py" \
  --data_root "$DATA_ROOT" \
  --model_name "$MODEL_NAME" \
  --output_dir "$OUT_DIR_LOCAL" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --batch_size "$TRAIN_BATCH_SIZE" \
  --eval_batch_size "$EVAL_BATCH_SIZE" \
  --grad_accum "$GRAD_ACCUM" \
  --dropout "$DROPOUT" \
  --lr_scheduler_type "$LR_SCHEDULER" \
  --max_src_len "$MAX_LEN" \
  --max_tgt_len "$MAX_LEN" \
  --beam "$BEAM" \
  --fp16 \
  --grad_ckpt \
  --early_stopping_patience "$PATIENCE"

DEST_DIR="$RUNS_DIR/${OUT_NAME}-${RUN_STAMP}"
mkdir -p "$DEST_DIR"
rsync -ah --info=progress2 "$OUT_DIR_LOCAL"/ "$DEST_DIR"/

echo "Training complete!"
echo "Output: $DEST_DIR"
date
