#!/bin/bash
set -e

if [ -z "$1" ]; then
  echo "Usage: ./scripts/run_training.sh <train_data.json> [output_dir]"
  exit 1
fi
TRAIN_DATA=$1
OUTPUT_DIR=${2:-./models/lora}

python training/pipeline.py --train-data "$TRAIN_DATA" --output-dir "$OUTPUT_DIR"
