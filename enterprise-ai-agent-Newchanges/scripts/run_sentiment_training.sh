#!/bin/bash
set -e

TRAIN_DATA=${1:-./training/examples/sentiment_dataset.json}
OUTPUT_PATH=${2:-./models/sentiment/naive_bayes.json}

python training/train_sentiment.py --train-data "$TRAIN_DATA" --output "$OUTPUT_PATH"
