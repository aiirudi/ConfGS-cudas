#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

MODEL_DIR="/workspace/RFGS/output/playroom"

python metrics.py -m "$MODEL_DIR"