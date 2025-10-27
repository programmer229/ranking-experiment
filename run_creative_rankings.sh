#!/usr/bin/env bash
set -euo pipefail

DATASET_PATH=${1:-data/reedsy_dataset.json}

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "Dataset not found at ${DATASET_PATH}" >&2
  echo "Provide a creative-writing dataset JSON (e.g. from scrape_reedsy.py)." >&2
  exit 1
fi
python main.py \
  --dataset creative_writing \
  --solutions-json "${DATASET_PATH}" \
  --ranking-scheme group \
  --judge llm \
  --summary-output summary_group.json \
  --llm-model gpt-5-nano


python main.py \
  --dataset creative_writing \
  --solutions-json "${DATASET_PATH}" \
  --ranking-scheme elo \
  --judge llm \
  --summary-output summary3.json \
  --llm-model gpt-5-nano

python main.py \
  --dataset creative_writing \
  --solutions-json "${DATASET_PATH}" \
  --ranking-scheme pairwise \
  --judge llm \
  --summary-output summary.json \
  --llm-model gpt-5-nano


python main.py \
  --dataset creative_writing \
  --solutions-json "${DATASET_PATH}" \
  --ranking-scheme llm_score \
  --judge llm \
  --summary-output summary2.json \
  --llm-model gpt-5-nano

echo "Pairwise summary -> summary.json"
echo "Group summary    -> summary_group.json"
echo "Score summary    -> summary2.json"
