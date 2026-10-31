#!/usr/bin/env bash
# Bayesian within-set field models on the journals-only data: journal field first, then paper field.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH" VOP_VARIANT=journal
L=/data/openalex_project_storage/vop/derived_journal/logs
for fit in journal_within field_within; do
  uv run --no-sync vop s07-hurdle --k 3 --bayes-only --fit "$fit" --draws 750 --tune 750 --chains 4 > "$L/s07_$fit.log" 2>&1
  uv run --no-sync vop s07-field-plot --k 3 --fit "$fit" > "$L/s07_${fit}_plot.log" 2>&1
  echo DONE > "$L/${fit}_done"
done
