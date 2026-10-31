#!/usr/bin/env bash
# Bayesian within-set field models. Usage: run_field_within.sh [variant] [fits...]; defaults: journal, journal field then paper field.
set -euo pipefail
cd "$(dirname "$0")/.."
export VOP_VARIANT="${1:-journal}" PATH="$HOME/.local/bin:$PATH"
shift || true
L=/data/openalex_project_storage/vop/derived_$VOP_VARIANT/logs
[ $# -gt 0 ] || set -- journal_within field_within
for fit in "$@"; do
  uv run --no-sync vop s07-hurdle --k 3 --bayes-only --fit "$fit" --draws 750 --tune 750 --chains 4 > "$L/s07_$fit.log" 2>&1
  uv run --no-sync vop s07-field-plot --k 3 --fit "$fit" > "$L/s07_${fit}_plot.log" 2>&1
  echo DONE > "$L/${fit}_done"
done
