#!/usr/bin/env bash
# Bayesian field model on the journals-only data; starts when the journal chain has finished.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH" VOP_VARIANT=journal
L=/data/openalex_project_storage/vop/derived_journal/logs
until [ -f "$L/done" ]; do sleep 60; done
uv run --no-sync vop s07-hurdle --k 3 --bayes-only --fit field_phil --draws 500 --tune 500 --chains 4 > "$L/s07_field_bayes.log" 2>&1
uv run --no-sync vop s07-field-plot --k 3 > "$L/s07_field_plot.log" 2>&1
echo DONE > "$L/field_done"
