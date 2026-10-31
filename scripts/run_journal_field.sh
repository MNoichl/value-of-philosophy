#!/usr/bin/env bash
# Journal-and-field matching: a variant sharing s01-s04 with the journals-only pipeline.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH" VOP_VARIANT=journal_field
B=/data/openalex_project_storage/vop/derived_journal
D=/data/openalex_project_storage/vop/derived_journal_field; L=$D/logs; mkdir -p "$L"
for s in s01 s02 s03 s04; do ln -sfn "$B/$s" "$D/$s"; done
vop() { uv run --no-sync vop "$@"; }
vop s05-matching --by-field > "$L/s05.log" 2>&1
until [ -f "$B/logs/field_within_done" ]; do sleep 60; done  # the running fit needs the memory
vop s08-prestige > "$L/s08.log" 2>&1
PYTHONPATH=src .venv/bin/python scripts/diagnostics/field_matched_compare.py > "$L/compare.log" 2>&1
vop s06-analysis > "$L/s06.log" 2>&1
vop s07-hurdle --k 3 > "$L/s07.log" 2>&1
echo DONE > "$L/done"
