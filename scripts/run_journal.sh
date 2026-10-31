#!/usr/bin/env bash
# Journals-only pipeline. Waits for the metadata scan, keeps memory low while another job holds the machine.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH" VOP_VARIANT=journal
L=/data/openalex_project_storage/vop/derived_journal/logs; mkdir -p "$L"
vop() { uv run --no-sync vop "$@"; }
until grep -q "^EXIT=0" /data/openalex_project_storage/vop/derived/s02b/run.log 2>/dev/null; do sleep 60; done
export VOP_MEMORY=8GB VOP_THREADS=3
vop s02j-journal-slim > "$L/s02j.log" 2>&1
vop s03-philosophers  > "$L/s03.log" 2>&1
vop s03-plots         > "$L/s03_plots.log" 2>&1
vop s04-focal-works   > "$L/s04.log" 2>&1
while tmux has-session -t bfull 2>/dev/null; do sleep 60; done
unset VOP_MEMORY VOP_THREADS
vop s05-matching      > "$L/s05.log" 2>&1
vop s08-prestige      > "$L/s08.log" 2>&1
vop s06-analysis      > "$L/s06.log" 2>&1
vop s07-hurdle --k 3  > "$L/s07.log" 2>&1
echo DONE > "$L/done"
