#!/usr/bin/env bash
# Rebuilds a fresh SURF Research Cloud workspace (Ubuntu) for this project.
# Idempotent: safe to rerun.
set -euo pipefail
cd "$(dirname "$0")"

DATA_ROOT="${DATA_ROOT:-/data/openalex_project_storage/vop}"
export AWS_DEFAULT_REGION=us-east-1

echo "== apt packages"
sudo apt-get update -qq
sudo apt-get install -y -qq tmux unzip curl git gh htop

echo "== uv (python env manager)"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "== aws cli (anonymous S3 download of the OpenAlex snapshot)"
if ! command -v aws >/dev/null; then
  tmp=$(mktemp -d)
  curl -sS "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "$tmp/awscliv2.zip"
  unzip -q "$tmp/awscliv2.zip" -d "$tmp" && sudo "$tmp/aws/install"
  rm -rf "$tmp"
fi

echo "== duckdb cli"
if ! command -v duckdb >/dev/null; then
  curl -fsSL https://install.duckdb.org | sh
fi

echo "== python environment"
uv sync --extra dev

echo "== data volume"
if [ ! -d "$(dirname "$DATA_ROOT")" ]; then
  echo "!! $(dirname "$DATA_ROOT") is not mounted. Attach the external storage volume first." >&2
  exit 1
fi
mkdir -p "$DATA_ROOT"/{raw,derived,dev_sample,tmp}
[ -e data ] || ln -s "$DATA_ROOT" data
[ -f config.toml ] || cp config.example.toml config.toml

echo "== git hooks"
uv run pre-commit install || true

echo "Done. Next: 'gh auth login', edit config.toml, then 'uv run vop --help'."
