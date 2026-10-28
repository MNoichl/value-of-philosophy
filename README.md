# Value of Philosophy

Do papers by philosophers that appear in non-philosophy journals get cited differently than
matched controls from the same journal, years and author count? Built on the OpenAlex Parquet snapshot.

See [PLAN.md](PLAN.md) for the design and the stage-by-stage pipeline.

## Quick start (on the SURF Research Cloud workspace)

```bash
git clone git@github.com:<you>/value_of_philosophy.git ~/value_of_philosophy
cd ~/value_of_philosophy
./bootstrap.sh            # installs uv, aws cli, duckdb cli, tmux; creates .venv; links data -> /data/openalex_project_storage/vop
cp config.example.toml config.toml   # then edit paths/threads
uv run vop --help
```

Pipeline stages live in `src/vop/` as `s00_download` ... `s06_analysis`; each writes to `data/derived/<stage>/`
and can be rerun on a dev sample with `--snapshot-root data/dev_sample`.

`old_version/` holds the 2025 Jupyter notebooks for reference only. Their known bugs are listed in PLAN.md §4.
