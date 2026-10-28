# Value of Philosophy — rebuild plan (2026-10-28)

Goal unchanged: do papers by philosophers that appear in non-philosophy journals get cited
differently than matched controls (same journal, ±years, ±authors)?

This rebuild has three aims: (1) develop locally in VS Code with Claude Code over SSH, with nothing
lost when the SURF workspace is deleted; (2) everything in git/GitHub; (3) switch from the gzipped
NDJSON snapshot to the OpenAlex Parquet snapshot and fix the known mistakes of the notebook version.

---

## 0. Decisions (the short version)

| Question | Decision |
|---|---|
| Where does code live? | Git repo `value_of_philosophy` on GitHub. Cloned on the remote at `~/value_of_philosophy`. Local Mac clone only for reading/offline edits. |
| Where does data live? | A SURF Research Cloud **external storage volume** (created *before* the workspace, attached at creation). Raw snapshot + derived parquet go there, never in the workspace root disk. |
| How do we survive workspace deletion? | Code → GitHub. Data → the volume. Environment → `bootstrap.sh` + `uv.lock` rebuilds a fresh workspace in ~10 min. Nothing of value stays on the workspace disk. |
| Where does Claude Code run? | On the remote, inside VS Code Remote-SSH (the extension runs on the remote host, so it sees the data). |
| Notebooks or scripts? | Scripts (`src/vop/*.py`, one stage per file, CLI via `typer`). Notebooks only for exploration, stripped of outputs on commit. |
| Query engine | DuckDB over Parquet. One full scan of `works` to build a slim table; every later stage runs on the slim table in seconds/minutes. No more hand-rolled multiprocessing gzip/orjson scanners. |
| Long jobs | Run under `tmux` (or `nohup`) so a dropped VS Code connection does not kill them. Every stage is idempotent and writes to `data/derived/<stage>/`. |

---

## 1. Development environment (VS Code + SSH + Claude Code)

1. **SURF Research Cloud setup**
   - Dashboard → *Create new storage*: one volume (size: ~1.2 TB if you keep the full works parquet
     locally; ~300 GB if you only keep the slim table and re-stream from S3). Storage is cheap in credits
     compared with a running VM.
   - Create the workspace (Ubuntu, ≥16 cores / ≥128 GB RAM like before) and **attach the volume at creation**
     (docs say it cannot be attached afterwards for some catalog items). Mount point e.g. `/data`.
   - SSH key: added through your SRAM profile; SRC injects it. Add a `Host surf` entry to `~/.ssh/config`
     on the Mac (`HostName <workspace-ip>`, `User mnoichl`, `IdentityFile ...`, `ServerAliveInterval 60`).
   - Verify the volume outlives the workspace with the SURF service desk before trusting it with
     irreplaceable derived data; additionally mirror `data/derived/` to Research Drive or a cheap object
     store (`rclone sync`) after each milestone.

2. **VS Code**
   - Install *Remote - SSH* and the *Claude Code* extension locally. Connect to `surf`, open
     `~/value_of_philosophy`. VS Code installs its server on the remote; the Claude Code extension runs there too.
   - On the remote: install Claude Code (`curl -fsSL https://claude.ai/install.sh | bash`) and log in once.
     Also install `gh` and run `gh auth login` so commits can be pushed from the remote.

3. **Reproducible environment**
   - `bootstrap.sh` (in the repo): installs `uv`, `aws` CLI, `duckdb` CLI, `tmux`, `gh`; `uv sync` from
     `pyproject.toml`/`uv.lock`; checks that `/data` is mounted; creates `data/` symlink → `/data/vop`.
   - `pyproject.toml` pins: duckdb, polars, pyarrow, pandas, pyfixest (or linearmodels), statsmodels,
     matplotlib, typer, pytest.
   - `.env` / `config.toml` (git-ignored, templated by `config.example.toml`) holds the only machine-specific
     thing: `SNAPSHOT_ROOT`, `DATA_ROOT`, `THREADS`, `MEMORY_LIMIT`.

4. **Local dev loop**
   - Every stage accepts `--snapshot-root`, so the same code runs against a small dev sample
     (`data/dev_sample/`) on the remote, or on the Mac if you copy the sample down. `make dev-sample`
     builds it from 2–3 parquet part files of `works` plus the full `sources` table.
   - `pytest` tests on the dev sample guard the ISSN matching, the philosopher-share computation and the
     matching join (these were the error-prone bits).

---

## 2. Git / GitHub

- `git init` locally now, first commit = this plan + `.gitignore` + the two old notebooks under
  `old_version/` (as a record, nothing more). Create the GitHub repo (`gh repo create ... --private`), push.
- `.gitignore`: `data/`, `*.duckdb`, `*.parquet`, `*.ndjson*`, `*.zip`, `.env`, `config.toml`, `.venv/`.
- `nbstripout` via pre-commit so notebooks never commit outputs (the old ones are 2 MB of outputs).
- Branch per pipeline stage, PRs reviewed with `/code-review`, `main` always runnable end to end on the
  dev sample (a GitHub Action can run `pytest` on a tiny committed sample).
- Derived results that matter for the paper (final matched frame, regression tables, figures) are
  tracked with **DVC** or simply uploaded as GitHub release assets once they are stable. Nothing >50 MB in git.

Proposed layout:
```
value_of_philosophy/
├── PLAN.md  README.md  pyproject.toml  uv.lock  bootstrap.sh  Makefile
├── config.example.toml
├── src/vop/
│   ├── config.py          # paths, duckdb connection with threads/memory
│   ├── s00_download.py    # aws s3 sync of parquet works+sources(+authors), manifest check
│   ├── s01_sources.py     # philosophy source_ids from PhilPapers ISSN list
│   ├── s02_works_slim.py  # the ONE full scan: slim works table
│   ├── s03_philosophers.py
│   ├── s04_focal_works.py
│   ├── s05_matching.py
│   ├── s06_analysis.py
│   └── sql/               # .sql files for the big queries
├── tests/
├── notebooks/             # exploration only, stripped
├── data -> /data/vop      # symlink, git-ignored
│   ├── raw/openalex/data/parquet/{works,sources,authors}
│   ├── derived/s01 ... s06
│   └── dev_sample/
├── inputs/philpapers_journals_with_issn.csv   # the one small input that IS committed
└── old_version/                # old notebooks, read-only
```

---

## 3. OpenAlex Parquet snapshot

Facts (official help page, 2026): bucket `s3://openalex/`, parquet under `data/parquet/{entity}/updated_date=*/part_*.parquet`,
≤400k records per file, a `manifest.json` per entity written last (present ⇒ complete),
`works/deleted_ids.csv.gz` and `merged_ids.csv.gz`. Each row is the full API-shaped entity, i.e. nested
columns (`authorships` = list of structs, `primary_location.source.*`, `counts_by_year`, `referenced_works`...).
The 2025 flat-NDJSON layout you used is now under `legacy-data/`.

- **Download** only what we need, resumable:
  ```
  aws s3 sync s3://openalex/data/parquet/works   /data/vop/raw/openalex/data/parquet/works   --no-sign-request
  aws s3 sync s3://openalex/data/parquet/sources /data/vop/raw/openalex/data/parquet/sources --no-sign-request
  ```
  Then confirm `manifest.json` record counts equal `SELECT count(*) FROM read_parquet(...)`.
  (Alternative if the volume is small: DuckDB `httpfs` reads S3 directly with projection pushdown; slower but
  needs no local copy. Decide after a 10-file timing test.)
- **First task on contact**: `DESCRIBE SELECT * FROM read_parquet('.../works/updated_date=*/part_0000.parquet' LIMIT 1)`
  and pin the schema in `docs/schema_works.md`; all SQL is written against the real column names, not guesses.
- **Why this is a big win**: DuckDB reads only the columns a query touches. The old pipeline parsed every
  JSON line fully three times (ISSN slices, author-works scan, matches scan) with custom multiprocessing.
  Now: one scan producing a slim table (~270M rows × ~12 columns, est. 20–40 GB), then everything else is
  joins on that.

---

## 4. Pipeline redesign (and the mistakes each stage fixes)

**s01 philosophy sources.** Join the PhilPapers ISSN list against the `sources` table on `issn_l` *and* the
`issn` array → canonical set of philosophy `source_id`s (+ display names for eyeballing). Fixes: ISSN string
normalisation scattered across cells, journals matched on one ISSN only, `journal_key` built from "first
element of the issn list" which could disagree with `issn_l`.

**s02 slim works table.** One DuckDB `COPY (...) TO parquet` over all works with:
`id, publication_year, type, is_retracted, language, primary_location.source.id AS source_id, issn_l,
list_transform(authorships, a -> a.author.id) AS author_ids, len(authorships) AS n_authors,
cited_by_count, counts_by_year (kept as list), len(referenced_works) AS n_refs, primary_topic.subfield.id/name,
primary_topic.field, updated_date`. Dedupe by `id` keeping latest `updated_date`; drop ids in
`deleted_ids`; map `merged_ids`. Fixes: no handling of merged/deleted works; three full scans; CSV round-trips
that turned list columns into strings and forced `ast.literal_eval` hacks.

**s03 philosophers.** `UNNEST(author_ids)` over the slim table, count works total and works in philosophy
sources per author, share = in/total, keep `total ≥ N` and `share > 0.5` (N and the threshold are
parameters; write out the full distribution so sensitivity analyses are free). Fixes the worst bug in
the old notebook: `authors_df.drop_duplicates(subset=["paper_id"])` kept only the *first author* of each
philosophy paper, so the candidate pool was first authors only. Also fixes `df['type'] >= 'article'`
(lexicographic string comparison, not a type filter).

**s04 focal works.** Works with ≥1 philosopher author, `source_id` NOT in philosophy sources,
`type = 'article'`, year window (e.g. 1970–2022, so citations have time to accrue), subfield not
Philosophy / HPS, `is_retracted = false`. Also record *how many* philosopher authors and whether the
philosopher is first/last author (useful moderators).

**s05 matching.** Pure SQL: candidates = slim works in the same `source_id`, `type='article'`,
`|year diff| ≤ k`, `|n_authors diff| ≤ k`, **no philosopher author** (the old sampler did not screen
controls for philosopher co-authors), not a focal work, not retracted. Sample `m` controls per focal work
with a fixed seed (`ORDER BY hash(id, seed)`), for k ∈ {1,2,3} as nested scenarios. Output one tidy
stacked parquet: `set_id, work_id, is_focal, scenario, year, n_authors, source_id, cited_by_count, cites_5y`.
Fixes: typo `"Philsopher"` in the label map, which after a CSV round-trip (booleans → strings) relabelled
every focal paper as a control in the regression; chained assignment `df['group'][mask] = ...` that
silently may not write.

**s06 analysis.**
- Outcome: both all-time `cited_by_count` and a fixed citation window (`cites_5y` from `counts_by_year`),
  because controls up to ±3 years older have had more time to be cited.
- Models: within-set fixed effects via `pyfixest` (`feols('log1p_cites ~ is_focal | set_id', vcov={'CRV1':'set_id'})`)
  and a Poisson FE (`fepois`) as the count-appropriate version. This replaces
  `smf.ols('... + C(selected_work_id)')`, which built tens of thousands of dummy columns and never finished.
- Descriptives + plots as before, plus heterogeneity by field, period, and philosopher position.
- All tables written to `data/derived/s06/` and figures to `reports/`.

---

## 5. Order of work

1. **Today, local**: `git init`, `.gitignore`, move notebooks to `old_version/`, commit, create GitHub repo, push.
2. **SRC**: create volume, create workspace with volume attached, SSH config, VS Code Remote-SSH, install
   Claude Code + gh on the remote, clone repo, run `bootstrap.sh`.
3. **Snapshot**: `s00_download` for `sources` first (small, minutes) → write `s01`. Start the `works`
   parquet sync in tmux (hours; ~600 GB). Meanwhile build `dev_sample/` from the first synced part files and
   write `s02`–`s05` against it, with tests.
4. **Full run**: `s02` on all works (one scan; expect ~1–3 h on 15 cores), then `s03`–`s05` (minutes).
5. **Analysis**: `s06`, compare with the old numbers (controls mean 24.8 / median 5 vs philosophers
   19.6 / 4) and understand any difference (the first-author bug alone should change the philosopher set a lot).
6. Mirror `data/derived/` off the volume, tag a release.

## 6. Open questions for you

- Is the remote from last time still a SURF Research Cloud workspace, and does it already have an external
  volume? (If the old `value_of_philosophy_datasets/` is on the workspace disk only, copy it to a volume or
  Research Drive *before* anything else.)
- GitHub: personal account or an org, private for now?
- Fresh snapshot (latest parquet release) or try to stay on the 2025-11-03 one? Parquet only exists for the
  new releases, so this is effectively a fresh snapshot and all numbers will shift slightly. Recommended: fresh.
