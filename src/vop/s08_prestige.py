"""Stage s08: coauthor prestige.

For every author on every paper in the matched sets (s05), measure prestige at publication time as
the all-time citations to the works that author published BEFORE the paper's year (so the paper
itself and later work do not enter). That count is ranked as a percentile within the paper's field
and publication year, among all authors appearing in the matched sets. Per paper we keep
  prestige_max        highest percentile among all authors
  prestige_max_nonphil highest percentile among authors who are not philosophers (s03)
  n_prior_works_max   prior works of the most prolific author (a career-length proxy)
Output: data/derived/s08/paper_prestige.parquet  (work_id, k-independent)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import typer

from vop.config import Config, connect, load_config


def _step(con, name, sql, path, t0):
    con.execute(f"COPY ({sql}) TO '{path}' (FORMAT parquet, COMPRESSION zstd)")
    n = con.execute(f"SELECT count(*) FROM read_parquet('{path}')").fetchone()[0]
    typer.echo(f"[{(time.time() - t0) / 60:6.1f} min] {name}: {n:,} rows")
    sys.stdout.flush()


def run(cfg: Config, slim_dir: Path, s03_dir: Path, s05_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "tmp"
    tmp.mkdir(exist_ok=True)
    slim_glob = str(slim_dir / "*" / "*.parquet")
    sets = s05_dir / "matched_sets.parquet"
    con = connect(cfg)
    con.execute(f"SET threads = {min(cfg.threads, 4)}")
    t0 = time.time()

    # 1. papers in the sample with their authors, field and year (any k; one row per paper-author)
    _step(con, "paper_authors", f"""
        WITH papers AS (
          SELECT DISTINCT s.work_id, s.publication_year, f.field AS set_field
          FROM read_parquet('{sets}') s
          JOIN (SELECT set_id, field FROM read_parquet('{sets}') WHERE is_focal AND k = 1) f USING (set_id)),
        ids AS (SELECT work_id, author_ids FROM read_parquet('{slim_glob}', hive_partitioning = true) w
                SEMI JOIN papers p ON w.work_id = p.work_id)
        SELECT DISTINCT p.work_id, p.publication_year, p.set_field, u.author_id
        FROM papers p JOIN ids i USING (work_id), UNNEST(i.author_ids) AS u(author_id)""",
        tmp / "paper_authors.parquet", t0)

    # 2. every work of every sample author: year and citations
    _step(con, "author_works", f"""
        WITH authors AS (SELECT DISTINCT author_id FROM read_parquet('{tmp / "paper_authors.parquet"}'))
        SELECT DISTINCT u.author_id, w.work_id, w.publication_year, w.cited_by_count
        FROM read_parquet('{slim_glob}', hive_partitioning = true) w, UNNEST(w.author_ids) AS u(author_id)
        SEMI JOIN authors a ON u.author_id = a.author_id
        WHERE w.type = 'article' AND NOT coalesce(w.is_paratext, false) AND w.publication_year IS NOT NULL""",
        tmp / "author_works.parquet", t0)

    # 3. prior-work citations per (paper, author), percentile within field x year
    _step(con, "author_prestige", f"""
        WITH pa AS (SELECT * FROM read_parquet('{tmp / "paper_authors.parquet"}')),
        prior AS (
          SELECT pa.work_id, pa.author_id, pa.set_field, pa.publication_year,
                 coalesce(sum(aw.cited_by_count), 0) AS prior_cites, count(aw.work_id) AS prior_works
          FROM pa LEFT JOIN read_parquet('{tmp / "author_works.parquet"}') aw
            ON aw.author_id = pa.author_id AND aw.publication_year < pa.publication_year
          GROUP BY 1, 2, 3, 4)
        SELECT *, percent_rank() OVER (PARTITION BY set_field, publication_year ORDER BY prior_cites) AS pct
        FROM prior""", tmp / "author_prestige.parquet", t0)

    # 4. paper-level maxima
    _step(con, "paper_prestige", f"""
        WITH phil AS (SELECT author_id FROM read_parquet('{s03_dir / "philosophers.parquet"}')),
        ap AS (SELECT a.*, (p.author_id IS NOT NULL) AS is_phil
               FROM read_parquet('{tmp / "author_prestige.parquet"}') a LEFT JOIN phil p USING (author_id))
        SELECT work_id,
               max(pct)                                         AS prestige_max,
               max(CASE WHEN NOT is_phil THEN pct END)          AS prestige_max_nonphil,
               max(prior_works)                                 AS n_prior_works_max,
               max(prior_cites)                                 AS prior_cites_max
        FROM ap GROUP BY work_id""", out_dir / "paper_prestige.parquet", t0)


def main(dev: bool = typer.Option(False)) -> None:
    cfg = load_config()
    d = "_dev" if dev else ""
    run(cfg, cfg.derived / f"s02{d}" / "works_slim", cfg.derived / f"s03{d}", cfg.derived / f"s05{d}",
        cfg.derived / f"s08{d}")
