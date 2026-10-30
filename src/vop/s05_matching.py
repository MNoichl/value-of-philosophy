"""Stage s05: matched controls.

For each focal work (s04) and each matching radius k = 1..max_year_diff, sample `controls_per_focal`
control articles from the same source within +-k years and +-k authors, excluding anything with a
philosopher author or that is itself a focal work.

Runs as a chain of materialised steps (each written to parquet) so every step streams and spills
instead of holding the whole pipeline in memory. Writes
  data/derived/s05/match_links.parquet   (focal_id, control_id, k, year_diff, author_diff)
  data/derived/s05/matched_sets.parquet  long table: focal + control rows with slim columns, per (set_id, k)
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import typer

from vop.config import Config, connect, load_config


def _step(con, name: str, sql: str, path: Path, t0: float) -> None:
    con.execute(f"COPY ({sql}) TO '{path}' (FORMAT parquet, COMPRESSION zstd)")
    n = con.execute(f"SELECT count(*) FROM read_parquet('{path}')").fetchone()[0]
    typer.echo(f"[{(time.time() - t0) / 60:6.1f} min] {name}: {n:,} rows", err=False)
    sys.stdout.flush()


def run(cfg: Config, slim_dir: Path, s03_dir: Path, s04_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    p = cfg.params
    max_k = int(p["max_year_diff"])
    assert max_k == int(p["max_author_diff"]), "nested scenarios assume equal year/author radii"
    cap, m, seed = int(p.get("bucket_cap", 50)), int(p["controls_per_focal"]), int(p["seed"])
    slim_glob = str(slim_dir / "*" / "*.parquet")
    focal = s04_dir / "focal_works.parquet"
    philosophers = s03_dir / "philosophers.parquet"

    con = connect(cfg)
    con.execute(f"SET threads = {min(cfg.threads, 4)}")
    con.execute(f"SET enable_progress_bar = {str(sys.stdout.isatty()).lower()}")
    t0 = time.time()

    # 1. focal works, slim
    _step(con, "focal", f"""
        SELECT work_id, source_id, publication_year, coalesce(authors_count, n_authorships) AS n_authors
        FROM read_parquet('{focal}')""", tmp / "focal.parquet", t0)

    # 2. candidate pool: articles in focal sources (streams over the slim table once)
    _step(con, "pool", f"""
        SELECT w.work_id, w.source_id, w.publication_year,
               coalesce(w.authors_count, w.n_authorships) AS n_authors, w.author_ids
        FROM read_parquet('{slim_glob}', hive_partitioning = true) w
        SEMI JOIN (SELECT DISTINCT source_id FROM read_parquet('{tmp / "focal.parquet"}')) f
          ON w.source_id = f.source_id
        WHERE w.type = 'article' AND NOT coalesce(w.is_paratext, false) AND NOT coalesce(w.is_retracted, false)
          AND len(w.author_ids) > 0
          AND w.publication_year BETWEEN {int(p["year_min"]) - max_k} AND {int(p["year_max"]) + max_k}""",
        tmp / "pool.parquet", t0)

    # 3. pool works with any philosopher author
    _step(con, "pool_with_phil", f"""
        SELECT DISTINCT pw.work_id
        FROM read_parquet('{tmp / "pool.parquet"}') pw, UNNEST(pw.author_ids) AS u(author_id)
        JOIN read_parquet('{philosophers}') ph ON u.author_id = ph.author_id""",
        tmp / "pool_with_phil.parquet", t0)

    # 4. eligible controls (no author_ids column from here on)
    _step(con, "eligible", f"""
        SELECT pw.work_id, pw.source_id, pw.publication_year, pw.n_authors
        FROM read_parquet('{tmp / "pool.parquet"}') pw
        ANTI JOIN read_parquet('{tmp / "pool_with_phil.parquet"}') x ON pw.work_id = x.work_id
        ANTI JOIN read_parquet('{tmp / "focal.parquet"}') f ON pw.work_id = f.work_id""",
        tmp / "eligible.parquet", t0)

    # 5. random cap per (source, year, n_authors) bucket so the pair join is bounded
    _step(con, "capped", f"""
        SELECT work_id, source_id, publication_year, n_authors FROM (
          SELECT *, row_number() OVER (PARTITION BY source_id, publication_year, n_authors
                                       ORDER BY hash(work_id, {seed})) AS r
          FROM read_parquet('{tmp / "eligible.parquet"}')
        ) WHERE r <= {cap}""", tmp / "capped.parquet", t0)

    # 6. pairs within the widest radius
    _step(con, "pairs", f"""
        SELECT f.work_id AS focal_id, c.work_id AS control_id,
               abs(f.publication_year - c.publication_year) AS year_diff,
               abs(f.n_authors - c.n_authors) AS author_diff,
               hash(f.work_id, c.work_id, {seed}) AS h
        FROM read_parquet('{tmp / "focal.parquet"}') f
        JOIN read_parquet('{tmp / "capped.parquet"}') c
          ON c.source_id = f.source_id
         AND c.publication_year BETWEEN f.publication_year - {max_k} AND f.publication_year + {max_k}
         AND c.n_authors BETWEEN f.n_authors - {max_k} AND f.n_authors + {max_k}""",
        tmp / "pairs.parquet", t0)

    # 7. sample m controls per (focal, k) for each nested radius k
    links = out_dir / "match_links.parquet"
    _step(con, "match_links", f"""
        SELECT focal_id, control_id, k, year_diff, author_diff FROM (
          SELECT p.*, k.k,
                 row_number() OVER (PARTITION BY p.focal_id, k.k ORDER BY p.h) AS rnk
          FROM read_parquet('{tmp / "pairs.parquet"}') p
          JOIN range(1, {max_k} + 1) AS k(k) ON greatest(p.year_diff, p.author_diff) <= k.k
        ) WHERE rnk <= {m}""", links, t0)

    # 8. long table of focal + control rows with the slim columns
    cols = ("work_id, publication_year, source_id, issn_l, coalesce(authors_count, n_authorships) AS n_authors, "
            "cited_by_count, counts_by_year, referenced_works_count, fwci, cnp, subfield, field, domain")
    sets = out_dir / "matched_sets.parquet"
    _step(con, "matched_sets", f"""
        WITH ids AS (SELECT DISTINCT control_id AS work_id FROM read_parquet('{links}')
                     UNION SELECT DISTINCT focal_id FROM read_parquet('{links}')),
        slim AS (SELECT {cols} FROM read_parquet('{slim_glob}', hive_partitioning = true) w
                 SEMI JOIN ids ON w.work_id = ids.work_id),
        sets AS (SELECT focal_id AS set_id, k FROM read_parquet('{links}') GROUP BY 1, 2),
        controls AS (
          SELECT l.focal_id AS set_id, l.k, false AS is_focal, l.year_diff, l.author_diff, s.*
          FROM read_parquet('{links}') l JOIN slim s ON s.work_id = l.control_id),
        focals AS (
          SELECT st.set_id, st.k, true AS is_focal, 0 AS year_diff, 0 AS author_diff, s.*
          FROM sets st JOIN slim s ON s.work_id = st.set_id)
        SELECT * FROM controls UNION ALL SELECT * FROM focals""", sets, t0)

    typer.echo(con.execute(f"""
        SELECT k, count(DISTINCT set_id) AS sets, sum((NOT is_focal)::int) AS controls,
               round(avg(CASE WHEN NOT is_focal THEN cited_by_count END), 1) AS ctrl_mean_cites,
               round(avg(CASE WHEN is_focal THEN cited_by_count END), 1) AS focal_mean_cites,
               median(CASE WHEN NOT is_focal THEN cited_by_count END) AS ctrl_med,
               median(CASE WHEN is_focal THEN cited_by_count END) AS focal_med
        FROM read_parquet('{sets}') GROUP BY k ORDER BY k
    """).fetchdf().to_string(index=False))
    typer.echo(con.execute(f"""
        SELECT k, n_controls, count(*) AS sets FROM
          (SELECT focal_id, k, count(*) n_controls FROM read_parquet('{links}') GROUP BY 1,2)
        GROUP BY 1,2 ORDER BY 1,2
    """).fetchdf().to_string(index=False))


def main(dev: bool = typer.Option(False, help="Run on the dev outputs")) -> None:
    cfg = load_config()
    d = "_dev" if dev else ""
    run(cfg, cfg.derived / f"s02{d}" / "works_slim", cfg.derived / f"s03{d}", cfg.derived / f"s04{d}",
        cfg.derived / f"s05{d}")
