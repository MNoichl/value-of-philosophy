"""Stage s05: matched controls.

For each focal work (s04) and each matching radius k = 1..max_year_diff, sample `controls_per_focal`
control articles from the same source within +-k years and +-k authors, excluding anything with a
philosopher author or that is itself a focal work. Writes
  data/derived/s05/match_links.parquet   (focal_id, control_id, k, year_diff, author_diff)
  data/derived/s05/matched_sets.parquet  long table: focal + control rows with the slim columns,
                                          one block per (set_id = focal_id, k), ready for s06
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import typer

from vop.config import Config, connect, load_config

SQL_PATH = Path(__file__).parent / "sql" / "s05_matching.sql"


def run(cfg: Config, slim_dir: Path, s03_dir: Path, s04_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = cfg.params
    max_k = int(p["max_year_diff"])
    assert max_k == int(p["max_author_diff"]), "nested scenarios assume equal year/author radii"
    slim_glob = str(slim_dir / "*" / "*.parquet")
    focal = str(s04_dir / "focal_works.parquet")
    sql = SQL_PATH.read_text().format(
        slim_glob=slim_glob, focal=focal, philosophers=str(s03_dir / "philosophers.parquet"),
        max_k=max_k, bucket_cap=int(p.get("bucket_cap", 200)),
        controls_per_focal=int(p["controls_per_focal"]), seed=int(p["seed"]),
    )
    con = connect(cfg)
    con.execute(f"SET enable_progress_bar = {str(sys.stdout.isatty()).lower()}")
    t0 = time.time()
    links = out_dir / "match_links.parquet"
    con.execute(f"COPY ({sql}) TO '{links}' (FORMAT parquet, COMPRESSION zstd)")
    typer.echo(f"match links in {(time.time() - t0) / 60:.1f} min -> {links}")

    cols = ("work_id, publication_year, source_id, issn_l, coalesce(authors_count, n_authorships) AS n_authors, "
            "cited_by_count, counts_by_year, referenced_works_count, fwci, cnp, subfield, field, domain")
    sets = out_dir / "matched_sets.parquet"
    con.execute(f"""
        COPY (
          WITH slim AS (SELECT {cols} FROM read_parquet('{slim_glob}', hive_partitioning = true)),
          sets AS (SELECT focal_id AS set_id, k, count(*) AS n_controls
                   FROM read_parquet('{links}') GROUP BY 1, 2),
          controls AS (
            SELECT l.focal_id AS set_id, l.k, false AS is_focal, l.year_diff, l.author_diff, s.*
            FROM read_parquet('{links}') l JOIN slim s ON s.work_id = l.control_id),
          focals AS (
            SELECT st.set_id, st.k, true AS is_focal, 0 AS year_diff, 0 AS author_diff, s.*
            FROM sets st JOIN slim s ON s.work_id = st.set_id)
          SELECT * FROM controls UNION ALL SELECT * FROM focals
        ) TO '{sets}' (FORMAT parquet, COMPRESSION zstd)
    """)
    typer.echo(f"matched sets -> {sets}")
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
