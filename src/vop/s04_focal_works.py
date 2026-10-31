"""Stage s04: focal works.

Articles by at least one philosopher (s03) in a source NOT on the philosophy list (s01), within the year
window, excluding philosophy-flavoured subfields, retractions and paratext. Built one publication year at a
time from s03's candidate table, so changing the threshold means rerunning from here only.
Output: data/derived/s04/focal_works.parquet with the slim columns plus philosopher-author flags.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import typer

from vop.config import Config, connect, load_config

DEFAULT_EXCLUDED_SUBFIELDS = ["Philosophy", "History and Philosophy of Science"]


def run(cfg: Config, slim_dir: Path, s03_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = out_dir / "parts"
    if parts_dir.exists():
        shutil.rmtree(parts_dir)
    parts_dir.mkdir()
    p = cfg.params
    excluded = ", ".join("'" + s.replace("'", "''") + "'" for s in p.get("excluded_subfields", DEFAULT_EXCLUDED_SUBFIELDS))
    phil = s03_dir / "philosophers.parquet"
    con = connect(cfg)
    t0 = time.time()
    for y in range(int(p["year_min"]), int(p["year_max"]) + 1):
        part, caw = slim_dir / f"publication_year={y}", s03_dir / "candidate_author_works" / f"publication_year={y}.parquet"
        if not part.exists() or not caw.exists():
            continue
        con.execute(f"""
            COPY (
              WITH pw AS (
                SELECT c.work_id, list(c.author_id) AS phil_author_ids
                FROM read_parquet('{caw}') c SEMI JOIN read_parquet('{phil}') h ON c.author_id = h.author_id
                WHERE NOT c.in_phil GROUP BY c.work_id
              )
              SELECT a.*, {y}::BIGINT AS publication_year, f.phil_author_ids,
                     len(f.phil_author_ids) AS n_philosophers,
                     list_contains(f.phil_author_ids, a.author_ids[1]) AS first_author_is_phil,
                     list_contains(f.phil_author_ids, a.author_ids[-1]) AS last_author_is_phil
              FROM read_parquet('{part}/*.parquet', hive_partitioning = false) a JOIN pw f USING (work_id)
              WHERE a.type = 'article' AND NOT coalesce(a.is_paratext, false) AND NOT coalesce(a.is_retracted, false)
                AND a.source_id IS NOT NULL AND coalesce(a.subfield, '') NOT IN ({excluded})
            ) TO '{parts_dir / f"{y}.parquet"}' (FORMAT parquet, COMPRESSION zstd)""")
    out = out_dir / "focal_works.parquet"
    con.execute(f"COPY (SELECT * FROM read_parquet('{parts_dir}/*.parquet')) TO '{out}' (FORMAT parquet, COMPRESSION zstd)")
    shutil.rmtree(parts_dir)
    typer.echo(f"focal works written in {(time.time() - t0) / 60:.1f} min -> {out}")
    typer.echo(con.execute(f"""
        SELECT count(*) AS focal_works, count(DISTINCT source_id) AS sources,
               sum(first_author_is_phil::int) AS first_author_phil,
               round(avg(n_philosophers), 2) AS mean_n_phil,
               round(avg(cited_by_count), 1) AS mean_cites, median(cited_by_count) AS median_cites
        FROM read_parquet('{out}')""").fetchdf().T.to_string(header=False))
    typer.echo(con.execute(f"SELECT source_type, count(*) n FROM read_parquet('{out}') GROUP BY 1 ORDER BY 2 DESC").fetchdf().to_string(index=False))
    typer.echo(con.execute(f"SELECT subfield, count(*) n FROM read_parquet('{out}') GROUP BY 1 ORDER BY 2 DESC LIMIT 25").fetchdf().to_string(index=False))


def main(dev: bool = typer.Option(False, help="Run on the dev outputs")) -> None:
    cfg = load_config()
    sfx = "_dev" if dev else ""
    run(cfg, cfg.derived / f"s02{sfx}" / "works_slim", cfg.derived / f"s03{sfx}", cfg.derived / f"s04{sfx}")
