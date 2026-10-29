"""Stage s03: who is a philosopher?

For every author with at least one article in a philosophy source (s01), count their articles in
total and in philosophy sources. Output:
  data/derived/s03/author_shares.parquet   all candidate authors with counts and share (for sensitivity)
  data/derived/s03/philosophers.parquet    authors with works_total >= min_works and share_phil > min_share

This fixes the old notebook's bug where only the first author of each philosophy paper was counted.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import typer

from vop.config import Config, connect, load_config

SQL_PATH = Path(__file__).parent / "sql" / "s03_philosophers.sql"


def run(cfg: Config, slim_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = cfg.params
    sql = SQL_PATH.read_text().format(
        slim_glob=str(slim_dir / "*" / "*.parquet"),
        phil_sources=str(cfg.derived / "s01" / "philosophy_sources.parquet"),
        year_min=int(p.get("author_year_min", 1950)),
    )
    con = connect(cfg)
    con.execute(f"SET enable_progress_bar = {str(sys.stdout.isatty()).lower()}")
    t0 = time.time()
    shares = out_dir / "author_shares.parquet"
    con.execute(f"COPY ({sql}) TO '{shares}' (FORMAT parquet, COMPRESSION zstd)")
    typer.echo(f"author shares computed in {(time.time() - t0) / 60:.1f} min -> {shares}")

    min_works, min_share = int(p["min_works"]), float(p["min_share"])
    phil = out_dir / "philosophers.parquet"
    con.execute(
        f"""
        COPY (SELECT * FROM read_parquet('{shares}')
              WHERE works_total >= {min_works} AND share_phil > {min_share})
        TO '{phil}' (FORMAT parquet, COMPRESSION zstd)
        """
    )
    typer.echo(con.execute(f"""
        SELECT count(*) AS candidate_authors,
               sum(CASE WHEN works_total >= {min_works} THEN 1 ELSE 0 END) AS with_min_works,
               sum(CASE WHEN works_total >= {min_works} AND share_phil > {min_share} THEN 1 ELSE 0 END) AS philosophers,
               sum(CASE WHEN works_total >= {min_works} AND share_phil > 0.75 THEN 1 ELSE 0 END) AS philosophers_075,
               sum(CASE WHEN works_total >= 10 AND share_phil > {min_share} THEN 1 ELSE 0 END) AS philosophers_10works
        FROM read_parquet('{shares}')
    """).fetchdf().T.to_string(header=False))
    typer.echo("share_phil distribution among authors with min_works:")
    typer.echo(con.execute(f"""
        SELECT round(share_phil, 1) AS share_bin, count(*) AS n
        FROM read_parquet('{shares}') WHERE works_total >= {min_works}
        GROUP BY 1 ORDER BY 1
    """).fetchdf().to_string(index=False))


def main(dev: bool = typer.Option(False, help="Run on the s02 dev output")) -> None:
    cfg = load_config()
    slim = cfg.derived / ("s02_dev" if dev else "s02") / "works_slim"
    run(cfg, slim, cfg.derived / ("s03_dev" if dev else "s03"))
