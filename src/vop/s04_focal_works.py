"""Stage s04: focal works.

Articles by at least one philosopher (s03) that appeared in a source NOT on the philosophy list (s01),
within the year window, excluding philosophy-flavoured subfields, retractions and paratext.
Output: data/derived/s04/focal_works.parquet with the slim columns plus philosopher-author flags.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import typer

from vop.config import Config, connect, load_config

SQL_PATH = Path(__file__).parent / "sql" / "s04_focal_works.sql"
DEFAULT_EXCLUDED_SUBFIELDS = ["Philosophy", "History and Philosophy of Science"]


def run(cfg: Config, slim_dir: Path, s03_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = cfg.params
    excluded = p.get("excluded_subfields", DEFAULT_EXCLUDED_SUBFIELDS)
    sql = SQL_PATH.read_text().format(
        slim_glob=str(slim_dir / "*" / "*.parquet"),
        phil_sources=str(cfg.derived / "s01" / "philosophy_sources.parquet"),
        philosophers=str(s03_dir / "philosophers.parquet"),
        year_min=int(p["year_min"]),
        year_max=int(p["year_max"]),
        excluded_subfields=", ".join("'" + s.replace("'", "''") + "'" for s in excluded),
    )
    con = connect(cfg)
    con.execute(f"SET enable_progress_bar = {str(sys.stdout.isatty()).lower()}")
    t0 = time.time()
    out = out_dir / "focal_works.parquet"
    con.execute(f"COPY ({sql}) TO '{out}' (FORMAT parquet, COMPRESSION zstd)")
    typer.echo(f"focal works written in {(time.time() - t0) / 60:.1f} min -> {out}")
    typer.echo(con.execute(f"""
        SELECT count(*) AS focal_works, count(DISTINCT source_id) AS sources,
               sum(first_author_is_phil::int) AS first_author_phil,
               round(avg(n_philosophers), 2) AS mean_n_phil,
               round(avg(cited_by_count), 1) AS mean_cites, median(cited_by_count) AS median_cites
        FROM read_parquet('{out}')
    """).fetchdf().T.to_string(header=False))
    typer.echo("top subfields:")
    typer.echo(con.execute(f"""
        SELECT subfield, count(*) n FROM read_parquet('{out}') GROUP BY 1 ORDER BY 2 DESC LIMIT 25
    """).fetchdf().to_string(index=False))


def main(dev: bool = typer.Option(False, help="Run on the dev outputs")) -> None:
    cfg = load_config()
    sfx = "_dev" if dev else ""
    run(cfg, cfg.derived / f"s02{sfx}" / "works_slim", cfg.derived / f"s03{sfx}", cfg.derived / f"s04{sfx}")
