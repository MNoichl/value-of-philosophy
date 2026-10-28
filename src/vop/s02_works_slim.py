"""Stage s02: slim works table.

The single full scan of the OpenAlex works parquet. Writes data/derived/s02/works_slim/ as parquet
partitioned by publication_year, with one row per work: ids, year, type, source, author ids,
citation counts (all-time and per year) and the primary topic hierarchy. Deleted works are removed
via works/deleted_ids.csv.gz. Everything downstream reads this table instead of the raw snapshot.
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import typer

from vop.config import Config, connect, load_config

SQL_PATH = Path(__file__).parent / "sql" / "s02_works_slim.sql"


def run(cfg: Config, snapshot_root: Path, out_dir: Path) -> None:
    works_glob = str(snapshot_root / "works" / "updated_date=*" / "part_*.parquet")
    deleted = snapshot_root / "works" / "deleted_ids.csv.gz"
    if not deleted.exists():  # dev sample: fall back to the full snapshot's file
        deleted = cfg.snapshot_root / "works" / "deleted_ids.csv.gz"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    sql = SQL_PATH.read_text().format(works_glob=works_glob, deleted_ids=deleted)
    con = connect(cfg)
    con.execute(f"SET enable_progress_bar = {str(sys.stdout.isatty()).lower()}")
    t0 = time.time()
    con.execute(
        f"COPY ({sql}) TO '{out_dir}' "
        "(FORMAT parquet, PARTITION_BY (publication_year), COMPRESSION zstd, ROW_GROUP_SIZE 200000)"
    )
    typer.echo(f"scan finished in {(time.time() - t0) / 60:.1f} min")

    stats = con.execute(
        f"""
        SELECT count(*) AS rows, count(DISTINCT work_id) AS distinct_ids,
               sum(CASE WHEN type = 'article' THEN 1 ELSE 0 END) AS articles,
               min(publication_year) AS y_min, max(publication_year) AS y_max
        FROM read_parquet('{out_dir}/*/*.parquet', hive_partitioning = true)
        """
    ).fetchone()
    typer.echo(f"rows={stats[0]:,} distinct={stats[1]:,} articles={stats[2]:,} years={stats[3]}..{stats[4]}")
    if stats[0] != stats[1]:
        typer.echo(f"WARNING: {stats[0] - stats[1]:,} duplicate work_ids; s03 must dedupe on updated_date")


def main(dev: bool = typer.Option(False, help="Run on the dev sample instead of the full snapshot")) -> None:
    cfg = load_config()
    root = cfg.dev_snapshot_root if dev else cfg.snapshot_root
    out = cfg.derived / ("s02_dev" if dev else "s02") / "works_slim"
    run(cfg, root, out)
