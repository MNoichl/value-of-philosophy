"""Stage s02b: document metadata for all OpenAlex articles.

One light pass over the raw works, without author lists, so it is far faster than s02. Writes
data/derived/s02b/works_meta/ partitioned by publication_year: work_id, title, has_abstract,
has_fulltext, first_page, last_page, language. It joins to the slim table on work_id and is the input
for defining a stricter research-article filter. Titles are kept so the rule can change without a re-scan.
"""

from __future__ import annotations

import shutil
import time

import typer

from vop.config import connect, load_config


def main(
    threads: int = typer.Option(3, help="low by default so it can run beside other jobs"),
    memory: str = typer.Option("8GB"),
) -> None:
    cfg = load_config()
    out = cfg.derived / "s02b" / "works_meta"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    con = connect(cfg)
    con.execute(f"SET threads = {threads}")
    con.execute(f"SET memory_limit = '{memory}'")
    glob = cfg.snapshot_root / "works" / "updated_date=*" / "part_*.parquet"
    t0 = time.time()
    con.execute(f"""
        COPY (
          SELECT id AS work_id, publication_year, title,
                 abstract_inverted_index IS NOT NULL AS has_abstract, has_fulltext,
                 biblio.first_page AS first_page, biblio.last_page AS last_page, language
          FROM read_parquet('{glob}', hive_partitioning = false)
          WHERE type = 'article'
        ) TO '{out}' (FORMAT parquet, PARTITION_BY (publication_year), COMPRESSION zstd, ROW_GROUP_SIZE 200000)
    """)
    typer.echo(f"scan finished in {(time.time() - t0) / 60:.1f} min")
    n = con.execute(f"SELECT count(*) FROM read_parquet('{out}/*/*.parquet', hive_partitioning = true)").fetchone()[0]
    typer.echo(f"articles: {n:,}")
