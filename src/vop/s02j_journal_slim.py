"""Stage s02j: journals-only slim table, the starting point of the journal variant (VOP_VARIANT=journal).

Keeps only type = 'article' in sources of type 'journal' from the full slim table and attaches abstract
availability and page counts from s02b. Everything downstream therefore sees journal articles only,
including philosopher status. No raw re-scan is needed.
"""

from __future__ import annotations

import time

import typer

from vop.config import connect, load_config


def main() -> None:
    cfg = load_config()
    if not cfg.variant:
        raise typer.BadParameter("run with VOP_VARIANT set, so the full tables are not overwritten")
    base = cfg.data_root / "derived"
    out = cfg.derived / "s02" / "works_slim"
    out.mkdir(parents=True, exist_ok=True)
    s01 = cfg.derived / "s01"
    if not s01.exists():
        s01.symlink_to(base / "s01")
    con = connect(cfg)
    t0 = time.time()
    parts = sorted(p for p in (base / "s02" / "works_slim").iterdir() if p.name.startswith("publication_year="))
    kept = 0
    for part in parts:
        meta = base / "s02b" / "works_meta" / part.name
        (out / part.name).mkdir(exist_ok=True)
        target = out / part.name / "data_0.parquet"
        if target.exists():
            continue
        con.execute(f"""
            COPY (
              SELECT w.*, coalesce(m.has_abstract, false) AS has_abstract, m.has_fulltext,
                     CASE WHEN try_cast(m.last_page AS BIGINT) >= try_cast(m.first_page AS BIGINT)
                           AND try_cast(m.last_page AS BIGINT) - try_cast(m.first_page AS BIGINT) < 500
                          THEN try_cast(m.last_page AS BIGINT) - try_cast(m.first_page AS BIGINT) + 1 END AS n_pages
              FROM read_parquet('{part}/*.parquet', hive_partitioning = false) w
              LEFT JOIN read_parquet('{meta}/*.parquet', hive_partitioning = false) m USING (work_id)
              WHERE w.type = 'article' AND w.source_type = 'journal'
            ) TO '{target}' (FORMAT parquet, COMPRESSION zstd)""")
    kept = con.execute(f"SELECT count(*) FROM read_parquet('{out}/*/*.parquet')").fetchone()[0]
    typer.echo(f"{kept:,} journal articles in {(time.time() - t0) / 60:.1f} min -> {out}")
