"""Stage s03: who is a philosopher?

Candidates are authors with at least one article in a philosophy source (s01). The full data is kept, so
the threshold can be decided later without touching the slim table again:
  candidate_author_works/  every (author, article) of every candidate, one file per publication year
  author_shares.parquet    per candidate: articles in and out of philosophy, share, years, citations
  philosophers.parquet     the current rule from config (min_works, min_share), a filter on the table above
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import typer

from vop.config import Config, connect, load_config

KEEP = ("w.type = 'article' AND NOT coalesce(w.is_paratext, false) AND NOT coalesce(w.is_retracted, false) "
        "AND len(w.author_ids) > 0")


def run(cfg: Config, slim_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    caw = out_dir / "candidate_author_works"
    caw.mkdir(exist_ok=True)
    p = cfg.params
    y0 = int(p.get("author_year_min", 1950))
    phil = cfg.derived / "s01" / "philosophy_sources.parquet"
    glob = str(slim_dir / "*" / "*.parquet")
    con = connect(cfg)
    t0 = time.time()
    cand = out_dir / "candidates.parquet"
    con.execute(f"""
        COPY (
          SELECT DISTINCT u.author_id FROM (
            SELECT w.author_ids FROM read_parquet('{glob}', hive_partitioning = true) w
            SEMI JOIN read_parquet('{phil}') s ON w.source_id = s.source_id
            WHERE {KEEP} AND w.publication_year >= {y0}
          ) a, UNNEST(a.author_ids) AS u(author_id)
          WHERE u.author_id IS NOT NULL
        ) TO '{cand}' (FORMAT parquet, COMPRESSION zstd)""")
    n = con.execute(f"SELECT count(*) FROM read_parquet('{cand}')").fetchone()[0]
    typer.echo(f"[{(time.time() - t0) / 60:6.1f} min] candidates: {n:,}")
    sys.stdout.flush()

    for part in sorted(q for q in slim_dir.iterdir() if q.name.startswith("publication_year=")):
        year = part.name.split("=")[1]
        target = caw / f"{part.name}.parquet"
        if not year.isdigit() or int(year) < y0 or target.exists():
            continue
        con.execute(f"""
            COPY (
              WITH w AS (
                SELECT w.work_id, w.source_id, w.cited_by_count, w.author_ids,
                       w.source_id IN (SELECT source_id FROM read_parquet('{phil}')) AS in_phil
                FROM read_parquet('{part}/*.parquet', hive_partitioning = false) w WHERE {KEEP}
              )
              SELECT DISTINCT u.author_id, w.work_id, w.source_id, {int(year)} AS publication_year,
                     w.cited_by_count, coalesce(w.in_phil, false) AS in_phil
              FROM w, UNNEST(w.author_ids) AS u(author_id)
              SEMI JOIN read_parquet('{cand}') c ON u.author_id = c.author_id
            ) TO '{target}' (FORMAT parquet, COMPRESSION zstd)""")
    typer.echo(f"[{(time.time() - t0) / 60:6.1f} min] candidate_author_works written")
    sys.stdout.flush()

    shares = out_dir / "author_shares.parquet"
    con.execute(f"""
        COPY (
          SELECT author_id,
                 count(*)                                                    AS works_total,
                 sum(in_phil::INT)::BIGINT                                   AS works_phil,
                 count(*) - sum(in_phil::INT)::BIGINT                        AS works_out,
                 sum(in_phil::INT) / count(*)                                AS share_phil,
                 min(publication_year)                                       AS first_year,
                 max(publication_year)                                       AS last_year,
                 min(publication_year) FILTER (WHERE in_phil)                AS first_phil_year,
                 max(publication_year) FILTER (WHERE in_phil)                AS last_phil_year,
                 sum(cited_by_count)::BIGINT                                 AS cites_total,
                 coalesce(sum(cited_by_count) FILTER (WHERE in_phil), 0)::BIGINT AS cites_phil
          FROM read_parquet('{caw}/*.parquet') GROUP BY author_id
        ) TO '{shares}' (FORMAT parquet, COMPRESSION zstd)""")
    typer.echo(f"[{(time.time() - t0) / 60:6.1f} min] author_shares written")

    min_works, min_share = int(p["min_works"]), float(p["min_share"])
    out = out_dir / "philosophers.parquet"
    con.execute(f"""COPY (SELECT * FROM read_parquet('{shares}') WHERE works_total >= {min_works} AND share_phil > {min_share})
                    TO '{out}' (FORMAT parquet, COMPRESSION zstd)""")
    typer.echo(con.execute(f"""
        SELECT count(*) AS candidate_authors,
               sum((works_total >= {min_works})::INT) AS with_min_works,
               sum((works_total >= {min_works} AND share_phil > {min_share})::INT) AS philosophers,
               sum((works_total >= {min_works} AND share_phil > 0.75)::INT) AS philosophers_075,
               sum((works_total >= 10 AND share_phil > {min_share})::INT) AS philosophers_10works
        FROM read_parquet('{shares}')""").fetchdf().T.to_string(header=False))


def main(dev: bool = typer.Option(False, help="Run on the s02 dev output")) -> None:
    cfg = load_config()
    slim = cfg.derived / ("s02_dev" if dev else "s02") / "works_slim"
    run(cfg, slim, cfg.derived / ("s03_dev" if dev else "s03"))
