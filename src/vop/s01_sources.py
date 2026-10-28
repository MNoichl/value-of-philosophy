"""Stage s01: philosophy sources.

Input : inputs/philpapers_journals.csv  (columns: name, issn, [n_articles, url]) produced by
        vop.philpapers from the saved PhilPapers journal archive HTML.
Output: data/derived/s01/philosophy_sources.parquet  one row per OpenAlex source matched
        data/derived/s01/philpapers_match_report.csv  one row per PhilPapers entry with match status

Matching is done against the OpenAlex `sources` table on issn_l and the issn array, after ISSN
normalisation. Entries whose ISSN fails the checksum, or that match nothing, are reported so they
can be fixed by hand (PhilPapers typos) rather than silently dropped.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from vop.config import Config, connect, load_config
from vop.issn import issn_checksum_ok, normalize_issn

app = typer.Typer()


def run(cfg: Config, snapshot_root: Path, journals_csv: Path) -> pd.DataFrame:
    out_dir = cfg.derived / "s01"
    out_dir.mkdir(parents=True, exist_ok=True)

    pp = pd.read_csv(journals_csv, dtype=str, keep_default_na=False)
    pp["issn8"] = pp["issn"].map(normalize_issn)
    pp["checksum_ok"] = pp["issn8"].map(lambda s: isinstance(s, str) and issn_checksum_ok(s))

    con = connect(cfg)
    con.register("pp", pp[["name", "issn8"]].dropna())
    sources_glob = str(snapshot_root / "sources" / "*" / "*.parquet")
    con.execute(
        f"""
        CREATE TEMP TABLE src AS
        SELECT id AS source_id, display_name, issn_l, issn, type, works_count,
               replace(issn_l, '-', '') AS issn_l8,
               list_transform(coalesce(issn, []), lambda x: replace(x, '-', '')) AS issn8_list
        FROM read_parquet('{sources_glob}')
        """
    )
    matched = con.execute(
        """
        SELECT DISTINCT pp.name AS philpapers_name, pp.issn8,
               s.source_id, s.display_name, s.issn_l, s.type, s.works_count,
               (s.issn_l8 = pp.issn8) AS matched_on_issn_l
        FROM pp
        JOIN src s ON s.issn_l8 = pp.issn8 OR list_contains(s.issn8_list, pp.issn8)
        """
    ).fetchdf()

    report = pp.merge(
        matched.groupby("issn8").agg(n_sources=("source_id", "nunique"),
                                     openalex_names=("display_name", lambda x: " | ".join(sorted(set(x))))),
        on="issn8", how="left",
    )
    report["n_sources"] = report["n_sources"].fillna(0).astype(int)
    report["status"] = "ok"
    report.loc[report["n_sources"] == 0, "status"] = "no_openalex_match"
    report.loc[~report["checksum_ok"], "status"] = "bad_checksum"
    report.loc[report["issn8"].isna() | (report["issn8"] == ""), "status"] = "no_issn"

    sources = matched.drop_duplicates("source_id").sort_values("works_count", ascending=False)
    sources.to_parquet(out_dir / "philosophy_sources.parquet", index=False)
    report.to_csv(out_dir / "philpapers_match_report.csv", index=False)

    typer.echo(report["status"].value_counts().to_string())
    typer.echo(f"{len(sources)} OpenAlex sources -> {out_dir / 'philosophy_sources.parquet'}")
    return report


@app.command()
def main(
    journals_csv: Path = typer.Option(Path("inputs/philpapers_journals.csv")),
    dev: bool = typer.Option(False, help="Use the dev snapshot root from config.toml"),
) -> None:
    cfg = load_config()
    run(cfg, cfg.dev_snapshot_root if dev else cfg.snapshot_root, journals_csv)


if __name__ == "__main__":
    app()
