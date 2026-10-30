"""Command line entry point: one sub-command per pipeline stage."""

from __future__ import annotations

import typer

app = typer.Typer(help="Value of philosophy pipeline. Stages are added as they are written.", no_args_is_help=True)


@app.callback()
def main() -> None:
    """Value of philosophy pipeline."""


from vop import s01_sources, s02_works_slim, s03_philosophers, s04_focal_works, s05_matching, s06_analysis, s07_hurdle, s08_prestige, s07_bayes_plot, s07_loo, s02b_works_meta  # noqa: E402

app.command("s01-sources", help="Match PhilPapers journals to OpenAlex sources.")(s01_sources.main)
app.command("s02-works-slim", help="One full scan of works -> slim parquet table.")(s02_works_slim.main)
app.command("s02b-works-meta", help="Abstract, pages and title for all articles (light scan).")(s02b_works_meta.main)
app.command("s03-philosophers", help="Per-author philosophy share; select philosophers.")(s03_philosophers.main)
app.command("s04-focal-works", help="Philosophers' articles in non-philosophy sources.")(s04_focal_works.main)
app.command("s05-matching", help="Sample matched controls per focal work.")(s05_matching.main)
app.command("s06-analysis", help="Within-set models, descriptives, figures.")(s06_analysis.main)
app.command("s07-hurdle", help="Hurdle log-normal citation models (FE and Bayesian).")(s07_hurdle.main)
app.command("s08-prestige", help="Coauthor prestige at publication time (prior-work citation percentile).")(s08_prestige.main)
app.command("s07-bayes-plot", help="Posterior plots from saved Bayesian fits.")(s07_bayes_plot.main)
app.command("s07-loo", help="Chunked PSIS-LOO comparison of saved fits.")(s07_loo.main)


@app.command()
def info() -> None:
    """Print the resolved configuration."""
    from vop.config import load_config

    cfg = load_config()
    for k, v in cfg.__dict__.items():
        typer.echo(f"{k:20} {v}")


if __name__ == "__main__":
    app()
