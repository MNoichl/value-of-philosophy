"""Command line entry point: one sub-command per pipeline stage."""

from __future__ import annotations

import typer

app = typer.Typer(help="Value of philosophy pipeline. Stages are added as they are written.", no_args_is_help=True)


@app.callback()
def main() -> None:
    """Value of philosophy pipeline."""


from vop import s01_sources, s02_works_slim, s03_philosophers  # noqa: E402

app.command("s01-sources", help="Match PhilPapers journals to OpenAlex sources.")(s01_sources.main)
app.command("s02-works-slim", help="One full scan of works -> slim parquet table.")(s02_works_slim.main)
app.command("s03-philosophers", help="Per-author philosophy share; select philosophers.")(s03_philosophers.main)


@app.command()
def info() -> None:
    """Print the resolved configuration."""
    from vop.config import load_config

    cfg = load_config()
    for k, v in cfg.__dict__.items():
        typer.echo(f"{k:20} {v}")


if __name__ == "__main__":
    app()
