"""Command line entry point: one sub-command per pipeline stage."""

from __future__ import annotations

import typer

app = typer.Typer(help="Value of philosophy pipeline. Stages are added as they are written.")


@app.command()
def info() -> None:
    """Print the resolved configuration."""
    from vop.config import load_config

    cfg = load_config()
    for k, v in cfg.__dict__.items():
        typer.echo(f"{k:20} {v}")


if __name__ == "__main__":
    app()
