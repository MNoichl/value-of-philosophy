"""Project configuration: paths, parameters and a configured DuckDB connection."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config.toml"


@dataclass(frozen=True)
class Config:
    data_root: Path
    snapshot_root: Path
    dev_snapshot_root: Path
    threads: int
    memory_limit: str
    temp_directory: Path
    params: dict

    @property
    def derived(self) -> Path:
        return self.data_root / "derived"


def load_config(path: Path = CONFIG_PATH) -> Config:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Copy config.example.toml to config.toml and edit it.")
    raw = tomllib.loads(path.read_text())
    p, d = raw["paths"], raw["duckdb"]
    return Config(
        data_root=Path(p["data_root"]),
        snapshot_root=Path(p["snapshot_root"]),
        dev_snapshot_root=Path(p["dev_snapshot_root"]),
        threads=int(d["threads"]),
        memory_limit=str(d["memory_limit"]),
        temp_directory=Path(d["temp_directory"]),
        params=raw.get("params", {}),
    )


def connect(cfg: Config, database: str | Path = ":memory:") -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(database))
    cfg.temp_directory.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET threads = {cfg.threads}")
    con.execute(f"SET memory_limit = '{cfg.memory_limit}'")
    con.execute(f"SET temp_directory = '{cfg.temp_directory}'")
    con.execute("SET preserve_insertion_order = false")
    return con
