from pathlib import Path

from vop.config import REPO_ROOT, load_config


def test_example_config_parses():
    cfg = load_config(REPO_ROOT / "config.example.toml")
    assert cfg.snapshot_root == Path("/data/openalex_project_storage/vop/raw/openalex/data/parquet")
    assert cfg.params["min_share"] == 0.5
