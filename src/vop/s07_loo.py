"""Leave-one-out comparison of saved hurdle fits, chunked so it fits in memory.

PSIS-LOO treats each observation independently, so the log-likelihood matrix (chains x draws x papers)
can be read from the NetCDF file a block of papers at a time. Peak memory is one block, not whole models.
Comparison uses the usual paired difference: elpd_diff = sum(d_i), se = sqrt(n * var(d_i)).
"""

from __future__ import annotations

from pathlib import Path

import arviz as az
import h5py
import numpy as np
import pandas as pd
import typer
from scipy.special import logsumexp

from vop.config import load_config


def _estimate_reff(ds, n_probe: int = 1000, seed: int = 42) -> float:
    """Relative efficiency: mean bulk ESS of a random sample of pointwise log-likelihoods, over total draws."""
    c, dr, n = ds.shape
    idx = np.sort(np.random.default_rng(seed).choice(n, size=min(n_probe, n), replace=False))
    arr = ds[:, :, idx]
    ess = az.ess(az.from_dict({"log_likelihood": {"y": arr}}), group="log_likelihood", var_names=["y"])["y"].values
    return float(np.clip(np.nanmean(ess) / (c * dr), 0.01, 1.0))


def pointwise(path: Path, chunk: int = 20000) -> tuple[pd.DataFrame, np.ndarray]:
    with h5py.File(path, "r") as h:
        ds = h["log_likelihood"]["y"]
        y = h["observed_data"]["y"][()]
        c, dr, n = ds.shape
        reff = _estimate_reff(ds)
        out = []
        for i in range(0, n, chunk):
            arr = ds[:, :, i:i + chunk]
            res = az.loo(az.from_dict({"log_likelihood": {"y": arr}}), pointwise=True, var_name="y", reff=reff)
            lpd = logsumexp(arr.reshape(c * dr, arr.shape[2]), axis=0) - np.log(c * dr)
            out.append(pd.DataFrame({"elpd_i": np.asarray(res.elpd_i).ravel(), "k": np.asarray(res.pareto_k).ravel(), "lpd_i": lpd}))
            typer.echo(f"  {path.name}: {min(i + chunk, n):,}/{n:,}")
    return pd.concat(out, ignore_index=True).assign(reff=reff), y


def compare(fits: dict[str, Path], out_csv: Path, chunk: int) -> pd.DataFrame:
    pts, ys = {}, {}
    for label, path in fits.items():
        pts[label], ys[label] = pointwise(path, chunk)
        pts[label].to_parquet(out_csv.parent / f"loo_pointwise_{path.stem}.parquet", index=False)
    first = next(iter(ys))
    aligned = all(np.array_equal(ys[first], ys[k], equal_nan=True) for k in ys)
    rows = []
    for k, p in pts.items():
        rows.append(dict(model=k, elpd=p.elpd_i.sum(), se=np.sqrt(len(p) * p.elpd_i.var()), p_loo=(p.lpd_i - p.elpd_i).sum(),
                         k_gt_0_7=int((p.k > 0.7).sum()), k_gt_1=int((p.k > 1).sum()), n=len(p), reff=p.reff.iloc[0]))
    t = pd.DataFrame(rows).sort_values("elpd", ascending=False).reset_index(drop=True)
    best = pts[t.model.iloc[0]].elpd_i.to_numpy()
    t["elpd_diff"] = t.elpd - t.elpd.iloc[0]
    if aligned:
        t["dse"] = [np.sqrt(len(best) * np.var(pts[m].elpd_i.to_numpy() - best, ddof=1)) for m in t.model]
    else:
        t["dse"] = np.nan
        typer.echo("WARNING: observed y differs in order between fits; paired SE not computed")
    t.to_csv(out_csv, index=False)
    typer.echo(t.round(2).to_string(index=False))
    return t


def main(
    pattern: list[str] = typer.Argument(..., help="label=filename pairs inside data/derived/s07"),
    out: str = typer.Option("loo_compare_full.csv"), chunk: int = typer.Option(20000),
) -> None:
    d = load_config().derived / "s07"
    fits = {x.split("=")[0]: d / x.split("=")[1] for x in pattern}
    compare(fits, d / out, chunk)
