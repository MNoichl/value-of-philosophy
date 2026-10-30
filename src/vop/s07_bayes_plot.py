"""Posterior plots for the Bayesian hurdle log-normal fits (s07).

Reads only the population-level variables from the saved InferenceData files (the per-set random
effects and log-likelihood groups stay on disk), so it is cheap to run while another model samples.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import typer
import xarray as xr

from vop.config import REPO_ROOT, load_config

VARS = ["a_h", "b_h", "a_m", "b_m", "a_s", "sd_set_h", "sd_set_m", "sd_set_s"]


def load_posterior(path: Path) -> xr.Dataset:
    with xr.open_dataset(path, group="posterior", engine="h5netcdf") as ds:
        return ds[[v for v in VARS if v in ds]].load()


def _flat(ds: xr.Dataset, var: str, driver: str | None = None) -> np.ndarray:
    da = ds[var].sel(driver=driver) if driver else ds[var]
    return da.values.reshape(-1)


def plot(fits: dict[str, xr.Dataset], out: Path) -> pd.DataFrame:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from vop.plotting import ARROW, CTRL, INK, PHIL, annotate, title, use_opinionated

    use_opinionated()
    panels = [("b_h", "phil", "Hurdle: philosopher log-odds of any citation", "odds"),
              ("b_m", "phil", "Log-normal: philosopher effect on log citations", "cites"),
              ("sd_set_h", None, "Between-set SD, hurdle", None),
              ("sd_set_m", None, "Between-set SD, log-normal mean", None),
              ("sd_set_s", None, "Between-set SD, log-normal sigma", None)]
    colors = [PHIL, CTRL, INK]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 4.6))
    fig.subplots_adjust(top=0.72, bottom=0.16, left=0.03, right=0.99, wspace=0.12)
    rows = []
    for ax, (var, drv, label, kind) in zip(axes, panels):
        for (name, ds), col in zip(fits.items(), colors):
            x = _flat(ds, var, drv)
            lo, hi = np.percentile(x, [5.5, 94.5])
            ax.hist(x, bins=60, density=True, histtype="stepfilled", alpha=0.35, color=col, label=name)
            ax.hist(x, bins=60, density=True, histtype="step", color=col, linewidth=1.5)
            rows.append(dict(fit=name, param=var + (f"[{drv}]" if drv else ""), mean=x.mean(), sd=x.std(),
                             hdi89_lo=lo, hdi89_hi=hi, exp_mean=np.exp(x.mean()) if kind else np.nan))
        if kind:
            ax.axvline(0, color=INK, linewidth=1)
            x = _flat(list(fits.values())[0], var, drv)
            txt = (f"odds ×{np.exp(x.mean()):.2f}" if kind == "odds" else f"citations {np.exp(x.mean()) - 1:+.1%}")
            annotate(ax, 0.97, 0.92, txt, transform=ax.transAxes, ha="right", fontsize=11, fontweight="bold")
        ax.set_title(label, loc="left", fontsize=11)
        ax.set_yticks([])
        ax.grid(axis="y", visible=False)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    title(fig, "Bayesian hurdle log-normal: posteriors",
          "Within-set random intercepts in hurdle, mean and sigma; 500 draws × 4 chains. "
          f"Fits: {', '.join(fits)}. Annotation: posterior-mean effect on the first fit.")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return pd.DataFrame(rows)


def main(k: int = typer.Option(3), dev: bool = typer.Option(False)) -> None:
    cfg = load_config()
    d = cfg.derived / ("s07_dev" if dev else "s07")
    fits = {}
    for label, pattern in [("full, phil only", "hurdle_phil_only_n*.nc"), ("full, with drivers", "hurdle_with_drivers_n*.nc"),
                           ("pilot 3000 sets, phil only", "hurdle_phil_only.nc")]:
        for f in sorted(d.glob(pattern)):
            fits[label] = load_posterior(f)
            typer.echo(f"loaded {f.name} as '{label}'")
    if not fits:
        raise typer.Exit("no posterior files found")
    out = REPO_ROOT / "reports" / f"bayes_posteriors_k{k}.png"
    summ = plot(fits, out)
    summ.to_csv(d / "bayes_posterior_summary.csv", index=False)
    typer.echo(summ.round(3).to_string(index=False))
    typer.echo(f"figure -> {out}")
