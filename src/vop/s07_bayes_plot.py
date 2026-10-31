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


DRIVER_LABELS = {
    "log_refs": "references, per sd", "year_dev": "later year, per sd", "nauth_dev": "more authors, per sd",
    "prestige_cnp": "top author track record, per sd", "no_prior_record": "no earlier papers (0/1)",
    "log_prior_works_max": "senior author prior works, per sd", "has_abstract": "abstract available (0/1)",
}


def plot(fits: dict[str, xr.Dataset], out: Path) -> pd.DataFrame:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from vop.plotting import AQUA, CTRL, INK, PHIL, annotate, title, use_opinionated

    use_opinionated()
    colors = dict(zip(fits, [CTRL, PHIL, AQUA]))
    fig = plt.figure(figsize=(16, 10.5))
    gs = fig.add_gridspec(2, 6, top=0.87, bottom=0.08, left=0.04, right=0.985, hspace=0.34, wspace=0.5)
    rows = []
    # philosopher effect in each process
    for spec, var, label, kind in [(gs[0, 0:3], "b_h", "Philosopher effect: cited at least once (log-odds)", "odds"),
                                   (gs[0, 3:6], "b_m", "Philosopher effect: log citations, if cited", "cites")]:
        ax = fig.add_subplot(spec)
        for i, (name, ds) in enumerate(fits.items()):
            x = _flat(ds, var, "phil")
            lo, hi = np.percentile(x, [5.5, 94.5])
            ax.hist(x, bins=50, density=True, histtype="stepfilled", alpha=0.3, color=colors[name], label=name)
            ax.hist(x, bins=50, density=True, histtype="step", color=colors[name], linewidth=1.5)
            eff = f"odds x{np.exp(x.mean()):.2f}" if kind == "odds" else f"citations {np.exp(x.mean()) - 1:+.1%}"
            annotate(ax, 0.985, 0.93 - 0.075 * i, f"{name}: {eff}", transform=ax.transAxes, ha="right", fontsize=10)
            rows.append(dict(fit=name, param=f"{var}[phil]", mean=x.mean(), sd=x.std(), eti89_lo=lo, eti89_hi=hi))
        ax.axvline(0, color=INK, linewidth=1)
        ax.set_title(label, loc="left", fontsize=13)
        ax.set_yticks([])
        ax.grid(axis="y", visible=False)
    # covariates, from the richest fit
    name = list(fits)[-1]
    ds = fits[name]
    drivers = [str(v) for v in ds["driver"].values if str(v) != "phil"]
    for spec, var, label in [(gs[1, 0:2], "b_h", "Covariates: cited at least once (log-odds)"), (gs[1, 2:4], "b_m", "Covariates: log citations, if cited")]:
        ax = fig.add_subplot(spec)
        for yi, drv in enumerate(drivers):
            x = _flat(ds, var, drv)
            lo, hi = np.percentile(x, [5.5, 94.5])
            ax.errorbar(x.mean(), yi, xerr=[[x.mean() - lo], [hi - x.mean()]], fmt="o", color=CTRL, ecolor=CTRL, capsize=3, markersize=6)
            rows.append(dict(fit=name, param=f"{var}[{drv}]", mean=x.mean(), sd=x.std(), eti89_lo=lo, eti89_hi=hi))
        ax.axvline(0, color=INK, linewidth=1)
        ax.set_ylim(len(drivers) - 0.5, -0.5)
        ax.set_yticks(range(len(drivers)), [DRIVER_LABELS.get(v, v) for v in drivers] if var == "b_h" else [])
        ax.tick_params(axis="y", labelsize=10)
        ax.set_title(label, loc="left", fontsize=13)
        ax.grid(axis="y", visible=False)
    # how much sets differ
    ax = fig.add_subplot(gs[1, 4:6])
    sds = [("sd_set_h", "cited at least once"), ("sd_set_m", "citations, mean"), ("sd_set_s", "citations, spread")]
    for yi, (var, lab) in enumerate(sds):
        for j, (nm, d_) in enumerate(fits.items()):
            x = _flat(d_, var)
            lo, hi = np.percentile(x, [5.5, 94.5])
            ax.errorbar(x.mean(), yi + (j - 1) * 0.2, xerr=[[x.mean() - lo], [hi - x.mean()]], fmt="o", color=colors[nm], ecolor=colors[nm], capsize=2, markersize=5)
            rows.append(dict(fit=nm, param=var, mean=x.mean(), sd=x.std(), eti89_lo=lo, eti89_hi=hi))
    ax.set_ylim(2.6, -0.6)
    ax.set_yticks(range(3), [b for _, b in sds])
    ax.tick_params(axis="y", labelsize=10)
    ax.set_title("Differences between matched sets (sd)", loc="left", fontsize=13)
    ax.grid(axis="y", visible=False)
    fig.axes[0].legend(frameon=False, fontsize=10, loc="upper left")
    title(fig, "Bayesian hurdle log-normal: posteriors",
          f"All matched sets, 500 draws x 4 chains, random effects per set in the hurdle, mean and spread. Intervals: 89%. Covariates from '{name}'.")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return pd.DataFrame(rows)


def main(k: int = typer.Option(3), dev: bool = typer.Option(False)) -> None:
    cfg = load_config()
    d = cfg.derived / ("s07_dev" if dev else "s07")
    fits = {}
    for label, pattern in [("philosopher only", "hurdle_phil_only_n*.nc"), ("+ year, authors, references", "hurdle_with_drivers_n*.nc"),
                           ("+ track record", "hurdle_with_track_n*.nc")]:
        for f in sorted(d.glob(pattern)):
            fits[label] = load_posterior(f)
            typer.echo(f"loaded {f.name} as '{label}'")
    if not fits:
        raise typer.Exit("no posterior files found")
    cfg.reports.mkdir(parents=True, exist_ok=True)
    out = cfg.reports / f"bayes_posteriors_k{k}.png"
    summ = plot(fits, out)
    summ.to_csv(d / "bayes_posterior_summary.csv", index=False)
    typer.echo(summ.round(3).to_string(index=False))
    typer.echo(f"figure -> {out}")
