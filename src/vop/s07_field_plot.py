"""Philosopher effect by field from the Bayesian field model (`vop s07-hurdle --bayes-only --fit field_phil`).

Same layout as the fixed-effects field figure. Also writes the field table and all pairwise field
differences with 89% intervals and the posterior probability that the difference is above zero.
"""

from __future__ import annotations

import itertools
import json

import matplotlib
import numpy as np
import pandas as pd
import typer
import xarray as xr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from vop import s07_hurdle as h  # noqa: E402
from vop.config import load_config  # noqa: E402
from vop.plotting import ARROW, CTRL, INK, PHIL, annotate, title, use_opinionated  # noqa: E402

# label, multiplier applied to the coefficient per raw unit, so rows read in natural units
UNITS = {
    "year_dev": ("All fields: +1 year (within set)", 1.0),
    "nauth_dev": ("All fields: +1 author", 1.0),
    "log_refs": ("All fields: references ×2.7", 1.0),
    "prestige_cnp": ("All fields: top author track record +10 pct. pts", 0.1),
    "no_prior_record": ("All fields: no earlier papers", 1.0),
    "log_prior_works_max": ("All fields: senior author prior works ×2.7", 1.0),
    "has_abstract": ("All fields: abstract available", 1.0),
}


def _iv(x: np.ndarray) -> tuple[float, float, float]:
    lo, hi = np.percentile(x, [5.5, 94.5])
    return float(x.mean()), float(lo), float(hi)


def main(k: int = typer.Option(3)) -> None:
    cfg = load_config()
    s07 = cfg.derived / "s07"
    fits = sorted(s07.glob("hurdle_field_phil_n*.nc"), key=lambda p: int(p.stem.split("_n")[-1]))
    if not fits:
        raise typer.Exit("no field model found")
    path = fits[-1]
    scales = json.loads(path.with_name(path.stem + ".scales.json").read_text())
    names = ["b_h_field", "b_m_field", "mu_h", "mu_m", "tau_h", "tau_m", "b_h", "b_m"]
    with xr.open_dataset(path, group="posterior", engine="h5netcdf") as ds:
        p = ds[names].load()
    fields = [str(f) for f in p["field"].values]
    H = p["b_h_field"].values.reshape(-1, len(fields))
    M = p["b_m_field"].values.reshape(-1, len(fields))

    d = h.load(cfg, cfg.derived / "s05", k)
    d["field_g"] = d["set_field"].fillna(h.NO_FIELD)
    obs = d.groupby(["field_g", "phil"])["cited"].mean().unstack()
    n_sets = d.groupby("field_g")["set_id"].nunique()
    rows = []
    for i, f in enumerate(fields):
        hm, hl, hh = _iv(H[:, i])
        mm, ml, mh = _iv(M[:, i])
        rows.append(dict(field=f, n_sets=int(n_sets.get(f, 0)), raw_ctrl=obs.loc[f, 0], raw_phil=obs.loc[f, 1],
                         h_mean=hm, h_lo=hl, h_hi=hh, odds_ratio=np.exp(hm), m_mean=mm, m_lo=ml, m_hi=mh, cites_pct=np.exp(mm) - 1,
                         p_h_pos=(H[:, i] > 0).mean(), p_m_pos=(M[:, i] > 0).mean()))
    t = pd.DataFrame(rows).sort_values("h_mean").reset_index(drop=True)
    t.to_csv(s07 / f"field_model_effects_k{k}.csv", index=False)
    pairs = []
    for a, b in itertools.combinations(range(len(fields)), 2):
        for part, Z in [("hurdle", H), ("lognormal", M)]:
            dx = Z[:, a] - Z[:, b]
            mn, lo, hi = _iv(dx)
            pairs.append(dict(part=part, field_a=fields[a], field_b=fields[b], diff=mn, lo=lo, hi=hi, p_a_gt_b=(dx > 0).mean()))
    pd.DataFrame(pairs).to_csv(s07 / f"field_model_differences_k{k}.csv", index=False)

    tau_h, tau_m = _iv(p["tau_h"].values.ravel()), _iv(p["tau_m"].values.ravel())
    p0, p1 = float(d.loc[d.phil == 0, "cited"].mean()), float(d.loc[d.phil == 1, "cited"].mean())
    block = [("All fields: philosopher", _iv(p["mu_h"].values.ravel()), _iv(p["mu_m"].values.ravel()), True)]
    for drv in [str(v) for v in p["driver"].values]:
        label, mult = UNITS.get(drv, (drv, 1.0))
        c = mult / scales[drv]["sd"]
        block.append((label, _iv(p["b_h"].sel(driver=drv).values.ravel() * c), _iv(p["b_m"].sel(driver=drv).values.ravel() * c), False))
    pd.DataFrame([dict(row=b[0], h_mean=b[1][0], h_lo=b[1][1], h_hi=b[1][2], m_mean=b[2][0], m_lo=b[2][1], m_hi=b[2][2]) for b in block]).to_csv(s07 / f"field_model_pooled_k{k}.csv", index=False)

    use_opinionated()
    n = len(t) + len(block)
    gap = 0.6
    ys = list(range(len(t))) + [len(t) + gap + j for j in range(len(block))]
    fh = 0.42 * n + 2.6
    fig, axes = plt.subplots(1, 2, figsize=(16, fh), sharey=True)
    fig.subplots_adjust(top=1 - 1.35 / fh, wspace=0.06, left=0.3, right=0.985, bottom=0.9 / fh)
    for ax, part in zip(axes, ["hurdle", "lognormal"]):
        recs = [(r.h_mean, r.h_lo, r.h_hi) if part == "hurdle" else (r.m_mean, r.m_lo, r.m_hi) for r in t.itertuples()]
        recs += [b[1] if part == "hurdle" else b[2] for b in block]
        cols = [PHIL] * len(t) + [PHIL if b[3] else CTRL for b in block]
        for yi, (m, lo, hi), c in zip(ys, recs, cols):
            ax.errorbar(m, yi, xerr=[[m - lo], [hi - m]], fmt="o", color=c, ecolor=c, capsize=3, markersize=6, linewidth=1.5)
        ax.axvline(0, color=INK, linewidth=1)
        ax.set_xlabel("coefficient (89% interval)")
        ax.set_title("Cited at least once" if part == "hurdle" else "Citations, if cited", loc="left", fontsize=14)
        xmax = max(r[2] for r in recs)
        xa = xmax * 1.1
        annotate(ax, xa, -0.9, f"controls {ARROW} phil. (raw)" if part == "hurdle" else "adjusted change", fontsize=10, fontweight="bold")
        for j, (yi, (m, _, _)) in enumerate(zip(ys, recs)):
            if part == "hurdle" and j < len(t):
                txt = f"{t.raw_ctrl[j]:.0%} {ARROW} {t.raw_phil[j]:.0%}"
            elif part == "hurdle":
                txt = f"{p0:.0%} {ARROW} {p1:.0%}" if j == len(t) else f"odds {np.exp(m) - 1:+.0%}"
            else:
                txt = f"{np.exp(m) - 1:+.0%}"
            annotate(ax, xa, yi, txt, fontsize=10)
        ax.set_xlim(right=xmax * 1.75)
        ax.set_ylim(ys[-1] + 0.5, -1.4)
        ax.axhline(len(t) - 0.5 + gap / 2, color="#d0cfca", linewidth=1)
        ax.grid(axis="y", visible=False)
    labels = [f"{r.field}  (n={r.n_sets:,})" for r in t.itertuples()] + [b[0] for b in block]
    axes[0].set_yticks(ys, labels)
    title(fig, "Philosophers' papers in other fields' journals: Bayesian field model",
          f"One model, philosopher effect per field, partially pooled. Spread across fields: {tau_h[0]:.2f} on log-odds, {tau_m[0]:.2f} on log citations. "
          "Left labels: raw observed shares.")
    cfg.reports.mkdir(parents=True, exist_ok=True)
    out = cfg.reports / f"hurdle_field_bayes_k{k}.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    typer.echo(t.round(3).to_string(index=False))
    typer.echo(f"tau_h {tau_h}\ntau_m {tau_m}\nfigure -> {out}")
