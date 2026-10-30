"""Stage s06: analysis.

Reads the matched sets (s05) and estimates the within-set effect of being a philosopher's paper on
citations, for each matching radius k. Outcomes:
  log1p(cited_by_count)          all-time citations (OLS with set fixed effects, cluster SE by set)
  cited_by_count                 Poisson with set fixed effects (count-appropriate)
  fwci                           OpenAlex field-weighted citation impact (field/year normalised)
  cites_5y                       citations in publication year + 4 following years, from counts_by_year.
                                 OpenAlex only carries counts_by_year from 2012 on, so this outcome is
                                 restricted to sets whose focal work was published 2012..(snapshot-5).
Writes tables to data/derived/s06/ and figures to reports/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import typer

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pyfixest as pf  # noqa: E402

from vop.config import Config, REPO_ROOT, connect, load_config  # noqa: E402

COLORS = {"Control": "#2a78d6", "Philosopher": "#eb6834"}
SNAPSHOT_YEAR = 2026
CBY_FIRST_YEAR = 2012


def load_sets(cfg: Config, s05_dir: Path, s04_dir: Path, window: int) -> pd.DataFrame:
    con = connect(cfg)
    return con.execute(f"""
        WITH s AS (
          SELECT set_id, k, is_focal, work_id, publication_year, source_id, n_authors,
                 cited_by_count, fwci, cnp, subfield, field, domain,
                 list_sum(list_transform(
                     list_filter(counts_by_year, lambda c: c."year" BETWEEN publication_year AND publication_year + {window} - 1),
                     lambda c: c.cited_by_count)) AS cites_w
          FROM read_parquet('{s05_dir / "matched_sets.parquet"}')
        ),
        f AS (SELECT work_id AS set_id, first_author_is_phil, last_author_is_phil, n_philosophers
              FROM read_parquet('{s04_dir / "focal_works.parquet"}'))
        SELECT s.*, coalesce(s.cites_w, 0) AS cites_window,
               (s.publication_year BETWEEN {CBY_FIRST_YEAR} AND {SNAPSHOT_YEAR - window}) AS window_valid,
               f.first_author_is_phil, f.last_author_is_phil, f.n_philosophers
        FROM s JOIN f USING (set_id)
    """).fetchdf()


def fit_models(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k, d in df.groupby("k"):
        d = d.copy()
        d["phil"] = d["is_focal"].astype(int)
        d["log_cites"] = np.log1p(d["cited_by_count"])
        specs = [
            ("log1p cites, OLS FE", "log_cites ~ phil | set_id", d, "ols"),
            ("cites, Poisson FE", "cited_by_count ~ phil | set_id", d, "pois"),
            ("fwci, OLS FE", "fwci ~ phil | set_id", d.dropna(subset=["fwci"]), "ols"),
            (f"cites in window, Poisson FE", "cites_window ~ phil | set_id", d[d["window_valid"]], "pois"),
        ]
        for name, fml, dd, kind in specs:
            if dd.empty:
                continue
            try:
                m = (pf.feols if kind == "ols" else pf.fepois)(fml, data=dd, vcov={"CRV1": "set_id"})
                b, se = float(m.coef()["phil"]), float(m.se()["phil"])
                rows.append(dict(k=k, model=name, n_obs=int(m._N), n_sets=dd["set_id"].nunique(),
                                 coef=b, se=se, ci_low=b - 1.96 * se, ci_high=b + 1.96 * se,
                                 ratio=float(np.exp(b)) if kind == "pois" or "log1p" in name else np.nan))
            except Exception as e:  # noqa: BLE001
                rows.append(dict(k=k, model=name, error=str(e)[:200]))
    return pd.DataFrame(rows)


def descriptives(df: pd.DataFrame) -> pd.DataFrame:
    g = df.assign(group=np.where(df["is_focal"], "Philosopher", "Control"))
    out = g.groupby(["k", "group"]).agg(
        n=("work_id", "size"), mean_cites=("cited_by_count", "mean"), median_cites=("cited_by_count", "median"),
        share_zero=("cited_by_count", lambda x: (x == 0).mean()), mean_fwci=("fwci", "mean"),
        median_fwci=("fwci", "median"),
    ).round(3)
    return out.reset_index()


def heterogeneity(df: pd.DataFrame, k: int) -> pd.DataFrame:
    d = df[df["k"] == k].copy()
    d["phil"] = d["is_focal"].astype(int)
    d["log_cites"] = np.log1p(d["cited_by_count"])
    d["decade"] = (d["publication_year"] // 10 * 10).astype(int)
    # set-level attributes come from the focal work
    focal_attr = d[d["is_focal"]].set_index("set_id")[["domain", "decade", "first_author_is_phil"]]
    rows = []
    for col in ["domain", "decade", "first_author_is_phil"]:
        for val, sets in focal_attr.groupby(col).groups.items():
            dd = d[d["set_id"].isin(sets)]
            if dd["set_id"].nunique() < 100:
                continue
            m = pf.feols("log_cites ~ phil | set_id", data=dd, vcov={"CRV1": "set_id"})
            b, se = float(m.coef()["phil"]), float(m.se()["phil"])
            rows.append(dict(split=col, value=str(val), n_sets=dd["set_id"].nunique(), coef=b, se=se,
                             ratio=float(np.exp(b))))
    return pd.DataFrame(rows)


def plot_distribution(df: pd.DataFrame, k: int, path: Path) -> None:
    d = df[df["k"] == k]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    for i, (label, sel) in enumerate([("Control", ~d["is_focal"]), ("Philosopher", d["is_focal"])]):
        x = d.loc[sel, "cited_by_count"].clip(lower=0) + 1
        ax.boxplot(x, positions=[i], vert=False, widths=0.5, whis=(5, 95), showfliers=False,
                   boxprops=dict(color=COLORS[label], linewidth=2), whiskerprops=dict(color=COLORS[label]),
                   capprops=dict(color=COLORS[label]), medianprops=dict(color=COLORS[label], linewidth=2))
    ax.set_xscale("log")
    ax.set_yticks([0, 1], ["Control", "Philosopher"])
    ax.set_xlabel("Citations + 1 (log scale); box = IQR, whiskers = 5th–95th percentile")
    ax.set_title(f"All-time citations, matched sets (k = ±{k})")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#e0dfdb", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_coefficients(models: pd.DataFrame, path: Path, df: pd.DataFrame | None = None) -> None:
    m = models.dropna(subset=["coef"]) if "coef" in models else models
    if m.empty:
        return
    base = {}
    if df is not None:
        for k, d in df.groupby("k"):
            c = d[~d["is_focal"]]
            base[int(k)] = dict(mean=c["cited_by_count"].mean(), fwci=c["fwci"].mean(),
                                win=c.loc[c["window_valid"], "cites_window"].mean())
    names = list(dict.fromkeys(m["model"]))
    fig, ax = plt.subplots(figsize=(10, 0.6 * len(m) + 1.5))
    y = 0
    ticks, labels = [], []
    for name in names:
        for _, r in m[m["model"] == name].iterrows():
            ax.errorbar(r["coef"], y, xerr=[[r["coef"] - r["ci_low"]], [r["ci_high"] - r["coef"]]],
                        fmt="o", color=COLORS["Philosopher"], ecolor=COLORS["Philosopher"], capsize=3, markersize=6)
            b = base.get(int(r["k"]))
            if b:
                if "fwci" in name:
                    txt = f"FWCI {b['fwci']:.2f} → {b['fwci'] + r['coef']:.2f}"
                elif "window" in name:
                    txt = f"mean {b['win']:.1f} → {b['win'] * np.exp(r['coef']):.1f} cites in {5} yrs"
                elif "Poisson" in name:
                    txt = f"mean {b['mean']:.1f} → {b['mean'] * np.exp(r['coef']):.1f} cites"
                else:
                    txt = f"×{np.exp(r['coef']):.3f} on (cites+1)"
                ax.text(0.26, y, txt, va="center", fontsize=8, color="#52514e")
            ticks.append(y)
            labels.append(f"{name}, k=±{int(r['k'])}")
            y += 1
        y += 0.5
    ax.axvline(0, color="#52514e", linewidth=1)
    ax.set_xlim(right=0.5)
    ax.set_yticks(ticks, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Coefficient on 'philosopher' within matched set (95% CI, SE clustered by set); "
                  "annotation: controls → philosophers")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#e0dfdb", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(dev: bool = typer.Option(False)) -> None:
    cfg = load_config()
    d = "_dev" if dev else ""
    out = cfg.derived / f"s06{d}"
    out.mkdir(parents=True, exist_ok=True)
    reports = REPO_ROOT / "reports"
    reports.mkdir(exist_ok=True)
    window = int(cfg.params.get("citation_window_years", 5))

    df = load_sets(cfg, cfg.derived / f"s05{d}", cfg.derived / f"s04{d}", window)
    typer.echo(f"{len(df):,} rows, {df['set_id'].nunique():,} sets")

    desc = descriptives(df)
    desc.to_csv(out / "descriptives.csv", index=False)
    typer.echo(desc.to_string(index=False))

    models = fit_models(df)
    models.to_csv(out / "models.csv", index=False)
    typer.echo(models.to_string(index=False))

    k_main = int(cfg.params["max_year_diff"])
    het = heterogeneity(df, k_main)
    het.to_csv(out / f"heterogeneity_k{k_main}.csv", index=False)
    typer.echo(het.to_string(index=False))

    plot_distribution(df, k_main, reports / f"citations_boxplot_k{k_main}.png")
    plot_coefficients(models, reports / "coefficients.png", df)
    typer.echo(f"figures -> {reports}")
