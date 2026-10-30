"""Stage s07: hurdle log-normal models of citations.

Two processes: whether a paper gets cited at all (the hurdle) and, given it is cited, how many
citations it accumulates (log-normal). Two tiers:

  frequentist  set fixed effects in both parts: FE logit (pyfixest.feglm) on 1[cites > 0] and
               FE OLS on log(cites) among cited papers; covariates beyond the matching variables.
  bayesian     PyMC HurdleLogNormal mirroring brms::hurdle_lognormal: random intercepts per matched
               set in the hurdle, in mu, and in sigma (non-centred), same drivers in hurdle and mu.
               Sampled with nutpie (NUTS). PSIS-LOO via ArviZ, with and without the drivers, so the
               model comparison in the brms write-up can be reproduced.

Covariates (within-set deviations, so they are identified separately from the set effect):
  log_refs    log1p(referenced_works_count)   reference-list length, a proxy for text length / genre
  year_dev    publication_year - set mean     residual age difference inside the +-k window
  nauth_dev   n_authors - set mean            residual team-size difference
"""

from __future__ import annotations

from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pyfixest as pf
import pymc as pm
import typer

from vop.config import REPO_ROOT, Config, connect, load_config

BASE_DRIVERS = ["phil", "log_refs", "year_dev", "nauth_dev"]
PRESTIGE_DRIVERS = ["prestige_max_nonphil", "log_prior_works_max"]
DRIVERS = list(BASE_DRIVERS)  # extended at runtime when s08 output exists


def load(cfg: Config, s05_dir: Path, k: int) -> pd.DataFrame:
    con = connect(cfg)
    d = con.execute(f"""
        WITH s AS (SELECT * FROM read_parquet('{s05_dir / "matched_sets.parquet"}') WHERE k = {k}),
             f AS (SELECT set_id, field AS set_field, domain AS set_domain FROM s WHERE is_focal)
        SELECT s.set_id, s.work_id, s.is_focal, s.publication_year, s.n_authors, s.cited_by_count,
               coalesce(s.referenced_works_count, 0) AS n_refs, f.set_field, f.set_domain
        FROM s JOIN f USING (set_id)
    """).fetchdf()
    d["phil"] = d["is_focal"].astype(int)
    d["cited"] = (d["cited_by_count"] > 0).astype(int)
    d["log_cites"] = np.log(d["cited_by_count"].where(d["cited_by_count"] > 0))
    d["log_refs"] = np.log1p(d["n_refs"])
    for col, new in [("publication_year", "year_dev"), ("n_authors", "nauth_dev")]:
        d[new] = d[col] - d.groupby("set_id")[col].transform("mean")
    prestige = s05_dir.parent / "s08" / "paper_prestige.parquet"
    if prestige.exists():
        pr = con.execute(f"SELECT * FROM read_parquet('{prestige}')").fetchdf()
        d = d.merge(pr, on="work_id", how="left")
        d["prestige_max_nonphil"] = d["prestige_max_nonphil"].fillna(0.0)  # no non-philosopher coauthor
        d["log_prior_works_max"] = np.log1p(d["n_prior_works_max"].fillna(0))
        for c in PRESTIGE_DRIVERS:
            if c not in DRIVERS:
                DRIVERS.append(c)
        typer.echo(f"prestige covariates attached ({len(pr):,} papers)")
    return d


def frequentist(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    specs = [
        ("hurdle: P(cited) FE logit, phil only", "cited ~ phil | set_id", "logit", d),
        ("hurdle: P(cited) FE logit, + covariates", "cited ~ " + " + ".join(DRIVERS) + " | set_id", "logit", d),
        ("log-normal: log cites | cited, FE OLS, phil only", "log_cites ~ phil | set_id", "ols", d[d.cited == 1]),
        ("log-normal: log cites | cited, FE OLS, + covariates", "log_cites ~ " + " + ".join(DRIVERS) + " | set_id",
         "ols", d[d.cited == 1]),
    ]
    for name, fml, kind, dd in specs:
        if kind == "logit":
            m = pf.feglm(fml, data=dd, family="logit", vcov={"CRV1": "set_id"})
        else:
            m = pf.feols(fml, data=dd, vcov={"CRV1": "set_id"})
        for term in m.coef().index:
            b, se = float(m.coef()[term]), float(m.se()[term])
            rows.append(dict(model=name, term=term, coef=b, se=se, ci_low=b - 1.96 * se, ci_high=b + 1.96 * se,
                             exp_coef=float(np.exp(b)), n_obs=int(m._N), n_sets=int(dd["set_id"].nunique())))
    return pd.DataFrame(rows)


def by_field(d: pd.DataFrame, min_sets: int = 300) -> pd.DataFrame:
    """Hurdle and log-normal parts (with covariates) estimated separately within each field."""
    rows = []
    fml = " + ".join(DRIVERS)
    for field, dd in d.groupby("set_field"):
        n_sets = dd["set_id"].nunique()
        if n_sets < min_sets:
            continue
        try:
            h = pf.feglm(f"cited ~ {fml} | set_id", data=dd, family="logit", vcov={"CRV1": "set_id"})
            m = pf.feols(f"log_cites ~ {fml} | set_id", data=dd[dd.cited == 1], vcov={"CRV1": "set_id"})
        except Exception as e:  # noqa: BLE001
            typer.echo(f"  {field}: skipped ({str(e)[:80]})")
            continue
        ctrl = dd[dd.phil == 0]
        p0 = float(ctrl["cited"].mean())                       # controls' share cited
        med0 = float(ctrl.loc[ctrl.cited == 1, "cited_by_count"].median())  # controls' median cites | cited
        for part, mod in [("hurdle", h), ("lognormal", m)]:
            for term in mod.coef().index:
                b, se = float(mod.coef()[term]), float(mod.se()[term])
                rows.append(dict(field=field, part=part, term=term, n_sets=n_sets, coef=b, se=se,
                                 ci_low=b - 1.96 * se, ci_high=b + 1.96 * se,
                                 ctrl_share_cited=p0, ctrl_median_cites_if_cited=med0,
                                 phil_share_cited=_shift_prob(p0, b) if (part, term) == ("hurdle", "phil") else np.nan,
                                 phil_median_cites_if_cited=med0 * np.exp(b) if (part, term) == ("lognormal", "phil") else np.nan))
    return pd.DataFrame(rows)


def _shift_prob(p0: float, log_odds: float) -> float:
    """Probability after adding `log_odds` to the baseline probability p0 on the logit scale."""
    return float(1 / (1 + np.exp(-(np.log(p0 / (1 - p0)) + log_odds))))


POOLED_ROWS = [  # (term, label) rows added below the fields; from the pooled with-covariate models
    ("phil", "All fields: philosopher"),
    ("year_dev", "All fields: +1 year (within set)"),
    ("nauth_dev", "All fields: +1 author"),
    ("log_refs", "All fields: references ×2.7"),
]


def plot_by_field(bf: pd.DataFrame, path: Path, pooled: pd.DataFrame | None = None, d: pd.DataFrame | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from vop.plotting import ARROW, CTRL, INK, PHIL, annotate, title, use_opinionated

    use_opinionated()
    fields = bf[(bf.term == "phil") & (bf.part == "hurdle")].sort_values("coef")["field"].tolist()
    rows = []  # (label, hurdle row, lognormal row, is_phil)
    for f in fields:
        h = bf[(bf.field == f) & (bf.term == "phil") & (bf.part == "hurdle")].iloc[0].copy()
        m = bf[(bf.field == f) & (bf.term == "phil") & (bf.part == "lognormal")].iloc[0]
        if d is not None:  # label with observed shares; FE-logit coefficients do not translate into shares
            sub = d[d.set_field == f]
            h["ctrl_share_cited"] = float(sub.loc[sub.phil == 0, "cited"].mean())
            h["phil_share_cited"] = float(sub.loc[sub.phil == 1, "cited"].mean())
        rows.append((f"{f}  (n={int(h.n_sets):,})", h, m, True))
    pooled_rows = []
    if pooled is not None and d is not None:
        p0, p1 = float(d.loc[d.phil == 0, "cited"].mean()), float(d.loc[d.phil == 1, "cited"].mean())
        extra = [(t, COVARIATE_LABELS[t][0].replace("Coauthor prestige", "All fields: coauthor prestige 0→1")
                  .replace("Senior-author career length", "All fields: senior author prior works ×2.7"))
                 for t in PRESTIGE_DRIVERS if t in set(pooled.term)]
        for term, label in POOLED_ROWS + extra:
            hp = pooled[(pooled.model.str.startswith("hurdle")) & pooled.model.str.contains("covariates") & (pooled.term == term)]
            mp = pooled[(pooled.model.str.startswith("log-normal")) & pooled.model.str.contains("covariates") & (pooled.term == term)]
            if hp.empty or mp.empty:
                continue
            h, m = hp.iloc[0].copy(), mp.iloc[0].copy()
            h["ctrl_share_cited"], h["phil_share_cited"] = p0, p1
            pooled_rows.append((label, h, m, term == "phil"))
    all_rows = rows + pooled_rows
    n = len(all_rows)
    gap = 0.6 if pooled_rows else 0.0
    ys = [i + (gap if i >= len(rows) else 0.0) for i in range(n)]
    h_in = 0.42 * n + 2.4 + gap * 0.42
    fig, axes = plt.subplots(1, 2, figsize=(16, h_in), sharey=True)
    fig.subplots_adjust(top=1 - 1.35 / h_in, wspace=0.06, left=0.3, right=0.985, bottom=0.9 / h_in)
    panels = {"hurdle": ("Cited at least once", f"observed: controls {ARROW} philosophers"),
              "lognormal": ("Citations, if cited", "adjusted change")}
    for ax, part, idx in zip(axes, ["hurdle", "lognormal"], [1, 2]):
        recs = [r[idx] for r in all_rows]
        cols = [PHIL if r[3] else CTRL for r in all_rows]
        for yi, rec, c in zip(ys, recs, cols):
            ax.errorbar(rec.coef, yi, xerr=[[rec.coef - rec.ci_low], [rec.ci_high - rec.coef]],
                        fmt="o", color=c, ecolor=c, capsize=3, markersize=6, linewidth=1.5)
        ax.axvline(0, color=INK, linewidth=1)
        ax.set_title(panels[part][0], loc="left", fontsize=14)
        ax.set_xlabel("coefficient (95% CI)")
        xmax = max(float(r.ci_high) for r in recs)
        xa = xmax * 1.1
        annotate(ax, xa, -0.9, panels[part][1], fontsize=10, fontweight="bold")
        for yi, rec, (_, _, _, is_phil) in zip(ys, recs, all_rows):
            if part == "hurdle" and is_phil:
                txt = f"{rec.ctrl_share_cited:.0%} {ARROW} {rec.phil_share_cited:.0%}"
            elif part == "hurdle":
                txt = f"odds {np.exp(rec.coef) - 1:+.0%}"
            else:
                txt = f"{np.exp(rec.coef) - 1:+.0%}"
            annotate(ax, xa, yi, txt, fontsize=10)
        ax.set_xlim(right=xmax * 1.75)
        ax.set_ylim(ys[-1] + 0.5, -1.4)
        ax.grid(axis="y", visible=False)
        if pooled_rows:
            ax.axhline(len(rows) - 0.5 + gap / 2, color="#d0cfca", linewidth=1)
    axes[0].set_yticks(ys, [r[0] for r in all_rows])
    title(fig, "Philosophers' papers in other fields' journals: effect by field",
          "Points: within-set coefficients with covariates (orange philosopher, blue pooled covariates per unit). "
          "Left labels: raw observed shares, not derived from the coefficient.")
    fig.savefig(path, dpi=160)
    plt.close(fig)


COVARIATE_LABELS = {
    "year_dev": ("Publication year", "per year later than the set mean"),
    "nauth_dev": ("Number of authors", "per additional author"),
    "log_refs": ("Reference-list length", "per log unit of references"),
    "prestige_max_nonphil": ("Coauthor prestige", "0 → 1 percentile of best non-philosopher coauthor"),
    "log_prior_works_max": ("Senior-author career length", "per log unit of prior works"),
}


def plot_covariates_by_field(bf: pd.DataFrame, path: Path) -> None:
    """One column per covariate, hurdle on top and log-normal below, one row per field."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from vop.plotting import CTRL, INK, annotate, title, use_opinionated

    use_opinionated()
    terms = [t for t in COVARIATE_LABELS if t in set(bf.term)]
    order = bf[(bf.part == "hurdle") & (bf.term == "phil")].sort_values("coef")["field"].tolist()
    n = len(order)
    h = 2 * (0.3 * n + 1.2) + 1.6
    fig, axes = plt.subplots(2, len(terms), figsize=(4.2 * len(terms) + 3, h), sharey=True)
    fig.subplots_adjust(top=1 - 1.3 / h, bottom=0.6 / h, left=0.25, right=0.985, wspace=0.08, hspace=0.25)
    y = np.arange(n)
    for j, term in enumerate(terms):
        for i, part in enumerate(["hurdle", "lognormal"]):
            ax = axes[i, j]
            sub = bf[(bf.part == part) & (bf.term == term)].set_index("field").reindex(order)
            ax.errorbar(sub["coef"], y, xerr=[sub["coef"] - sub["ci_low"], sub["ci_high"] - sub["coef"]],
                        fmt="o", color=CTRL, ecolor=CTRL, capsize=3, markersize=5, linewidth=1.3)
            ax.axvline(0, color=INK, linewidth=1)
            ax.set_ylim(n - 0.5, -0.8)
            ax.grid(axis="y", visible=False)
            if i == 0:
                ax.set_title(COVARIATE_LABELS[term][0], loc="left", fontsize=13)
            ax.set_xlabel(("odds of any citation" if part == "hurdle" else "citations if cited")
                          + f", {COVARIATE_LABELS[term][1]}", fontsize=9)
            xmax = float(sub["ci_high"].max())
            for yi, (_, r) in zip(y, sub.iterrows()):
                annotate(ax, xmax * 1.08, yi, f"{np.exp(r.coef) - 1:+.0%}", fontsize=8)
            ax.set_xlim(right=xmax * 1.6)
    for i in range(2):
        axes[i, 0].set_yticks(y, order)
    axes[0, 0].set_ylabel("Hurdle", fontsize=12)
    axes[1, 0].set_ylabel("Log-normal", fontsize=12)
    title(fig, "Covariate effects by field",
          "Same within-set models as the philosopher plot. Annotations: multiplicative change per unit "
          "(odds in the top row, citations in the bottom row).")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def build_bayesian(d: pd.DataFrame, with_drivers: bool) -> pm.Model:
    set_idx, sets = pd.factorize(d["set_id"])
    y = d["cited_by_count"].to_numpy().astype(float)
    X = d[DRIVERS].to_numpy() if with_drivers else d[["phil"]].to_numpy()
    # standardise the continuous drivers only; keep phil as a 0/1 indicator so its coefficient is a log-odds /
    # log-ratio directly comparable with the frequentist FE estimates
    X = X.astype(float)
    for j in range(1, X.shape[1]):
        sd = X[:, j].std()
        X[:, j] = (X[:, j] - X[:, j].mean()) / (sd if sd > 0 else 1)
    with pm.Model(coords={"set": sets, "driver": DRIVERS if with_drivers else ["phil"]}) as model:
        # hurdle (logit of P(cited))
        a_h = pm.Normal("a_h", 0, 1.5)
        b_h = pm.Normal("b_h", 0, 1, dims="driver")
        s_h = pm.HalfNormal("sd_set_h", 1)
        z_h = pm.Normal("z_h", 0, 1, dims="set")
        psi = pm.math.sigmoid(a_h + X @ b_h + s_h * z_h[set_idx])
        # log-normal mean
        a_m = pm.Normal("a_m", 1, 1.5)
        b_m = pm.Normal("b_m", 0, 1, dims="driver")
        s_m = pm.HalfNormal("sd_set_m", 1)
        z_m = pm.Normal("z_m", 0, 1, dims="set")
        mu = a_m + X @ b_m + s_m * z_m[set_idx]
        # log-normal sigma: intercept + set random effect only (as in the brms spec)
        a_s = pm.Normal("a_s", 0, 1)
        s_s = pm.HalfNormal("sd_set_s", 0.5)
        z_s = pm.Normal("z_s", 0, 1, dims="set")
        sigma = pm.math.exp(a_s + s_s * z_s[set_idx])
        pm.HurdleLogNormal("y", psi=psi, mu=mu, sigma=sigma, observed=y)
    return model


def bayesian(d: pd.DataFrame, out: Path, draws: int, tune: int, chains: int, loglik: bool = True) -> pd.DataFrame:
    rows, idatas = [], {}
    for label, with_drivers in [("phil_only", False), ("with_drivers", True)]:
        model = build_bayesian(d, with_drivers)
        with model:
            idata = pm.sample(draws=draws, tune=tune, chains=chains, nuts_sampler="nutpie",
                              idata_kwargs={"log_likelihood": loglik}, progressbar=False)
        idata.to_netcdf(out / f"hurdle_{label}_n{d.set_id.nunique()}.nc")
        typer.echo(f"[{label}] sampled {d.set_id.nunique()} sets", err=False)
        idatas[label] = idata
        summ = az.summary(idata, var_names=["a_h", "b_h", "a_m", "b_m", "a_s", "sd_set_h", "sd_set_m", "sd_set_s"])
        summ.insert(0, "model", label)
        typer.echo(f"[{label}] divergences: {int(idata.sample_stats['diverging'].sum())}")
        typer.echo(summ.round(3).to_string())
        rows.append(summ.reset_index().rename(columns={"index": "param"}))
    if not loglik:
        return pd.concat(rows, ignore_index=True)
    try:
        comp = az.compare(idatas)  # PSIS-LOO by default
        comp.to_csv(out / "loo_compare.csv")
        typer.echo(comp.to_string())
    except Exception as e:  # noqa: BLE001
        typer.echo(f"LOO comparison failed: {e}")
    return pd.concat(rows, ignore_index=True)


def main(
    k: int = typer.Option(3, help="matching radius to analyse"),
    bayes: bool = typer.Option(False, help="also fit the Bayesian hurdle log-normal"),
    n_sets: int = typer.Option(0, help="Bayesian: random subsample of this many sets (0 = all)"),
    draws: int = typer.Option(1000), tune: int = typer.Option(1000), chains: int = typer.Option(4),
    loglik: bool = typer.Option(True, help="store pointwise log-likelihood for LOO (memory heavy on full data)"),
    dev: bool = typer.Option(False),
    plot_only: bool = typer.Option(False, help="redraw the field figures from the saved tables, no refitting"),
) -> None:
    cfg = load_config()
    sfx = "_dev" if dev else ""
    out = cfg.derived / f"s07{sfx}"
    out.mkdir(parents=True, exist_ok=True)
    d = load(cfg, cfg.derived / f"s05{sfx}", k)
    typer.echo(f"k={k}: {len(d):,} rows, {d.set_id.nunique():,} sets, share cited {d.cited.mean():.3f}")

    if plot_only:
        freq = pd.read_csv(out / f"hurdle_frequentist_k{k}.csv")
        bf = pd.read_csv(out / f"hurdle_by_field_k{k}.csv")
        plot_by_field(bf, REPO_ROOT / "reports" / f"hurdle_by_field_k{k}.png", pooled=freq, d=d)
        plot_covariates_by_field(bf, REPO_ROOT / "reports" / f"covariates_by_field_k{k}.png")
        typer.echo("figures redrawn from saved tables")
        return

    freq = frequentist(d)
    freq.to_csv(out / f"hurdle_frequentist_k{k}.csv", index=False)
    typer.echo(freq.round(4).to_string(index=False))

    bf = by_field(d)
    bf.to_csv(out / f"hurdle_by_field_k{k}.csv", index=False)
    typer.echo(bf[bf.term == "phil"].round(3).to_string(index=False))
    plot_by_field(bf, REPO_ROOT / "reports" / f"hurdle_by_field_k{k}.png", pooled=freq, d=d)
    plot_covariates_by_field(bf, REPO_ROOT / "reports" / f"covariates_by_field_k{k}.png")

    if bayes:
        if n_sets:
            keep = pd.Series(sorted(d["set_id"].unique())).sample(n_sets, random_state=int(cfg.params["seed"]))
            d = d[d["set_id"].isin(keep)]
            typer.echo(f"Bayesian subsample: {len(d):,} rows, {n_sets} sets")
        summ = bayesian(d, out, draws, tune, chains, loglik)
        summ.to_csv(out / f"hurdle_bayesian_k{k}_n{n_sets or 'all'}.csv", index=False)
        typer.echo(summ.round(3).to_string(index=False))
