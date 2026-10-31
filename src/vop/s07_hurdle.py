"""Stage s07: hurdle log-normal models of citations.

Two processes: whether a paper gets cited at all (the hurdle) and, given it is cited, how many
citations it accumulates (log-normal). Two tiers:

  frequentist  set fixed effects in both parts: FE logit (pyfixest.feglm) on 1[cites > 0] and
               FE OLS on log(cites) among cited papers; covariates beyond the matching variables.
  bayesian     PyMC HurdleLogNormal mirroring brms::hurdle_lognormal: random intercepts per matched
               set in the hurdle, in mu, and in sigma (non-centred), same drivers in hurdle and mu.
               Sampled with nutpie (NUTS). PSIS-LOO via ArviZ, with and without the drivers, so the
               model comparison in the brms write-up can be reproduced. The field fits (journal_within,
               field_within) centre the philosopher indicator and covariates within sets.

Covariates (within-set deviations, so they are identified separately from the set effect):
  log_refs    log1p(referenced_works_count)   reference-list length, a proxy for text length / genre
  year_dev    publication_year - set mean     residual age difference inside the +-k window
  nauth_dev   n_authors - set mean            residual team-size difference
"""

from __future__ import annotations

from pathlib import Path

import gc
import json
import arviz as az
import numpy as np
import pandas as pd
import pyfixest as pf
import pymc as pm
import pytensor.tensor as pt
import typer

from vop.config import REPO_ROOT, Config, connect, load_config

BASE_DRIVERS = ["phil", "log_refs", "year_dev", "nauth_dev"]
PRESTIGE_DRIVERS = ["prestige_cnp", "no_prior_record", "log_prior_works_max"]  # all authors: symmetric between focal and control
DRIVERS = list(BASE_DRIVERS)  # extended at runtime when s08 output exists


def load(cfg: Config, s05_dir: Path, k: int) -> pd.DataFrame:
    con = connect(cfg)
    sets_path = s05_dir / "matched_sets.parquet"
    cols = set(con.execute(f"DESCRIBE SELECT * FROM read_parquet('{sets_path}')").fetchdf().column_name)
    meta = ", s.has_abstract, s.n_pages" if "has_abstract" in cols else ""
    d = con.execute(f"""
        WITH s AS (SELECT * FROM read_parquet('{s05_dir / "matched_sets.parquet"}') WHERE k = {k}),
             f AS (SELECT set_id, field AS set_field, domain AS set_domain FROM s WHERE is_focal)
        SELECT s.set_id, s.work_id, s.is_focal, s.source_id, s.publication_year, s.n_authors, s.cited_by_count,
               coalesce(s.referenced_works_count, 0) AS n_refs, f.set_field, f.set_domain{meta}
        FROM s JOIN f USING (set_id)
        ORDER BY s.set_id, s.work_id  -- fixed order: LOO comparisons pair fits row by row
    """).fetchdf()
    d["phil"] = d["is_focal"].astype(int)
    d["cited"] = (d["cited_by_count"] > 0).astype(int)
    d["log_cites"] = np.log(d["cited_by_count"].where(d["cited_by_count"] > 0))
    d["log_refs"] = np.log1p(d["n_refs"])
    for col, new in [("publication_year", "year_dev"), ("n_authors", "nauth_dev")]:
        d[new] = d[col] - d.groupby("set_id")[col].transform("mean")
    if meta:
        d["has_abstract"] = d["has_abstract"].fillna(False).astype(int)
        if "has_abstract" not in DRIVERS:
            DRIVERS.append("has_abstract")
    prestige = s05_dir.parent / "s08" / "paper_prestige.parquet"
    if prestige.exists():
        pr = con.execute(f"SELECT * FROM read_parquet('{prestige}')").fetchdf()
        d = d.merge(pr, on="work_id", how="left")
        # no scored earlier paper: flag it and centre the score, so the flag reads relative to an average record
        d["no_prior_record"] = d["prestige_cnp"].isna().astype(int)
        d["prestige_cnp"] = d["prestige_cnp"].fillna(d["prestige_cnp"].mean())
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
        extra = [(t, COVARIATE_LABELS[t][0].replace("Top author track record", "All fields: top author track record +10 pct. pts")
                  .replace("No earlier papers", "All fields: no earlier papers")
                  .replace("Abstract available", "All fields: abstract available")
                  .replace("Senior-author career length", "All fields: senior author prior works ×2.7"))
                 for t in PRESTIGE_DRIVERS + ["has_abstract"] if t in set(pooled.term)]
        for term, label in POOLED_ROWS + extra:
            hp = pooled[(pooled.model.str.startswith("hurdle")) & pooled.model.str.contains("covariates") & (pooled.term == term)]
            mp = pooled[(pooled.model.str.startswith("log-normal")) & pooled.model.str.contains("covariates") & (pooled.term == term)]
            if hp.empty or mp.empty:
                continue
            h, m = hp.iloc[0].copy(), mp.iloc[0].copy()
            if term == "prestige_cnp":  # plotted per 10 percentile points so it does not stretch the axis
                for r_ in (h, m):
                    r_["coef"], r_["ci_low"], r_["ci_high"] = r_["coef"] / 10, r_["ci_low"] / 10, r_["ci_high"] / 10
            h["ctrl_share_cited"], h["phil_share_cited"] = p0, p1
            pooled_rows.append((label, h, m, term == "phil"))
    all_rows = rows + pooled_rows
    n = len(all_rows)
    gap = 0.6 if pooled_rows else 0.0
    ys = [i + (gap if i >= len(rows) else 0.0) for i in range(n)]
    h_in = 0.42 * n + 2.4 + gap * 0.42
    fig, axes = plt.subplots(1, 2, figsize=(16, h_in), sharey=True)
    fig.subplots_adjust(top=1 - 1.35 / h_in, wspace=0.06, left=0.3, right=0.985, bottom=0.9 / h_in)
    panels = {"hurdle": ("Cited at least once", f"controls {ARROW} phil. (raw)"),
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
    "prestige_cnp": ("Top author track record", "0 → 1 mean percentile of earlier papers"),
    "no_prior_record": ("No earlier papers", "no author has a scored earlier paper"),
    "has_abstract": ("Abstract available", "abstract recorded vs none"),
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


def build_bayesian(d: pd.DataFrame, drivers: list[str]) -> pm.Model:
    set_idx, sets = pd.factorize(d["set_id"])
    y = d["cited_by_count"].to_numpy().astype(float)
    X = d[drivers].to_numpy()
    # standardise the continuous drivers only; keep phil as a 0/1 indicator so its coefficient is a log-odds /
    # log-ratio directly comparable with the frequentist FE estimates
    X = X.astype(float)
    for j in range(1, X.shape[1]):
        if set(np.unique(X[:, j])) <= {0.0, 1.0}:  # flags stay 0/1
            continue
        sd = X[:, j].std()
        typer.echo(f"  {drivers[j]}: mean {X[:, j].mean():.4f}, sd {sd:.4f} (coefficient is per sd)")
        X[:, j] = (X[:, j] - X[:, j].mean()) / (sd if sd > 0 else 1)
    with pm.Model(coords={"set": sets, "driver": drivers}) as model:
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


NO_FIELD = "No field assigned"


FIELD_FITS = {  # fit label -> (column grouping the sets, wording for the figure)
    "field_within": ("set_field", "field of the philosopher paper"),
    "journal_within": ("j_field", "main field of the journal"),
}
BINARY_DRIVERS = {"has_abstract", "no_prior_record"}
BETWEEN_DRIVERS = ["log_refs", "has_abstract", "prestige_cnp", "no_prior_record", "log_prior_works_max"]


def attach_journal_field(cfg: Config, d: pd.DataFrame) -> pd.DataFrame:
    """Add each set's journal field: the most common field among all articles in that journal, all years."""
    path = cfg.derived / "s07" / "journal_field.parquet"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        connect(cfg).execute(f"""
            COPY (
                WITH c AS (SELECT source_id, field, count(*) AS n
                           FROM read_parquet('{cfg.derived / "s02" / "works_slim"}/**/*.parquet') GROUP BY 1, 2),
                     r AS (SELECT *, sum(n) OVER (PARTITION BY source_id) AS n_articles,
                                  row_number() OVER (PARTITION BY source_id ORDER BY n DESC, field NULLS LAST) AS rn FROM c)
                SELECT source_id, field AS j_field, n * 1.0 / n_articles AS j_share, n_articles FROM r WHERE rn = 1
            ) TO '{path}' (FORMAT parquet)""")
    jf = pd.read_parquet(path)
    return d.merge(jf, on="source_id", how="left", validate="many_to_one")


def _within_between(d: pd.DataFrame, cols: list[str], between: list[str], mask: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deviations from the set mean and standardised set means, computed on the rows in `mask`; 0 elsewhere."""
    sub = d.loc[mask, cols].astype(float)
    means = sub.groupby(d.loc[mask, "set_id"]).transform("mean")
    within = (sub - means).reindex(d.index).fillna(0.0)
    b = means[between]
    b = ((b - b.mean()) / b.std()).reindex(d.index).fillna(0.0)
    return within, b


def _field_effects(name: str, mu, tau, n_sets: np.ndarray, min_centred: int = 300):
    """Per-field effects: centred where the data are informative, non-centred where they are not.

    Fields with many sets mix badly when non-centred, because the data pin down each effect. Fields with few
    sets give a funnel when centred and the spread is small. Mixing the two avoids both problems.
    """
    big = np.flatnonzero(n_sets >= min_centred)
    small = np.flatnonzero(n_sets < min_centred)
    b = pt.zeros(len(n_sets))
    if len(big):
        b = pt.set_subtensor(b[big], pm.Normal(f"{name}_big", mu, tau, shape=len(big)))
    if len(small):
        b = pt.set_subtensor(b[small], mu + tau * pm.Normal(f"{name}_z", 0, 1, shape=len(small)))
    return pm.Deterministic(name, b, dims="field")


def build_field_model(d: pd.DataFrame, scales_path: Path, group: str) -> pm.Model:
    """Hurdle log-normal with a philosopher effect per field, partially pooled across fields.

    logit P(cited)     = a_h + b_h_field[field] * phil_w + X_w * b_h + X_b * bb_h + set effect
    mean of log(cites) = a_m + b_m_field[field] * phil_w + X_w * b_m + X_b * bb_m + set effect
    log sigma          = a_s + set effect

    Set effects are shrunk, so they do not absorb field baselines or between-set differences. Without
    centring, those leak into the philosopher and covariate coefficients. `_w` terms are deviations from
    the set mean and `X_b` holds the set means, so `b_*_field` and `b_*` rest on within-set contrasts only.
    The citation part only sees cited papers, so it is centred on the cited papers of each set.
    """
    d = d.reset_index(drop=True)
    set_idx, sets = pd.factorize(d["set_id"])
    field_idx, fields = pd.factorize(d[group], sort=True)
    n_sets = d.groupby(field_idx)["set_id"].nunique().to_numpy()
    y = d["cited_by_count"].to_numpy().astype(float)
    drivers = [x for x in DRIVERS if x != "phil"]
    between = [x for x in BETWEEN_DRIVERS if x in drivers]
    cols = ["phil"] + drivers
    w_h, x_b_h = _within_between(d, cols, between, pd.Series(True, index=d.index))
    w_m, x_b_m = _within_between(d, cols, between, d["cited_by_count"] > 0)
    # one scale for both parts, so the plot converts both with the same divisor; binary drivers stay in raw units
    scales = {c: {"mean": 0.0, "sd": 1.0 if c in BINARY_DRIVERS else float(w_h[c].std())} for c in drivers}
    scales_path.write_text(json.dumps(scales, indent=1))
    sd = np.array([scales[c]["sd"] for c in drivers])
    ph_h, ph_m = w_h["phil"].to_numpy(), w_m["phil"].to_numpy()
    X_h, X_m = w_h[drivers].to_numpy() / sd, w_m[drivers].to_numpy() / sd
    B_h, B_m = x_b_h.to_numpy(), x_b_m.to_numpy()
    coords = {"set": sets, "driver": drivers, "field": list(fields), "between": between}
    with pm.Model(coords=coords) as model:
        a_h = pm.Normal("a_h", 0, 1.5)
        mu_h = pm.Normal("mu_h", 0, 1)
        tau_h = pm.HalfNormal("tau_h", 0.5)
        b_h_field = _field_effects("b_h_field", mu_h, tau_h, n_sets)
        b_h = pm.Normal("b_h", 0, 1, dims="driver")
        bb_h = pm.Normal("b_h_between", 0, 1, dims="between")
        s_h = pm.HalfNormal("sd_set_h", 1)
        z_h = pm.Normal("z_h", 0, 1, dims="set")
        psi = pm.math.sigmoid(a_h + b_h_field[field_idx] * ph_h + X_h @ b_h + B_h @ bb_h + s_h * z_h[set_idx])
        a_m = pm.Normal("a_m", 1, 1.5)
        mu_m = pm.Normal("mu_m", 0, 1)
        tau_m = pm.HalfNormal("tau_m", 0.5)
        b_m_field = _field_effects("b_m_field", mu_m, tau_m, n_sets)
        b_m = pm.Normal("b_m", 0, 1, dims="driver")
        bb_m = pm.Normal("b_m_between", 0, 1, dims="between")
        s_m = pm.HalfNormal("sd_set_m", 1)
        z_m = pm.Normal("z_m", 0, 1, dims="set")
        mu = a_m + b_m_field[field_idx] * ph_m + X_m @ b_m + B_m @ bb_m + s_m * z_m[set_idx]
        a_s = pm.Normal("a_s", 0, 1)
        s_s = pm.HalfNormal("sd_set_s", 0.5)
        z_s = pm.Normal("z_s", 0, 1, dims="set")
        sigma = pm.math.exp(a_s + s_s * z_s[set_idx])
        pm.HurdleLogNormal("y", psi=psi, mu=mu, sigma=sigma, observed=y)
    return model


def bayesian(d: pd.DataFrame, out: Path, draws: int, tune: int, chains: int, loglik: bool = True,
             fit: tuple[str, ...] = ("with_track",)) -> pd.DataFrame:
    """Fit the named models. LOO is left to `vop s07-loo`, which works in chunks from the saved files."""
    specs = {"phil_only": ["phil"], "with_drivers": BASE_DRIVERS, "with_track": list(DRIVERS)}
    rows = []
    all_rows = d
    for label in fit:
        d = all_rows
        if label in FIELD_FITS:
            group = FIELD_FITS[label][0]
            d = d[d[group].notna()]  # no field: sparse records, and for the paper field an artefact of selection
            n_ = d.set_id.nunique()
            model = build_field_model(d, out / f"hurdle_{label}_n{n_}.scales.json", group)
            names = ["a_h", "mu_h", "tau_h", "b_h_field", "b_h", "b_h_between", "a_m", "mu_m", "tau_m", "b_m_field", "b_m",
                     "b_m_between", "a_s", "sd_set_h", "sd_set_m", "sd_set_s"]
        else:
            model = build_bayesian(d, specs[label])
            names = ["a_h", "b_h", "a_m", "b_m", "a_s", "sd_set_h", "sd_set_m", "sd_set_s"]
        with model:
            idata = pm.sample(draws=draws, tune=tune, chains=chains, nuts_sampler="nutpie",
                              idata_kwargs={"log_likelihood": loglik}, progressbar=False)
        idata.to_netcdf(out / f"hurdle_{label}_n{d.set_id.nunique()}.nc")
        typer.echo(f"[{label}] sampled {d.set_id.nunique()} sets", err=False)
        summ = az.summary(idata, var_names=names)
        summ.insert(0, "model", label)
        typer.echo(f"[{label}] divergences: {int(idata.sample_stats['diverging'].sum())}")
        typer.echo(summ.round(3).to_string())
        rows.append(summ.reset_index().rename(columns={"index": "param"}))
        d[["set_id", "work_id"]].to_parquet(out / f"hurdle_{label}_n{d.set_id.nunique()}.rows.parquet", index=False)
        del idata, model, summ
        gc.collect()
    return pd.concat(rows, ignore_index=True)


def main(
    k: int = typer.Option(3, help="matching radius to analyse"),
    bayes: bool = typer.Option(False, help="also fit the Bayesian hurdle log-normal"),
    n_sets: int = typer.Option(0, help="Bayesian: random subsample of this many sets (0 = all)"),
    draws: int = typer.Option(1000), tune: int = typer.Option(1000), chains: int = typer.Option(4),
    loglik: bool = typer.Option(True, help="store pointwise log-likelihood for LOO (memory heavy on full data)"),
    dev: bool = typer.Option(False),
    plot_only: bool = typer.Option(False, help="redraw the field figures from the saved tables, no refitting"),
    bayes_only: bool = typer.Option(False, help="skip the fixed-effects models and figures"),
    fit: str = typer.Option("with_track", help="comma list: phil_only,with_drivers,with_track,field_within,journal_within"),
) -> None:
    cfg = load_config()
    sfx = "_dev" if dev else ""
    out = cfg.derived / f"s07{sfx}"
    out.mkdir(parents=True, exist_ok=True)
    cfg.reports.mkdir(parents=True, exist_ok=True)
    d = load(cfg, cfg.derived / f"s05{sfx}", k)
    typer.echo(f"k={k}: {len(d):,} rows, {d.set_id.nunique():,} sets, share cited {d.cited.mean():.3f}")
    if "journal_within" in fit.split(","):
        d = attach_journal_field(cfg, d)

    if plot_only:
        freq = pd.read_csv(out / f"hurdle_frequentist_k{k}.csv")
        bf = pd.read_csv(out / f"hurdle_by_field_k{k}.csv")
        plot_by_field(bf, cfg.reports / f"hurdle_by_field_k{k}.png", pooled=freq, d=d)
        plot_covariates_by_field(bf, cfg.reports / f"covariates_by_field_k{k}.png")
        typer.echo("figures redrawn from saved tables")
        return

    if bayes_only:
        if n_sets:
            keep = pd.Series(sorted(d["set_id"].unique())).sample(n_sets, random_state=int(cfg.params["seed"]))
            d = d[d["set_id"].isin(keep)]
        summ = bayesian(d, out, draws, tune, chains, loglik, tuple(fit.split(",")))
        summ.to_csv(out / f"hurdle_bayesian_k{k}_{fit.replace(',', '_')}.csv", index=False)
        return

    freq = frequentist(d)
    freq.to_csv(out / f"hurdle_frequentist_k{k}.csv", index=False)
    typer.echo(freq.round(4).to_string(index=False))

    bf = by_field(d)
    bf.to_csv(out / f"hurdle_by_field_k{k}.csv", index=False)
    typer.echo(bf[bf.term == "phil"].round(3).to_string(index=False))
    plot_by_field(bf, cfg.reports / f"hurdle_by_field_k{k}.png", pooled=freq, d=d)
    plot_covariates_by_field(bf, cfg.reports / f"covariates_by_field_k{k}.png")

    if bayes:
        if n_sets:
            keep = pd.Series(sorted(d["set_id"].unique())).sample(n_sets, random_state=int(cfg.params["seed"]))
            d = d[d["set_id"].isin(keep)]
            typer.echo(f"Bayesian subsample: {len(d):,} rows, {n_sets} sets")
        summ = bayesian(d, out, draws, tune, chains, loglik, tuple(fit.split(",")))
        summ.to_csv(out / f"hurdle_bayesian_k{k}_n{n_sets or 'all'}.csv", index=False)
        typer.echo(summ.round(3).to_string(index=False))
