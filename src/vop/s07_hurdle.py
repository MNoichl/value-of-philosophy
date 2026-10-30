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

from vop.config import Config, connect, load_config

DRIVERS = ["phil", "log_refs", "year_dev", "nauth_dev"]


def load(cfg: Config, s05_dir: Path, k: int) -> pd.DataFrame:
    con = connect(cfg)
    d = con.execute(f"""
        SELECT set_id, work_id, is_focal, publication_year, n_authors, cited_by_count,
               coalesce(referenced_works_count, 0) AS n_refs, domain
        FROM read_parquet('{s05_dir / "matched_sets.parquet"}') WHERE k = {k}
    """).fetchdf()
    d["phil"] = d["is_focal"].astype(int)
    d["cited"] = (d["cited_by_count"] > 0).astype(int)
    d["log_cites"] = np.log(d["cited_by_count"].where(d["cited_by_count"] > 0))
    d["log_refs"] = np.log1p(d["n_refs"])
    for col, new in [("publication_year", "year_dev"), ("n_authors", "nauth_dev")]:
        d[new] = d[col] - d.groupby("set_id")[col].transform("mean")
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


def build_bayesian(d: pd.DataFrame, with_drivers: bool) -> pm.Model:
    set_idx, sets = pd.factorize(d["set_id"])
    y = d["cited_by_count"].to_numpy().astype(float)
    X = d[DRIVERS].to_numpy() if with_drivers else d[["phil"]].to_numpy()
    X = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1)  # standardised drivers; phil stays 0/1-ish
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


def bayesian(d: pd.DataFrame, out: Path, draws: int, tune: int, chains: int) -> pd.DataFrame:
    rows, idatas = [], {}
    for label, with_drivers in [("phil_only", False), ("with_drivers", True)]:
        model = build_bayesian(d, with_drivers)
        with model:
            idata = pm.sample(draws=draws, tune=tune, chains=chains, nuts_sampler="nutpie",
                              idata_kwargs={"log_likelihood": True}, progressbar=False)
        idata.to_netcdf(out / f"hurdle_{label}.nc")
        idatas[label] = idata
        summ = az.summary(idata, var_names=["a_h", "b_h", "a_m", "b_m", "a_s", "sd_set_h", "sd_set_m", "sd_set_s"])
        summ.insert(0, "model", label)
        rows.append(summ.reset_index().rename(columns={"index": "param"}))
    comp = az.compare(idatas, ic="loo")
    comp.to_csv(out / "loo_compare.csv")
    typer.echo(comp.to_string())
    return pd.concat(rows, ignore_index=True)


def main(
    k: int = typer.Option(3, help="matching radius to analyse"),
    bayes: bool = typer.Option(False, help="also fit the Bayesian hurdle log-normal"),
    n_sets: int = typer.Option(0, help="Bayesian: random subsample of this many sets (0 = all)"),
    draws: int = typer.Option(1000), tune: int = typer.Option(1000), chains: int = typer.Option(4),
    dev: bool = typer.Option(False),
) -> None:
    cfg = load_config()
    sfx = "_dev" if dev else ""
    out = cfg.derived / f"s07{sfx}"
    out.mkdir(parents=True, exist_ok=True)
    d = load(cfg, cfg.derived / f"s05{sfx}", k)
    typer.echo(f"k={k}: {len(d):,} rows, {d.set_id.nunique():,} sets, share cited {d.cited.mean():.3f}")

    freq = frequentist(d)
    freq.to_csv(out / f"hurdle_frequentist_k{k}.csv", index=False)
    typer.echo(freq.round(4).to_string(index=False))

    if bayes:
        if n_sets:
            keep = pd.Series(sorted(d["set_id"].unique())).sample(n_sets, random_state=int(cfg.params["seed"]))
            d = d[d["set_id"].isin(keep)]
            typer.echo(f"Bayesian subsample: {len(d):,} rows, {n_sets} sets")
        summ = bayesian(d, out, draws, tune, chains)
        summ.to_csv(out / f"hurdle_bayesian_k{k}_n{n_sets or 'all'}.csv", index=False)
        typer.echo(summ.round(3).to_string(index=False))
