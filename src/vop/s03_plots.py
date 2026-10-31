"""Who counts as a philosopher: distribution figures and threshold tables from the full s03 output.

Nothing here applies a threshold to the data. The current rule appears only as a reference line, and the
grids show how many authors and focal papers each rule would give.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import typer

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

from vop.config import connect, load_config  # noqa: E402
from vop.plotting import CTRL, INK, PHIL, title, use_opinionated  # noqa: E402

MIN_WORKS = [1, 2, 3, 5, 10, 20, 50]
MIN_SHARE = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
AQUA = "#1baf7a"
TICKS = [0, 1, 2, 3]
LABELS = ["1", "10", "100", "1,000"]


def _grid(ax, g: pd.DataFrame, label: str) -> None:
    m = g.to_numpy(dtype=float)
    ax.imshow(np.maximum(m, 1), cmap="Blues", norm=LogNorm(vmin=max(1, np.nanmin(m[m > 0])), vmax=np.nanmax(m)), aspect="auto")
    thresh = np.sqrt(max(1, np.nanmin(m[m > 0])) * np.nanmax(m))
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, f"{int(m[i, j]):,}", ha="center", va="center", fontsize=8.5, color="white" if m[i, j] > thresh else "#0b0b0b")
    ax.set_xticks(range(m.shape[1]), [f">{c:.0%}" for c in g.columns], fontsize=10)
    ax.set_yticks(range(m.shape[0]), [f"≥{i}" for i in g.index], fontsize=10)
    ax.set_xlabel("share of articles in philosophy sources", fontsize=11)
    ax.set_ylabel("minimum articles", fontsize=11)
    ax.set_title(label, loc="left", fontsize=13)
    ax.grid(False)


def _hex(ax, x, y, xl, yl, label) -> None:
    hb = ax.hexbin(x, y, gridsize=55, cmap="Blues", norm=LogNorm(), mincnt=1, linewidths=0.1)
    ax.set_xlabel(xl, fontsize=11)
    ax.set_ylabel(yl, fontsize=11)
    ax.set_title(label, loc="left", fontsize=13)
    ax.grid(False)
    plt.colorbar(hb, ax=ax, fraction=0.04, pad=0.02).ax.tick_params(labelsize=8)


def main() -> None:
    cfg = load_config()
    s03 = cfg.derived / "s03"
    con = connect(cfg)
    s = con.execute(f"SELECT author_id, works_total, works_phil, share_phil FROM read_parquet('{s03 / 'author_shares.parquet'}')").fetchdf()
    s["works_out"] = s.works_total - s.works_phil
    grid = pd.DataFrame({t: [int(((s.works_total >= w) & (s.share_phil > t)).sum()) for w in MIN_WORKS] for t in MIN_SHARE}, index=MIN_WORKS)
    grid.to_csv(s03 / "threshold_grid_authors.csv")

    papers = None
    if (s03 / "candidate_author_works").exists():
        y0, y1 = int(cfg.params["year_min"]), int(cfg.params["year_max"])
        rows = {}
        for w in MIN_WORKS:
            sel = ", ".join(f"sum((m > {t})::INT) AS t{i}" for i, t in enumerate(MIN_SHARE))
            r = con.execute(f"""
                WITH j AS (SELECT c.work_id, max(a.share_phil) AS m
                           FROM read_parquet('{s03 / "candidate_author_works"}/*.parquet') c
                           JOIN read_parquet('{s03 / "author_shares.parquet"}') a USING (author_id)
                           WHERE NOT c.in_phil AND c.publication_year BETWEEN {y0} AND {y1} AND a.works_total >= {w}
                           GROUP BY c.work_id)
                SELECT {sel} FROM j""").fetchone()
            rows[w] = list(r)
        papers = pd.DataFrame(rows, index=MIN_SHARE).T.fillna(0)
        papers.to_csv(s03 / "threshold_grid_papers.csv")

    use_opinionated()
    fig, ax = plt.subplots(2, 3, figsize=(19, 11.5))
    fig.subplots_adjust(top=0.875, bottom=0.07, left=0.05, right=0.985, wspace=0.26, hspace=0.3)
    a = ax[0, 0]
    bins = np.linspace(0, 1, 41)
    for w, c in [(1, CTRL), (5, PHIL), (10, AQUA)]:
        a.hist(s.share_phil[s.works_total >= w], bins=bins, histtype="step", color=c, linewidth=2, label=f"at least {w} article{'s' if w > 1 else ''}")
    a.set_yscale("log")
    a.axvline(0.5, color=INK, linewidth=1, linestyle="--")
    a.set_xlabel("share of articles in philosophy sources", fontsize=11)
    a.set_ylabel("authors (log scale)", fontsize=11)
    a.set_title("Share of articles in philosophy", loc="left", fontsize=13)
    a.legend(frameon=False, fontsize=10, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.09))

    lt, lp, lo = np.log10(s.works_total + 1), np.log10(s.works_phil + 1), np.log10(s.works_out + 1)
    _hex(ax[0, 1], lp, lo, "articles in philosophy sources + 1", "articles elsewhere + 1", "Articles in and out")
    ax[0, 1].plot([0, 3], [0, 3], color=INK, linewidth=1, linestyle="--")
    _hex(ax[0, 2], lt, s.share_phil, "all articles + 1", "share in philosophy", "Output against share")
    ax[0, 2].axhline(0.5, color=INK, linewidth=1, linestyle="--")
    for b in (ax[0, 1], ax[0, 2]):
        b.set_xticks(TICKS, LABELS, fontsize=10)
    ax[0, 1].set_yticks(TICKS, LABELS, fontsize=10)
    a = ax[1, 0]
    c5 = s[s.works_total >= 5].sort_values("share_phil")
    a.plot(c5.share_phil, 1 - np.arange(len(c5)) / len(c5), color=CTRL, linewidth=2)
    a.axvline(0.5, color=INK, linewidth=1, linestyle="--")
    a.set_xlabel("share of articles in philosophy sources", fontsize=11)
    a.set_ylabel("share of authors at or above this value", fontsize=11)
    a.set_title("Authors with at least 5 articles", loc="left", fontsize=13)
    _grid(ax[1, 1], grid, "Authors each rule would count")
    if papers is not None:
        _grid(ax[1, 2], papers, "Their articles outside philosophy, in the study years")
    else:
        ax[1, 2].axis("off")
    title(fig, "Who counts as a philosopher",
          f"All {len(s):,} authors with at least one article in a philosophy source. Dashed lines: current 50% rule, or equal counts in and out. Colour: authors, log scale.")
    cfg.reports.mkdir(parents=True, exist_ok=True)
    out = cfg.reports / "philosopher_distribution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    typer.echo(f"figure -> {out}")
    typer.echo(grid.to_string())
    if papers is not None:
        typer.echo(papers.astype(int).to_string())
    base = cfg.data_root / "derived" / "s03" / "philosophers.parquet"
    this = s03 / "philosophers.parquet"
    if cfg.variant and base.exists():
        typer.echo(con.execute(f"""SELECT (SELECT count(*) FROM read_parquet('{base}')) AS all_sources,
            (SELECT count(*) FROM read_parquet('{this}')) AS this_variant,
            (SELECT count(*) FROM read_parquet('{base}') b SEMI JOIN read_parquet('{this}') t USING (author_id)) AS both_rules
            """).fetchdf().to_string(index=False))
