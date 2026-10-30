"""Plot styling: use the `opinionated` package (MNoichl/opinionated).

Its __init__ currently fails on setuptools >= 80 (no pkg_resources) and matplotlib >= 3.10
(no matplotlib.style.core), so this shim applies the same stylesheet and bundled fonts directly
and falls back gracefully. `title()` mirrors opinionated.set_title_and_suptitle.
"""

from __future__ import annotations

import glob
from importlib.util import find_spec
from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager as fm
import matplotlib.pyplot as plt

ARROW = "\u2192"  # rendered with the fallback font (Roboto Condensed has no arrow glyph)
PHIL = "#eb6834"
CTRL = "#2a78d6"
INK = "#52514e"


def use_opinionated(style: str = "opinionated_rc") -> None:
    try:
        import opinionated  # noqa: F401  (works if the package has been fixed upstream)
        plt.style.use(style)
        return
    except Exception:  # noqa: BLE001
        pass
    spec = find_spec("opinionated")  # locate the package without importing it
    pkg = Path(spec.submodule_search_locations[0])
    for f in glob.glob(str(pkg / "fonts" / "*.ttf")):
        fm.fontManager.addfont(f)
    plt.style.use(str(pkg / "data" / f"{style}.mplstyle"))
    installed = {f.name for f in fm.fontManager.ttflist}
    wanted = [f for f in mpl.rcParams["font.sans-serif"] if f in installed]
    if wanted:
        mpl.rcParams["font.sans-serif"] = wanted + ["DejaVu Sans"]
    else:  # Roboto Condensed is downloaded at runtime by opinionated; use the bundled Fira Sans instead
        mpl.rcParams["font.sans-serif"] = [f for f in ["Fira Sans Condensed", "Fira Sans"] if f in installed] + ["DejaVu Sans"]


def title(fig, main: str, sub: str = "", x: float = 0.02) -> None:
    """Bold title and grey subtitle at the top-left; reserves ~1.1 inch of figure height."""
    h = fig.get_size_inches()[1]
    fig.text(x, 1 - 0.18 / h, main, fontsize=20, fontweight="bold", ha="left", va="top")
    if sub:
        fig.text(x, 1 - 0.62 / h, sub, fontsize=12, ha="left", va="top", color=INK)


def annotate(ax, x, y, text, **kw) -> None:
    """Text in the ink colour with an explicit font fallback so arrows render."""
    ax.text(x, y, text, va="center", color=INK, family=["Roboto Condensed", "DejaVu Sans"], **kw)
