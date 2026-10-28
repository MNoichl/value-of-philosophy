"""Parse the saved PhilPapers "Journal archive" page (All philosophy journals) into a CSV.

Each entry on the page looks like
  <a href='/pub/46'>American Philosophical Quarterly</a>&nbsp;&nbsp;<span style='font-size:smaller'>
      ISSN 00030481 &nbsp; 2202 articles between 1964 and 2026</span><br>
with the ISSN part optional. Only the stdlib is used so it also runs on a laptop without the venv.
"""

from __future__ import annotations

import csv
import html
import re
import sys
from pathlib import Path

from vop.issn import hyphenate, issn_checksum_ok, normalize_issn

ENTRY = re.compile(
    r"<a href='/pub/(?P<pub_id>\d+)'>(?P<name>.*?)</a>&nbsp;&nbsp;"
    r"<span style='font-size:smaller'>(?P<meta>.*?)</span><br>",
    re.S,
)
ISSN_RE = re.compile(r"ISSN\s+(?P<issn>[0-9Xx\- ]{8,10})")
ARTICLES_RE = re.compile(r"(?P<n>\d+) articles between (?P<first>.*?) and (?P<last>.*?)\s*$")
TOTAL_RE = re.compile(r"<b>(\d+) journals on this list\.</b>")


def parse(html_path: Path) -> list[dict]:
    text = html_path.read_text(encoding="utf-8")
    rows = []
    for m in ENTRY.finditer(text):
        name = html.unescape(re.sub(r"\s+", " ", m["name"])).strip()
        meta = html.unescape(m["meta"]).replace("\xa0", " ")
        issn_m = ISSN_RE.search(meta)
        issn_raw = issn_m["issn"].strip() if issn_m else ""
        issn8 = normalize_issn(issn_raw) if issn_raw else None
        art = ARTICLES_RE.search(meta.strip())
        rows.append(
            {
                "pub_id": int(m["pub_id"]),
                "name": name,
                "issn_raw": issn_raw,
                "issn": hyphenate(issn8) if issn8 else "",
                "issn_checksum_ok": bool(issn8) and issn_checksum_ok(issn8),
                "n_articles": int(art["n"]) if art else None,
                "first_year": art["first"].strip() if art else "",
                "last_year": art["last"].strip() if art else "",
                "url": f"https://philpapers.org/pub/{m['pub_id']}",
            }
        )
    total = TOTAL_RE.search(text)
    if total and int(total[1]) != len(rows):
        raise RuntimeError(f"Page says {total[1]} journals but parsed {len(rows)}; the format changed.")
    return rows


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    src = Path(argv[0]) if argv else Path("inputs/raw/philpapers/journals.html")
    dst = Path(argv[1]) if len(argv) > 1 else Path("inputs/philpapers_journals.csv")
    rows = parse(src)
    with dst.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    n_issn = sum(1 for r in rows if r["issn"])
    n_bad = sum(1 for r in rows if r["issn"] and not r["issn_checksum_ok"])
    n_dup = len(rows) - len({r["issn"] for r in rows if r["issn"]}) - (len(rows) - n_issn)
    print(f"{len(rows)} journals parsed -> {dst}")
    print(f"  with ISSN: {n_issn}   without ISSN: {len(rows) - n_issn}")
    print(f"  ISSN failing checksum: {n_bad}   ISSNs shared by >1 entry: {n_dup}")


if __name__ == "__main__":
    main()
