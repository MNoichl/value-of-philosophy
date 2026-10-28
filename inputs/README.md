Small, committed inputs.

- `philpapers_journals_with_issn.csv` — the PhilPapers journal list with ISSNs that defines "philosophy journals".
  Rebuilt 2026-10-28 from `raw/philpapers/journals.html` (the "All philosophy journals" archive page,
  saved from a browser) with `python -m vop.philpapers`. 1698 journals, 1467 with an ISSN.
  Column `issn_checksum_ok` flags ISSNs that fail the ISO check digit (PhilPapers typos);
  `data/derived/s01/philpapers_match_report.csv` shows which entries matched an OpenAlex source.
