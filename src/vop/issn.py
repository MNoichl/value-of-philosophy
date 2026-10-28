"""ISSN helpers: normalisation and checksum validation.

PhilPapers lists ISSNs by hand and occasionally gets a digit wrong. The checksum catches
single-digit typos and most transpositions, so an invalid ISSN is flagged rather than silently
matching nothing in OpenAlex.
"""

from __future__ import annotations

import re

_NON_ISSN = re.compile(r"[^0-9X]")


def normalize_issn(raw: str | None) -> str | None:
    """Return the ISSN as 8 characters without hyphen, upper-case X, or None if it is not 8 chars."""
    if raw is None:
        return None
    s = _NON_ISSN.sub("", str(raw).strip().upper())
    return s if len(s) == 8 else None


def issn_checksum_ok(issn8: str) -> bool:
    """Validate the ISSN check digit (ISO 3297: weights 8..2, mod 11, X = 10)."""
    if not isinstance(issn8, str) or len(issn8) != 8 or not re.fullmatch(r"[0-9]{7}[0-9X]", issn8):
        return False
    total = sum(int(c) * w for c, w in zip(issn8[:7], range(8, 1, -1)))
    check = (11 - total % 11) % 11
    expected = "X" if check == 10 else str(check)
    return issn8[7] == expected


def hyphenate(issn8: str) -> str:
    return f"{issn8[:4]}-{issn8[4:]}"
