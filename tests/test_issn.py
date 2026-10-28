from vop.issn import hyphenate, issn_checksum_ok, normalize_issn


def test_normalize():
    assert normalize_issn("0003-0481") == "00030481"
    assert normalize_issn(" 0003-048x ") == "0003048X"
    assert normalize_issn("12345") is None
    assert normalize_issn(None) is None


def test_checksum_known_good():
    # American Philosophical Quarterly, Mind, Philosophical Review
    for good in ("00030481", "00264423", "00318108"):
        assert issn_checksum_ok(good), good


def test_checksum_catches_typo_and_transposition():
    assert not issn_checksum_ok("00030482")  # last digit wrong
    assert not issn_checksum_ok("00300481")  # digits transposed


def test_hyphenate():
    assert hyphenate("00030481") == "0003-0481"
