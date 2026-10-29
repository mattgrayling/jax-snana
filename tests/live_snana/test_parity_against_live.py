"""Live-SNANA parity check.

Regenerates each registered fixture in a temp directory by actually running
``snlc_sim.exe``, then byte-compares the live DUMP / HEAD.FITS / PHOT.FITS
against the bundled copies. Catches semantic drift between the SNANA version
the bundle was captured against and the version currently installed.

Skipped by default — requires ``--live-snana`` (or ``RUN_LIVE_SNANA=1``)
plus a working ``snlc_sim.exe`` on PATH and ``$JAX_TEST_FIXTURE_ROOT``-aware
data tree (the top-level conftest sets that env var to
``tests/fixtures/data``).
"""

import re
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

from jax_snlc_core.snana_files import open_text


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"

# Make tests/fixtures/regenerate.py importable as a module.
if str(FIXTURES_DIR) not in sys.path:
    sys.path.insert(0, str(FIXTURES_DIR))

import regenerate as _regen  # type: ignore  # noqa: E402


# SNANA records the model directory as an absolute path in the DUMP header, so
# a bundle captured from a different checkout of this repo differs on that line
# while every generated value matches. Anchoring on the tests/fixtures segment
# rather than on the key name covers any header path, and replaces only the
# prefix — a genuine model change still shows as a difference.
_CHECKOUT_PREFIX = re.compile(r"\S*(?=/tests/fixtures/)")


def _comparable(path: Path) -> str:
    """File contents with the checkout root reduced to a stable token."""
    with open_text(path) as stream:
        return _CHECKOUT_PREFIX.sub("<CHECKOUT>", stream.read())


def _verify_pinned_version() -> str:
    """Run snlc_sim.exe --version, return version string, fail loudly on mismatch."""
    pinned = (FIXTURES_DIR / "snana_version.txt").read_text().strip()
    actual = _regen.get_snana_version()
    assert actual == pinned, (
        f"SNANA version drift: pinned={pinned!r} actual={actual!r}. "
        "Audit and `python tests/fixtures/regenerate.py --all --force-version-bump` "
        "if the change is intentional."
    )
    return actual


@pytest.mark.parametrize("fixture", _regen.FIXTURES, ids=lambda fx: fx.name)
def test_live_matches_bundled(fixture):
    """Regenerate to a tmp dir and byte-compare DUMP/HEAD/PHOT vs the bundle."""
    if not shutil.which("snlc_sim.exe"):
        pytest.skip("snlc_sim.exe not on PATH")
    _verify_pinned_version()

    with tempfile.TemporaryDirectory(prefix=f"live_{fixture.name}_") as tmp:
        tmp_path = Path(tmp)
        _regen.regenerate_fixture(fixture, snana_version="<live>", out_dir=tmp_path)

        # Compare the text outputs: DUMP, plus whatever tables the fit stage
        # wrote. Both are deterministic given seed + input + SNANA version.
        # FITS output is binary and is compared column-by-column, with
        # tolerances, by the per-test parity suites instead.
        #
        # A bundle may hold tables this harness cannot regenerate — the BBC
        # bundles carry a FITRES.TEXT built by their own build.py — so tables
        # are compared where the live run produced one. DUMP is required,
        # since regeneration always writes it when the bundle has one.
        names = sorted(
            p.name.removesuffix(".gz") for p in fixture.root.iterdir()
            if p.name in ("DUMP", "DUMP.gz") or p.name.endswith((".TEXT", ".TEXT.gz"))
        )
        # Some fixtures record their reference events only in FITS tables.
        if not names:
            names = [name for name in ("HEAD.FITS", "PHOT.FITS", "SPEC.FITS")
                     if _regen._find(fixture.root, name) is not None]
        if "DUMP" in names:
            assert _regen._find(tmp_path, "DUMP") is not None, (
                f"live regen produced no DUMP for {fixture.name}"
            )
        compared = [n for n in names if _regen._find(tmp_path, n) is not None]
        assert compared, (
            f"{fixture.name}: nothing to compare — the bundle holds no DUMP "
            f"and no table the live run reproduces (bundled: {names})"
        )

        diffs = []
        for name in compared:
            live = _regen._find(tmp_path, name)
            bundled = _regen._find(fixture.root, name)
            if name.endswith(".FITS"):
                difference = _regen._diff_fits(live, bundled)
            else:
                difference = _comparable(live) != _comparable(bundled)
            if difference:
                diffs.append(name)
        assert not diffs, (
            f"{fixture.name}: {', '.join(diffs)} diverge(s) from the bundle "
            f"(compared {', '.join(compared)}). Either SNANA semantics "
            "changed or the bundle is stale."
        )
