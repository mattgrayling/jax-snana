"""Fixture trimming and compression preserve the recorded scientific data."""

import numpy as np
import pytest
from astropy.io import fits

from fixtures.regenerate import trim_replay_photometry


def test_trim_replay_preserves_selected_events(tmp_path):
    dump = tmp_path / 'DUMP'
    dump.write_text('VARNAMES: CID GENZ\nSN: 30 0.3\nSN: 30 0.3\n'
                    'SN: 10 0.1\nSN: 20 0.2\n')
    original_dump = dump.read_bytes()
    head = fits.BinTableHDU.from_columns([
        fits.Column(name='SNID', format='8A', array=['10', '20', '30']),
        fits.Column(name='PTROBS_MIN', format='J', array=[1, 4, 7]),
        fits.Column(name='PTROBS_MAX', format='J', array=[2, 5, 8]),
        fits.Column(name='REDSHIFT_HELIO', format='D', array=[.1, .2, .3]),
    ])
    phot = fits.BinTableHDU.from_columns([
        fits.Column(name='MJD', format='D', array=[1., 2., -777., 3., 4., -777., 5., 6., -777.]),
        fits.Column(name='FLUXCAL', format='E', array=np.arange(9)),
    ])
    head_path, phot_path = tmp_path / 'HEAD.FITS', tmp_path / 'PHOT.FITS'
    head.writeto(head_path)
    phot.writeto(phot_path)
    trim_replay_photometry(tmp_path, 2)
    with fits.open(head_path) as h, fits.open(phot_path) as p:
        np.testing.assert_array_equal(h[1].data['SNID'], ['10', '30'])
        np.testing.assert_array_equal(h[1].data['REDSHIFT_HELIO'], [.1, .3])
        np.testing.assert_array_equal(h[1].data['PTROBS_MIN'], [1, 4])
        np.testing.assert_array_equal(h[1].data['PTROBS_MAX'], [2, 5])
        np.testing.assert_array_equal(p[1].data, phot.data[[0, 1, 2, 6, 7, 8]])
    assert dump.read_bytes() == original_dump
    once = head_path.read_bytes(), phot_path.read_bytes()
    trim_replay_photometry(tmp_path, 2)
    assert (head_path.read_bytes(), phot_path.read_bytes()) == once


def test_trim_replay_requires_event_markers(tmp_path):
    (tmp_path / 'DUMP').write_text('VARNAMES: CID\nSN: 1\n')
    fits.BinTableHDU.from_columns([
        fits.Column(name='SNID', format='8A', array=['1']),
        fits.Column(name='PTROBS_MIN', format='J', array=[1]),
        fits.Column(name='PTROBS_MAX', format='J', array=[1]),
    ]).writeto(tmp_path / 'HEAD.FITS')
    fits.BinTableHDU.from_columns([
        fits.Column(name='MJD', format='D', array=[1., 2.]),
    ]).writeto(tmp_path / 'PHOT.FITS')
    with pytest.raises(ValueError, match='missing end marker'):
        trim_replay_photometry(tmp_path, 1)


@pytest.mark.parametrize('n_values', [100, 300_000])
def test_gzip_fixture_preserves_bytes(tmp_path, n_values):
    import gzip
    from fixtures.regenerate import gzip_fixture

    path = tmp_path / 'sample.FITS'
    fits.PrimaryHDU(np.zeros(n_values, dtype=np.float32)).writeto(path)
    original = path.read_bytes()
    stored = gzip_fixture(path)
    assert stored == tmp_path / 'sample.FITS.gz'
    assert not path.exists()
    assert gzip.decompress(stored.read_bytes()) == original
    first = stored.read_bytes()
    path.write_bytes(original)
    assert gzip_fixture(path).read_bytes() == first


@pytest.mark.parametrize("row_prefix", ["SN:", "ROW:", "OBS:"])
def test_compressed_table_comparison(tmp_path, row_prefix):
    from fixtures.regenerate import _diff, gzip_fixture

    plain = tmp_path / "live.TEXT"
    bundled = tmp_path / "bundled.TEXT"
    content = f"VARNAMES: CID VALUE\n{row_prefix} 1 2.5\n"
    plain.write_text(content)
    bundled.write_text(content)
    compressed = gzip_fixture(bundled)
    assert _diff(plain, compressed) is None
    assert _diff(compressed, plain) is None
    plain.write_text(content.replace("2.5", "3.5"))
    assert "VALUE" in _diff(plain, compressed)
    assert "VALUE" in _diff(compressed, plain)


def test_readme_comparison_ignores_run_location_and_time(tmp_path):
    from fixtures.regenerate import _diff_readme

    bundled = tmp_path / "bundled.README"
    live = tmp_path / "live.README"
    bundled.write_text("TIME_START: yesterday\nCWD: first\nCPU_MINUTES: 1\n"
                       "NGENLC_WRITE: 100 # (50/sec, LC passing trigger)\nOMEGA_MATTER: 0.3\n")
    live.write_text("TIME_START: today\nCWD: second\nCPU_MINUTES: 2\n"
                    "NGENLC_WRITE: 100 # (40/sec, LC passing trigger)\nOMEGA_MATTER: 0.3\n")
    assert _diff_readme(live, bundled) is None
    live.write_text(live.read_text().replace("0.3", "0.4"))
    assert "OMEGA_MATTER" in _diff_readme(live, bundled)
    live.write_text(bundled.read_text().replace("NGENLC_WRITE: 100", "NGENLC_WRITE: 99"))
    assert "NGENLC_WRITE" in _diff_readme(live, bundled)


def test_asset_checksums():
    import hashlib
    from fixtures.regenerate import DATA_ROOT

    for line in (DATA_ROOT / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        assert hashlib.sha256((DATA_ROOT / name).read_bytes()).hexdigest() == expected, name
