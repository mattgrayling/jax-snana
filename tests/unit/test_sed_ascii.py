"""Unit tests for the 3-column SED ASCII reader."""

DAYS = (-150.0, 500.0)

import gzip

import numpy as np
import pytest

from jax_snlc_core.io.sed_ascii import SED_WAVELENGTH_RANGE, read_sed_file
from jax_snlc_core.utils.error_handling import SNANAError


def _write_sed(path, phases, waves, flux_2d):
    """Write a SNANA-format 3-column SED file."""
    with open(path, "w") as f:
        for i, p in enumerate(phases):
            for j, w in enumerate(waves):
                f.write(f"{p:.3f} {w:.3f} {flux_2d[i, j]:.3e}\n")


def test_read_sed_file_basic_grid(tmp_path):
    phases = np.array([0.0, 5.0, 10.0])
    waves = np.array([4000.0, 5000.0, 6000.0])
    # Synthetic flux: peak (max) at phase=5 (index 1).
    flux = np.array([
        [1.0, 1.0, 1.0],
        [3.0, 4.0, 5.0],
        [2.0, 2.0, 2.0],
    ])
    p = tmp_path / "test.SED"
    _write_sed(p, phases, waves, flux)

    out_phases, out_waves, out_flux = read_sed_file(str(p), DAYS, SED_WAVELENGTH_RANGE, uniform_days=False)
    np.testing.assert_allclose(out_waves, waves)
    assert out_flux.shape == (3, 3)
    # The reader peak-shifts so the max-flux phase is at 0.
    peak_idx = int(np.argmax(np.sum(out_flux, axis=1)))
    assert abs(out_phases[peak_idx]) < 1e-6


def test_read_sed_file_skips_comments_and_blanks(tmp_path):
    p = tmp_path / "with_comments.SED"
    p.write_text("""
# top comment
! another comment
% and another
0.0 4000.0 1.0
0.0 5000.0 1.0

5.0 4000.0 2.0
5.0 5000.0 2.0
""")
    out_phases, out_waves, _ = read_sed_file(str(p), DAYS, SED_WAVELENGTH_RANGE, uniform_days=False)
    np.testing.assert_array_equal(out_waves, [4000.0, 5000.0])
    assert out_phases.size == 2


def test_read_sed_file_gzipped(tmp_path):
    """Gzipped .SED.gz inputs are handled transparently."""
    p = tmp_path / "test.SED.gz"
    with gzip.open(p, "wt") as f:
        f.write("0.0 4000.0 1.0\n")
        f.write("0.0 5000.0 1.0\n")
        f.write("5.0 4000.0 5.0\n")
        f.write("5.0 5000.0 5.0\n")
    phases, waves, flux = read_sed_file(str(p), DAYS, SED_WAVELENGTH_RANGE, uniform_days=False)
    assert phases.size == 2
    assert waves.size == 2
    assert flux.shape == (2, 2)


@pytest.mark.parametrize("bad_row, message", [
    ("incomplete row\n", "2 words, expected at least 3"),
    ("0.0 4500.0 abc\n", "line 2 '0.0 4500.0 abc' is not three numbers"),
])
def test_read_sed_file_aborts_on_a_bad_row(tmp_path, bad_row, message):
    """rd_sedFlux aborts on a row with fewer than three words
    (sntools.c:8505-8513); a row that does not parse aborts too, naming the
    file and the line."""
    p = tmp_path / "bad_row.SED"
    p.write_text("0.0 4000.0 1.0\n" + bad_row + "0.0 5000.0 1.0\n")
    with pytest.raises(SNANAError, match=message) as error:
        read_sed_file(str(p), DAYS, SED_WAVELENGTH_RANGE, uniform_days=False)
    assert str(p) in str(error.value)


def test_read_sed_file_keeps_only_rows_inside_the_ranges(tmp_path):
    """rd_sedFlux drops rows outside the day and wavelength ranges as it reads
    (sntools.c:8538-8542), so a template's stray lambda=0 row or a far-out
    epoch never reaches the model's grid."""
    p = tmp_path / "wide.SED"
    p.write_text(
        "-200.0 0.0 1.0\n-200.0 4000.0 1.0\n-200.0 5000.0 1.0\n"
        "0.0 0.0 1.0\n0.0 4000.0 3.0\n0.0 5000.0 3.0\n"
        "5.0 0.0 1.0\n5.0 4000.0 2.0\n5.0 5000.0 2.0\n"
    )
    phases, waves, flux = read_sed_file(str(p), DAYS, SED_WAVELENGTH_RANGE,
                                        t0shift_peakmag=False, uniform_days=False)
    np.testing.assert_array_equal(phases, [0.0, 5.0])
    np.testing.assert_array_equal(waves, [4000.0, 5000.0])
    assert flux.shape == (2, 2)
    with pytest.raises(SNANAError, match="inside days"):
        read_sed_file(str(p), (100.0, 200.0), SED_WAVELENGTH_RANGE,
                      uniform_days=False)


@pytest.mark.parametrize("rows, message", [
    ("5.0 4000.0 1.0\n5.0 5000.0 1.0\n0.0 4000.0 1.0\n0.0 5000.0 1.0\n",
     "a new day must increment"),
    ("0.0 4000.0 1.0\n0.0 5000.0 1.0\n5.0 4000.0 1.0\n",
     "the wavelength grid must be the same"),
    ("0.0 4000.0 1.0\n0.0 5000.0 1.0\n5.0 4000.0 1.0\n5.0 6000.0 1.0\n",
     "expected LAM=5000.00"),
    ("0.0 5000.0 1.0\n0.0 4000.0 1.0\n5.0 5000.0 1.0\n5.0 4000.0 1.0\n",
     "found LAM=4000.0 at day 0.000000 but expected LAM=5000.00"),
])
def test_read_sed_file_aborts_on_a_broken_grid(tmp_path, rows, message):
    """rd_sedFlux aborts when a day decreases, when a day lists a different
    number of wavelengths, when a wavelength fails to repeat, or when the
    wavelengths fall within a day (sntools.c:8576-8624, :8675-8690)."""
    p = tmp_path / "bad.SED"
    p.write_text(rows)
    with pytest.raises(SNANAError, match=message):
        read_sed_file(str(p), DAYS, SED_WAVELENGTH_RANGE, uniform_days=False)


def test_read_sed_file_uniform_days_only_where_asked(tmp_path):
    """Non-uniform day bins abort unless the model allows them, as NON1ASED
    and SIMSED do through OPTMASK_DAYLIST_SEDMODEL (sntools.c:8549-8560)."""
    p = tmp_path / "steps.SED"
    _write_sed(p, np.array([0.0, 1.0, 2.0, 4.0]), np.array([4000.0, 5000.0]),
               np.ones((4, 2)))
    phases, _, _ = read_sed_file(str(p), DAYS, SED_WAVELENGTH_RANGE,
                                 t0shift_peakmag=False, uniform_days=False)
    np.testing.assert_array_equal(phases, [0.0, 1.0, 2.0, 4.0])
    with pytest.raises(SNANAError, match="day bins must be uniform"):
        read_sed_file(str(p), DAYS, SED_WAVELENGTH_RANGE, uniform_days=True)


def test_read_sed_file_reads_a_nan_flux_as_zero(tmp_path):
    """rd_sedFlux reads a NaN flux as zero (sntools.c:8520)."""
    p = tmp_path / "nan.SED"
    p.write_text("0.0 4000.0 nan\n0.0 5000.0 1.0\n")
    _, _, flux = read_sed_file(str(p), DAYS, SED_WAVELENGTH_RANGE,
                               t0shift_peakmag=False, uniform_days=False)
    np.testing.assert_array_equal(flux, [[0.0, 1.0]])


def test_read_sed_file_day_step_tolerance(tmp_path):
    """A day step may differ from the one before by up to 5e-4 days
    (sntools.c:8549-8560)."""
    def days(last):
        p = tmp_path / f"d{last}.SED"
        p.write_text("".join(f"{d} {w} 1.0\n" for d in (0.0, 1.0, last)
                             for w in (4000.0, 5000.0)))
        return str(p)

    read_sed_file(days("2.0004"), DAYS, SED_WAVELENGTH_RANGE,
                  t0shift_peakmag=False, uniform_days=True)
    with pytest.raises(SNANAError, match="day bins must be uniform"):
        read_sed_file(days("2.0006"), DAYS, SED_WAVELENGTH_RANGE,
                      t0shift_peakmag=False, uniform_days=True)


def test_read_sed_file_non_uniform_wavelength_step_only_warns(tmp_path,
                                                              caplog):
    """A non-uniform wavelength step warns and reads on (sntools.c:8595-8606)."""
    p = tmp_path / "lam.SED"
    p.write_text("".join(f"0.0 {w} 1.0\n" for w in (4000.0, 5000.0, 7000.0)))
    _, waves, _ = read_sed_file(str(p), DAYS, SED_WAVELENGTH_RANGE,
                                t0shift_peakmag=False, uniform_days=False)
    np.testing.assert_array_equal(waves, [4000.0, 5000.0, 7000.0])
    assert "non-uniform wavelength step" in caplog.text
