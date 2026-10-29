"""HOSTLIB_SPECBASIS_FILE / SPECDATA_FILE reader (read_specTable_SNANA)."""

import os
from pathlib import Path

import numpy as np
import pytest

from jax_snlc_core.logging import SNANAError
from jax_snlc_sim.hostlib.spectra import read_host_spec_table

_TABLE = """DOCUMENTATION:
  PURPOSE: test
DOCUMENTATION_END:

FLAM_SCALE:        2.0E-30   # trailing comment
FLAM_SCALE_POWZ1: -2

VARNAMES: ROW WAVELENGTH {p}00 {p}01
ROW: 1  2000.0  1.0  10.0
ROW: 2  2010.0  2.0  20.0
ROW: 3  2020.0  3.0  30.0
"""


def _write(tmp_path, prefix='SPECBASIS', text=None):
    path = tmp_path / "spec.dat"
    path.write_text(text if text is not None else _TABLE.format(p=prefix))
    return str(path)


@pytest.mark.parametrize("mode", ['SPECBASIS', 'SPECDATA'])
def test_reads_wavelength_and_template_columns(tmp_path, mode):
    t = read_host_spec_table(_write(tmp_path, mode), mode)
    assert t.n_spec == 2
    assert np.allclose(t.wave, [2000.0, 2010.0, 2020.0], atol=0)
    assert np.allclose(t.flam[0], [1.0, 2.0, 3.0], atol=0)
    assert np.allclose(t.flam[1], [10.0, 20.0, 30.0], atol=0)
    assert list(t.index) == [0, 1]


def test_flam_scale_and_powz1_give_the_redshift_normalisation(tmp_path):
    """FLAM = sum(coeff*basis) * FLAM_SCALE * (1+z)^POWZ1."""
    t = read_host_spec_table(_write(tmp_path), 'SPECBASIS')
    assert t.flam_scale == 2.0e-30 and t.flam_scale_powz1 == -2.0
    # abs=0: these fluxes are ~1e-30, so approx's default abs=1e-12 would
    # compare any two of them equal.
    assert t.scale(1.0) == pytest.approx(2.0e-30 / 4.0, abs=0)
    assert t.scale(0.0) == pytest.approx(2.0e-30, abs=0)


def test_wavelength_column_detected_by_name(tmp_path):
    """SNANA accepts wave/WAVE/lam/LAM as substrings
    (sntools_host.c:1885-1888)."""
    text = _TABLE.format(p='SPECBASIS').replace('WAVELENGTH', 'lam_rest')
    t = read_host_spec_table(_write(tmp_path, text=text), 'SPECBASIS')
    assert np.allclose(t.wave, [2000.0, 2010.0, 2020.0], atol=0)


def test_eazy_templates_list_resolves_against_the_list_files_directory(tmp_path):
    """SNANA keeps only each row's basename and resolves it against the list
    file's own directory, ignoring both the row's relative path and its scale
    column (sntools_host.c:2021). The row order, not the index column, numbers
    the basis vectors.
    """
    (tmp_path / "t1.dat").write_text("DOCUMENTATION:\nPURPOSE: template\nDOCUMENTATION_END:\n"
                                     "! comment\n% comment\n@ comment\n100.0 1.0\n200.0 2.0\n")
    (tmp_path / "t2.dat").write_text("# wave flux\n100.0 10.0\n200.0 20.0\n")
    # Row order reversed w.r.t. the index column, and the paths are bogus:
    # only the basename survives, so template order follows the rows.
    (tmp_path / "list.param").write_text(
        "7 nowhere/t2.dat 3.0\n1 elsewhere/t1.dat 0.5\n")
    text = ("FLAM_SCALE: 1.0\n"
            f"EAZY_TEMPLATES_LIST_FILE: {tmp_path / 'list.param'}\n")
    t = read_host_spec_table(_write(tmp_path, text=text), 'SPECBASIS')
    assert np.allclose(t.wave, [100.0, 200.0], atol=0)
    assert np.allclose(t.flam[0], [10.0, 20.0], atol=0)   # t2 read first
    assert np.allclose(t.flam[1], [1.0, 2.0], atol=0)
    assert list(t.index) == [0, 1]


def test_shipped_eazy_basis_matches_what_snana_loads(tmp_path):
    """The only real SPECBASIS set SNANA ships. snlc_sim.exe reports reading
    12 templates from this file; the wavelength grid is shared across them.
    """
    path = os.path.expandvars(
        '$SNDATA_ROOT/models/EAZY/SNANA_SPECBASIS_EAZY.DAT')
    assert os.path.exists(path), f"EAZY basis missing from $SNDATA_ROOT: {path}"
    t = read_host_spec_table(path, 'SPECBASIS')
    assert t.n_spec == 12
    assert t.flam.shape == (12, len(t.wave))
    assert t.flam_scale == 3.63078e-30 and t.flam_scale_powz1 == -2.0


def test_bin_edges_sit_midway_between_centres(tmp_path):
    """sntools_host.c:1814 — interior edges are midpoints; the first and last
    bins mirror their single neighbour's half-width."""
    t = read_host_spec_table(_write(tmp_path), 'SPECBASIS')
    assert np.allclose(t.wave_min, [1995.0, 2005.0, 2015.0], atol=0)
    assert np.allclose(t.wave_max, [2005.0, 2015.0, 2025.0], atol=0)


def test_grid_is_truncated_at_the_sedmodel_maximum(tmp_path):
    text = ("VARNAMES: ROW WAVELENGTH SPECBASIS00\n"
            "ROW: 1 20000.0 1.0\nROW: 2 25000.0 2.0\nROW: 3 40000.0 3.0\n")
    t = read_host_spec_table(_write(tmp_path, text=text), 'SPECBASIS')
    assert np.allclose(t.wave, [20000.0, 25000.0], atol=0)
    assert t.flam.shape == (1, 2)


def test_upper_truncation_uses_the_last_bins_lower_edge(tmp_path):
    """SNANA bounds the grid by WAVE_MIN at *both* ends (sntools_host.c:2288),
    so the last basis bin's own upper half is treated as off-grid. Pinned
    because it is an asymmetry, not an obvious consequence of the binning.
    """
    text = ("FLAM_SCALE: 1.0\nVARNAMES: ROW WAVELENGTH SPECBASIS00\n"
            "ROW: 1 1000.0 1.0\nROW: 2 1010.0 1.0\nROW: 3 1020.0 1.0\n")
    t = read_host_spec_table(_write(tmp_path, text=text), 'SPECBASIS')
    assert (t.wave_min[-1], t.wave_max[-1]) == (1015.0, 1025.0)
    # Wholly inside [wave_min[-1], wave_max[-1]] — the last bin's upper half.
    assert not t.on_grid(0.0, [1016.0], [1024.0])[0]


def test_hostlib_column_names_follow_snanas_spelling(tmp_path):
    """snlc_sim.exe aborts with "Could not find required HOSTLIB var
    'COEFF_SPECBASIS00'", so the prefix and the 2-digit read-order index are
    both pinned by SNANA's own message.
    """
    from jax_snlc_sim.hostlib.spectra import hostlib_column_names
    t = read_host_spec_table(_write(tmp_path), 'SPECBASIS')
    assert hostlib_column_names(t) == ['COEFF_SPECBASIS00', 'COEFF_SPECBASIS01']
    d = read_host_spec_table(_write(tmp_path, 'SPECDATA'), 'SPECDATA')
    assert hostlib_column_names(d) == ['IDSPECDATA']


def test_coefficients_are_gathered_in_basis_order(tmp_path):
    from jax_snlc_sim.hostlib.spectra import hostlib_coefficients
    t = read_host_spec_table(_write(tmp_path), 'SPECBASIS')
    cols = {'COEFF_SPECBASIS00': np.array([1.0, 2.0]),
            'COEFF_SPECBASIS01': np.array([10.0, 20.0]),
            'ZTRUE': np.array([0.1, 0.2])}
    assert np.allclose(hostlib_coefficients(t, cols),
                       [[1.0, 10.0], [2.0, 20.0]], atol=0)


def test_a_missing_coefficient_column_aborts(tmp_path):
    """A dropped column must not quietly become a zero coefficient."""
    from jax_snlc_sim.hostlib.spectra import hostlib_coefficients
    t = read_host_spec_table(_write(tmp_path), 'SPECBASIS')
    with pytest.raises(SNANAError, match="COEFF_SPECBASIS01"):
        hostlib_coefficients(t, {'COEFF_SPECBASIS00': np.array([1.0])})


def test_hostlib_load_gathers_coefficients_in_redshift_sorted_order(tmp_path):
    """The HOSTLIB build must request the COEFF_ columns and store them aligned
    with the redshift-sorted galaxy order the selector indexes by.

    Set on the config after parsing: the parser's own gate aborts on these
    keys until the spectra path exists, so this runs the build directly.
    """
    from jax_snlc_sim.inputs import parse_snana_input_file
    from jax_snlc_sim.builder import build_simulation

    spec = tmp_path / "spec.dat"
    spec.write_text(_TABLE.format(p='SPECBASIS'))
    hostlib = tmp_path / "T.HOSTLIB"
    # Deliberately not in redshift order, so a plain file-order gather fails.
    hostlib.write_text(
        "VARNAMES: GALID ZTRUE RA_GAL DEC_GAL "
        "COEFF_SPECBASIS00 COEFF_SPECBASIS01\n"
        "GAL: 1 0.30 10.0 -30.0 3.0 30.0\n"
        "GAL: 2 0.10 11.0 -30.0 1.0 10.0\n"
        "GAL: 3 0.20 12.0 -30.0 2.0 20.0\n")

    cfg = parse_snana_input_file(_fixture_input())
    cfg.host.hostlib_file = str(hostlib)
    cfg.host.specbasis_file = str(spec)
    # a range the three galaxies cover: the read-time window keeps them all
    cfg.rates.redshift_range = (0.11, 0.29)
    hostlib = build_simulation(cfg, 500).hostlib

    ztrue = np.asarray(hostlib.catalogue.redshifts)
    assert np.all(np.diff(ztrue) >= 0), "loader should sort by redshift"
    coeffs = hostlib.spectra.coeffs
    assert coeffs.shape == (3, 2)
    # Coefficients travel with their galaxy, not with the file row.
    assert np.allclose(coeffs[:, 0], ztrue * 10.0, atol=1e-6)


def test_host_epoch_without_a_spec_source_aborts(tmp_path):
    """TAKE_SPECTRUM: HOST with no SPECBASIS/SPECDATA has no SED to draw from.

    SNANA reaches this check only once a HOSTLIB is loaded and segfaults
    otherwise; jax catches it at config time.
    Without the check jax would emit an SN spectrum at phase 0 labelled as a
    host spectrum.
    """
    from jax_snlc_sim.inputs import parse_snana_input_file
    inp = tmp_path / "in.input"
    inp.write_text(open(_fixture_input()).read()
                   + "\nTAKE_SPECTRUM: HOST TEXPOSE_ZPOLY(2000,0)\n")
    with pytest.raises(SNANAError, match="without spec basis or data"):
        parse_snana_input_file(str(inp))


def test_both_contamination_fractions_abort(tmp_path):
    """HOSTFRAC and HOSTSNFRAC each set the same scale, so SNANA rejects the
    pair rather than letting the second silently win (snlc_sim.c:11737)."""
    from jax_snlc_sim.inputs import parse_snana_input_file
    inp = tmp_path / "in.input"
    inp.write_text(open(_fixture_input()).read()
                   + "\nTAKE_SPECTRUM_HOSTFRAC: 0.1"
                   + "\nTAKE_SPECTRUM_HOSTSNFRAC: 0.1\n")
    with pytest.raises(SNANAError, match="Cannot use both"):
        parse_snana_input_file(str(inp))


def test_contamination_without_a_host_epoch_aborts(tmp_path):
    """There is nothing to contaminate with. SNANA indexes GENSPEC.IMJD_HOST,
    left at its -9 initialiser when no HOST epoch was asked for."""
    from jax_snlc_sim.inputs import parse_snana_input_file
    inp = tmp_path / "in.input"
    inp.write_text(open(_fixture_input()).read()
                   + "\nTAKE_SPECTRUM_HOSTFRAC: 0.1\n")
    with pytest.raises(SNANAError, match="needs a host spectrum"):
        parse_snana_input_file(str(inp))


# ----------------------------------------------------------------------------
# Config gate: the keys are read and their file validated, but the SED is not
# yet evaluated onto the spectrograph bins, so they must abort rather than be
# accepted and quietly produce SN-only spectra.
# ----------------------------------------------------------------------------


def _fixture_input():
    return str(Path(__file__).parent.parent / "fixtures" / "snana"
               / "atmosphere_replay" / "input.input")


def test_out_of_range_idspecdata_aborts(tmp_path):
    """SNANA's guard is `IDSPEC > NSPECDATA` while its message says the last
    valid ID is NSPECDATA-1, so IDSPEC == NSPECDATA reaches an unallocated
    FLAM_BASIS row. jax uses the stated bound.
    """
    table = read_host_spec_table(_write(tmp_path, text=_TABLE.format(p='SPECDATA')),
                                 'SPECDATA')
    n = table.n_spec
    ok = table.event_seds(np.array([[n - 1]]), np.zeros(1))
    assert ok.shape[0] == 1
    for bad in (n, -1):
        with pytest.raises(SNANAError, match="Invalid IDSPECDATA"):
            table.event_seds(np.array([[bad]]), np.zeros(1))


def test_malformed_host_spectrum_file_fails_at_load(tmp_path):
    """The named file is read when the builder runs, so a bad table is caught
    before generation. Config parsing does not open it — an override may still
    replace the path."""
    from jax_snlc_sim.inputs import parse_snana_input_file
    from jax_snlc_sim.hostlib.spectra import load_host_spec_table
    spec = tmp_path / "spec.dat"
    spec.write_text("VARNAMES: ROW WAVELENGTH FLUX\nROW: 1 2000.0 1.0\n")
    inp = tmp_path / "in.input"
    inp.write_text(open(_fixture_input()).read()
                   + f"\nHOSTLIB_SPECBASIS_FILE: {spec}\n")
    cfg = parse_snana_input_file(str(inp))
    with pytest.raises(SNANAError, match="SPECBASIS"):
        load_host_spec_table(cfg)
