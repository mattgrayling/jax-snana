"""Unit tests for the WGTMAP file reader."""

import numpy as np
import pytest

from jax_snlc_core.logging import SNANAError
from jax_snlc_core.models import MODEL_REGISTRY
from jax_snlc_core.snana_files import WgtMapData, read_wgtmap, read_wgtmap_varnames
from jax_snlc_sim.hostlib.host_selection_weights import wgtmap_snvar_names
from jax_snlc_sim.population import declared_population, population_spellings

# what a SALT run declares: its population under each spelling, plus the
# host dust every model draws
SNVARS = wgtmap_snvar_names(population_spellings(
    *declared_population(MODEL_REGISTRY["salt"], {})[:2]))


@pytest.fixture(scope="module")
def wgtmap_2d_path(data_root):
    return data_root / "wgtmap" / "JAX_TEST_X1_LOGMASS.WGTMAP"


@pytest.fixture(scope="module")
def wgtmap_3d_path(data_root):
    return data_root / "wgtmap" / "JAX_TEST_X1_C_LOGMASS.WGTMAP"


def test_read_2d_wgtmap_grid_shape(wgtmap_2d_path):
    """2-D WGTMAP (x1, LOGMASS): one SN-var dim + one galaxy-var dim."""
    data = read_wgtmap(str(wgtmap_2d_path), SNVARS)
    assert isinstance(data, WgtMapData)
    assert "x1" in data.varnames
    assert "LOGMASS" in data.varnames
    assert data.ndim == 2
    assert "x1" in data.snvar_names
    assert "LOGMASS" not in data.snvar_names


def test_read_3d_wgtmap_grid_shape(wgtmap_3d_path):
    """3-D WGTMAP (x1, c, LOGMASS): two SN-var dims + one galaxy."""
    data = read_wgtmap(str(wgtmap_3d_path), SNVARS)
    assert data.ndim == 3
    assert set(data.varnames) == {"x1", "c", "LOGMASS"}
    assert set(data.snvar_names) == {"x1", "c"}


def test_varnames_spelling_and_row_order(tmp_path):
    """SNANA accepts plain VARNAMES:, and grid cells are placed by their
    coordinates, not by file row order."""
    path = tmp_path / "SHUFFLED.WGTMAP"
    path.write_text(
        "VARNAMES: LOGMASS WGT\n"
        "WGT: 10.0 2.0\n"
        "WGT:  8.0 1.0\n"
        "WGT: 12.0 3.0\n")
    data = read_wgtmap(str(path), SNVARS)
    np.testing.assert_allclose(np.asarray(data.wgt), [1.0, 2.0, 3.0])


def test_dimensions_are_positional_before_wgt(tmp_path):
    """Dimension columns are those before the WGT token; function columns
    follow it regardless of name."""
    path = tmp_path / "POSITIONAL.WGTMAP"
    path.write_text(
        "VARNAMES_WGTMAP: LOGMASS x1 WGT SNMAGSHIFT\n"
        "WGT:  9.0 -1.0 1.0 0.1\n"
        "WGT:  9.0  1.0 2.0 0.1\n"
        "WGT: 11.0 -1.0 3.0 0.1\n"
        "WGT: 11.0  1.0 4.0 0.1\n")
    data = read_wgtmap(str(path), SNVARS)
    assert data.varnames == ["LOGMASS", "x1"]
    assert data.snmagshift is not None
    np.testing.assert_allclose(np.asarray(data.wgt),
                               [[1.0, 2.0], [3.0, 4.0]])


def test_non_uniform_bins_abort(tmp_path):
    path = tmp_path / "NONUNIFORM.WGTMAP"
    path.write_text(
        "VARNAMES_WGTMAP: LOGMASS WGT\n"
        "WGT:  8.0 1.0\n"
        "WGT:  9.0 2.0\n"
        "WGT: 11.0 3.0\n")
    with pytest.raises(SNANAError, match="non-uniform"):
        read_wgtmap(str(path), SNVARS)


def test_any_models_parameter_is_an_sn_variable(tmp_path):
    """The caller names the per-SN quantities, so a dimension naming a
    non-SALT model's parameter, in SNANA's spelling, bins on the drawn value
    under the model's name rather than being looked for as a HOSTLIB
    column."""
    path = tmp_path / "THETA.WGTMAP"
    path.write_text(
        "VARNAMES_WGTMAP: THETA LOGMASS WGT\n"
        "WGT: -1.0  9.0 1.0\n"
        "WGT: -1.0 11.0 2.0\n"
        "WGT:  1.0  9.0 3.0\n"
        "WGT:  1.0 11.0 4.0\n")
    names = wgtmap_snvar_names(population_spellings(
        *declared_population(MODEL_REGISTRY["bayesn"], {})[:2]))
    data = read_wgtmap(str(path), names)
    assert data.snvar_names == ["theta"]
    assert "LOGMASS" not in data.snvar_names


def test_host_dust_is_an_sn_variable_for_every_model(tmp_path):
    """AV and RV are drawn per SN whatever the model, so they bin as SN
    variables under a model that declares neither."""
    path = tmp_path / "AV.WGTMAP"
    path.write_text(
        "VARNAMES_WGTMAP: AV LOGMASS WGT\n"
        "WGT: 0.0  9.0 1.0\n"
        "WGT: 0.0 11.0 2.0\n"
        "WGT: 1.0  9.0 3.0\n"
        "WGT: 1.0 11.0 4.0\n")
    names = wgtmap_snvar_names(population_spellings(
        *declared_population(MODEL_REGISTRY["bayesn"], {})[:2]))
    data = read_wgtmap(str(path), names)
    assert data.snvar_names == ["AV"]


def test_duplicate_cell_aborts(tmp_path):
    path = tmp_path / "DUP.WGTMAP"
    path.write_text(
        "VARNAMES_WGTMAP: LOGMASS WGT\n"
        "WGT: 9.0 1.0\n"
        "WGT: 9.0 2.0\n")
    with pytest.raises(SNANAError, match="appears"):
        read_wgtmap(str(path), SNVARS)


@pytest.mark.parametrize('key', ['VARNAMES', 'VARNAMES_WGTMAP'])
def test_varnames_reader_only_needs_header(tmp_path, key):
    """Discover dimensions without parsing grid rows needed only by the full reader."""
    path = tmp_path / 'header.WGTMAP'
    path.write_text(f'{key}: LOGMASS x1 WGT SNMAGSHIFT # columns\n'
                    'WGT: incomplete grid\n')
    assert read_wgtmap_varnames(str(path)) == ['LOGMASS', 'x1']


@pytest.mark.parametrize('header', ['', 'VARNAMES: LOGMASS', 'VARNAMES: WGT'])
def test_both_readers_report_the_same_header_error(tmp_path, header):
    """Both entry points use the same header rules and named errors."""
    path = tmp_path / 'bad.WGTMAP'
    path.write_text(header + '\n')
    with pytest.raises(SNANAError) as names_error:
        read_wgtmap_varnames(str(path))
    with pytest.raises(SNANAError) as grid_error:
        read_wgtmap(str(path), SNVARS)
    assert str(names_error.value) == str(grid_error.value)


def test_missing_wgtmap_raises_named_error(tmp_path):
    """Header-only reads preserve the normal SNANA missing-file diagnostic."""
    with pytest.raises(SNANAError, match='file not found'):
        read_wgtmap_varnames(str(tmp_path / 'missing.WGTMAP'))
