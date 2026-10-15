"""Unit tests for the NON1ASED template-library model."""

import glob
import os
import shutil

import numpy as np
import pytest

from jax_snlc_core.models.non1ased_model import (
    NON1ASEDModel, read_non1a_list)
from jax_snlc_core.utils.error_handling import SNANAError

_MODEL_DIR = "tests/fixtures/data/models/NON1ASED.V19_TEST"


def test_path_non1ased_fallback_search_dir(tmp_path):
    """SNANA PATH_NON1ASED: a NON1A SED not found in the model dir is resolved
    from the fallback search directory (by basename)."""
    sed = glob.glob(os.path.join(_MODEL_DIR, "*.SED*"))[0]
    moddir = tmp_path / "mod"
    altdir = tmp_path / "alt"
    moddir.mkdir()
    altdir.mkdir()
    # model dir has only NON1A.LIST; the SED lives only in the alt dir
    shutil.copy(os.path.join(_MODEL_DIR, "NON1A.LIST"), moddir / "NON1A.LIST")
    shutil.copy(sed, altdir / os.path.basename(sed))

    with pytest.raises(SNANAError):
        NON1ASEDModel.load(str(moddir), {})  # no fallback -> SED missing

    model = NON1ASEDModel.load(str(moddir), {'PATH_NON1ASED': str(altdir)})
    assert len(model.template_catalog) == len(read_non1a_list(
        os.path.join(_MODEL_DIR, "NON1A.LIST"))[0])


def test_unknown_non1a_index_fatal_at_configure():
    """A configured NON1A INDEX absent from the model's NON1A.LIST is fatal
    at model construction, matching SNANA's INDEXVALID check
    (genmag_NON1ASED.c:408-412); known indices resolve to dense stack
    positions with no dictionary fallback."""
    from jax_snlc_core.utils.error_handling import SNANAError

    model = NON1ASEDModel.load(_MODEL_DIR, {})
    known = sorted(model.index_map)

    def non1a_config(indices):
        n = len(indices)
        return {'SIGMACLIP_MAGSMEAR': (-3.0, 3.0), 'non1a_config': {
            'index_per_template': indices,
            'weight_per_template': np.ones(n),
            'ispec1a_per_template': np.zeros(n, dtype=bool)}}

    model.configure_generation(non1a_config(known), 100, 0.0)
    assert list(model._stack_pos_per_template) == [
        model.index_map[i] for i in known]

    with pytest.raises(SNANAError, match="Invalid NON1A index"):
        model.configure_generation(non1a_config(known + [99999]), 100, 0.0)


def test_only_the_selected_templates_are_read(tmp_path):
    """The input's NON1A rows select the templates; the loader reads those
    SEDs and no others (SNANA reads one SED per selected index,
    genmag_NON1ASED.c:122), so a library whose unselected files are
    missing still loads. An index absent from NON1A.LIST is fatal at
    construction."""
    source = "tests/fixtures/data/models/NON1ASED.K10_TEST"
    catalogue, _ = read_non1a_list(os.path.join(source, "NON1A.LIST"))
    chosen = [catalogue[1], catalogue[4]]
    model_dir = tmp_path / "lib"
    model_dir.mkdir()
    shutil.copy(os.path.join(source, "NON1A.LIST"), model_dir / "NON1A.LIST")
    for entry in chosen:
        name = entry['sed_file'] + ".gz"
        shutil.copy(os.path.join(source, name), model_dir / name)

    indices = [entry['index'] for entry in chosen]
    model = NON1ASEDModel.load(
        str(model_dir), {'non1a_config': {'index_per_template': indices}})
    assert model.index_map == {indices[0]: 0, indices[1]: 1}
    assert model.jax_template_seds.shape[0] == 2
    assert len(model.template_catalog) == len(catalogue)

    with pytest.raises(SNANAError, match="Invalid NON1A index = 99999"):
        NON1ASEDModel.load(str(model_dir),
                      {'non1a_config': {'index_per_template': [99999]}})


def test_non1a_list_header_keys_and_comment_markers(tmp_path):
    """FLUX_SCALE is a running value per entry, RESTLAMBDA_RANGE is
    model-level, and #/!/% start comments (read_NON1A_LIST,
    genmag_NON1ASED.c:619-660)."""
    lst = tmp_path / "NON1A.LIST"
    lst.write_text(
        "! header comment\n"
        "RESTLAMBDA_RANGE: 2500 11000\n"
        "FLUX_SCALE: 2.0\n"
        "NON1A: 101 II a.SED  % trailing comment\n"
        "FLUX_SCALE: 0.5\n"
        "PEC1A: 102 Ia91bg b.SED\n"
        "% NON1A: 999 II ghost.SED\n")
    templates, restlambda_range = read_non1a_list(str(lst))
    assert restlambda_range == (2500.0, 11000.0)
    assert [(t['index'], t['flux_scale'], t['is_pec1a']) for t in templates] \
        == [(101, 2.0, False), (102, 0.5, True)]


def _late_time_seds(model, template=0):
    """The template's DAYMAX and a callable giving its rest-frame SED."""
    daymax = float(np.asarray(model.jax_template_phase_max)[template])
    def sed(phase):
        return np.asarray(model.generate_sed(
            phase, {'template_idx': template, 'mu': 0.0}, {}))
    return daymax, sed


def test_the_late_time_line_is_anchored_on_the_boundary_sed():
    """get_flux_SEDMODEL fits its line through DAYMAX-0.01 and evaluates it
    from there (genmag_SEDtools.c:1555), so the SED at the anchor is the
    boundary SED the loader tabulated."""
    model = NON1ASEDModel.load(_MODEL_DIR, {'MINSLOPE_EXTRAPMAG_LATE': 0.0})
    daymax, sed = _late_time_seds(model)
    np.testing.assert_array_equal(sed(daymax - 0.01),
                                  np.asarray(model.jax_late_boundary_seds[0]))


def test_the_late_time_decay_starts_a_full_day_before_daymax():
    """SNANA replaces the phase interpolation for every Trest > DAYMAX-1.0
    (genmag_SEDtools.c:1530), so the exponential decay is already running
    half a day before DAYMAX and log(flux) is linear in phase across it.
    """
    model = NON1ASEDModel.load(_MODEL_DIR, {'MINSLOPE_EXTRAPMAG_LATE': 0.0})
    daymax, sed = _late_time_seds(model)

    def curvature(base):
        s1, s2, s3 = sed(base), sed(base + 1.0), sed(base + 2.0)
        return np.abs(np.log(s2 / s1) - np.log(s3 / s2))

    # inside the window: one straight line through all three phases
    assert curvature(daymax - 0.5).max() < 1e-5
    # a day earlier the first phase is still the template interpolation, so
    # the three no longer lie on one line
    assert curvature(daymax - 1.5).max() > 1e-3
