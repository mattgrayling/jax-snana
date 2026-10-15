"""Unit tests for the SNANA input-file parser's pure helpers."""

import logging
import os
import re
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from jax_snlc_core.utils import sndata_root
from jax_snlc_core.utils.error_handling import SNANAError
from jax_snlc_sim.builder import resolve_counts
from jax_snlc_sim.zvariation import parse_variation_keys, shift_array
from jax_snlc_sim.config_parser import (
    _ARGV_ONLY_KEYS,
    _parse_cutwin_filter_triplets,
    group_override_tokens,
    is_key,
    known_keys,
    parse_snana_input_file,
    read_keys,
)


_FIXTURES = Path(__file__).parent.parent / "fixtures" / "snana"
_A_FIXTURE_INPUT = str(_FIXTURES / "atmosphere_replay" / "input.input")


def _write_with_extra(tmp_path, extra_lines=""):
    """A bundled fixture input, optionally with appended lines."""
    path = tmp_path / "in.input"
    path.write_text(open(_A_FIXTURE_INPUT).read() + "\n" + extra_lines)
    return str(path)


def _write_broken(tmp_path):
    """A bundled fixture input with one value the build cannot read."""
    text = open(_A_FIXTURE_INPUT).read().replace(
        "NGENTOT_LC:  150", "NGENTOT_LC:  FIXME")
    assert "FIXME" in text, "fixture no longer sets NGENTOT_LC as expected"
    path = tmp_path / "broken.input"
    path.write_text(text)
    return str(path)


def _parse_with_extra(tmp_path, extra_lines=""):
    return parse_snana_input_file(_write_with_extra(tmp_path, extra_lines))


@pytest.mark.parametrize("genmodel", ['FIXMAG 20', 'RANMAG 18 22', 'SIMLIB',
                                      'fixmag 25', 'LCLIB'])
def test_magnitude_dictating_genmodels_are_rejected(genmodel):
    """These set the magnitude directly instead of integrating an SED
    (GENMODEL_NAME table, snlc_sim.c:9463-9479). Unimplemented, so they must
    say so: the multi-token forms otherwise die with a TypeError inside
    posixpath, and bare SIMLIB and LCLIB parse to an empty model_type and are
    accepted silently.
    """
    from jax_snlc_sim.config_parser import _check_genmodel_implemented
    with pytest.raises(SNANAError, match="not implemented"):
        _check_genmodel_implemented(genmodel)


def _field_path(cfg, dotted):
    for part in dotted.split('.'):
        cfg = getattr(cfg, part)
    return cfg


@pytest.mark.parametrize("base,field", [
    ("FUDGE_MAGERR", "instrument.fudge_magerr"),
    ("FUDGE_ZPTERR", "instrument.fudge_zpterr"),
    ("MJD_TEMPLATE", "survey.mjd_template"),
])
def test_key_plus_filter_keys_reach_the_config(tmp_path, base, field):
    """Both SNANA spellings are read: the bare key sets every band
    (snlc_sim.c:3687), `KEY_FILTER` only the named ones (:3696)."""
    bare = _field_path(_parse_with_extra(tmp_path, f"{base}: 5.0"), field)
    assert bare and set(bare.values()) == {5.0}

    named = _field_path(_parse_with_extra(tmp_path, f"{base}_FILTER: g 5.0"), field)
    assert named["g"] == 5.0
    assert set(named) - {"g"} and set(v for k, v in named.items() if k != "g") == {0.0}

    assert _field_path(_parse_with_extra(tmp_path), field) == {}


def test_parse_cutwin_filter_triplets_rejects_short_rows():
    """A malformed 'lo hi filters' row for a recognised cut key is fatal:
    silently dropping it would remove the cut."""
    cfg = {
        "CUTWIN_SNRMAX": [
            [3, 30, "griz"],
            [5, 50],          # too short
        ]
    }
    with pytest.raises(SNANAError, match="lo hi filters"):
        _parse_cutwin_filter_triplets(cfg, "CUTWIN_SNRMAX")


def test_solid_angle_names_the_fields(tmp_path):
    """SOLID_ANGLE(FIELDLIST) carries the field list with the solid angle
    (parse_input_SOLID_ANGLE, snlc_sim.c:4498-4530); naming the fields a
    second time through SIMLIB_FIELDLIST aborts (:4522-4525)."""
    cfg = _parse_with_extra(tmp_path, "SOLID_ANGLE(DEEP+SHALLOW): 0.02\n")
    assert cfg.rates.solid_angle == 0.02
    assert cfg.survey.simlib_fieldlist == "DEEP+SHALLOW"
    with pytest.raises(SNANAError, match="two places"):
        _parse_with_extra(tmp_path,
                          "SIMLIB_FIELDLIST: DEEP\nSOLID_ANGLE(DEEP): 0.02\n")


def test_resolve_model_type_hint_and_failure():
    """The path decides; the hint is only a fallback; neither is fatal."""
    from jax_snlc_core.models import resolve_model_type
    from jax_snlc_core.utils.error_handling import SNANAError
    assert resolve_model_type('SALT3.K21', hint='bayesn') == 'salt'
    assert resolve_model_type('MLCS2k2.X', hint='salt') == 'salt'
    with pytest.raises(SNANAError):
        resolve_model_type('MLCS2k2.X')


def test_infer_model_type_from_path_one_per_model():
    from jax_snlc_core.models import infer_model_type_from_path
    assert infer_model_type_from_path('SALT2.JLA-B14') == 'salt'
    assert infer_model_type_from_path('SALT3.K21') == 'salt'
    assert infer_model_type_from_path('BAYESN.W22') == 'bayesn'
    assert infer_model_type_from_path('SIMSED.SNIa-91bg') == 'simsed'
    assert infer_model_type_from_path('NON1ASED.V19_CC+HostXT') == 'non1ased'
    assert infer_model_type_from_path('REDBACK.arnett_bolometric') == 'redback'


def test_infer_model_type_from_path_non1a_aliases():
    """SNANA accepts NONIA/NON1A/NONIASED/NON1ASED (snlc_sim.c GENMODEL_NAME)."""
    from jax_snlc_core.models import infer_model_type_from_path
    assert infer_model_type_from_path('NON1A.V19') == 'non1ased'
    assert infer_model_type_from_path('NONIA.V19') == 'non1ased'


def test_infer_model_type_from_path_matches_basename_only():
    from jax_snlc_core.models import infer_model_type_from_path
    assert infer_model_type_from_path('/x/bayesn_runs/SALT3.K21') == 'salt'


def test_infer_model_type_from_path_prefix_not_substring():
    """SNANA matches with strncmp: the name must START with an alias."""
    from jax_snlc_core.models import infer_model_type_from_path
    assert infer_model_type_from_path('my_salt3_test') is None


def test_no_warning_on_known_input(tmp_path, caplog):
    """A real input file consumes every key: no unrecognized-key warning."""
    with caplog.at_level(logging.WARNING, logger="jax_snlc_sim.config_parser"):
        _parse_with_extra(tmp_path)
    assert "Unrecognized SNANA input key" not in caplog.text


def test_warns_on_unrecognized_key(tmp_path, caplog):
    """A bogus key (and a near-miss typo) is listed in one warning."""
    with caplog.at_level(logging.WARNING, logger="jax_snlc_sim.config_parser"):
        _parse_with_extra(tmp_path, "TOTALLY_BOGUS_KEY: 1\nGENMAG_TYPOO: 2\n")
    assert "Unrecognized SNANA input key" in caplog.text
    assert "TOTALLY_BOGUS_KEY" in caplog.text
    assert "GENMAG_TYPOO" in caplog.text


@pytest.mark.parametrize("line", [
    # SNANA aborts on these too (parse_input_OBSOLETE)
    "EXTINC_MILKYWAY: 1", "RVMW: 3.1", "GEN_SNDATA_SIM: 1", "SPECTYPE: 1",
    "SMEARFLAG_HOSTGAL: 1",
    # superseded or unported features
    "HOSTNOISE_FILE: some.DAT", "GENMODEL_ERRSCALE: 0.1", "FUDGEOPT_FLUXERR: 1",
    # SNANA parses these and never reads them back
    "EFFERR_STOPGEN: 0.5", "USE_SIMLIB_MAGOBS: 1", "GENRANGE_TOBS: -5.0 5.0",
    "HOSTLIB_GENRANGE_NSIGZ: -3.0 3.0", "STRETCH_TEMPLATE_FILE: some.dat",
    "GENSMEAR_RANFLAT_FIX: 0.5", "GENSMEAR_RANFlat_FIX: 0.5",
    # command-line only; SNANA aborts on it in an input file
    "HOSTLIB_MSKOPT_ADD: 8",
    # the fitter's flux non-linearity map, a fit-namelist key here
    "NONLINEARITY_FILE: some.DAT",
    # +HOSTNBR utility mode, LCLIB rate, stdout dumps, host-SED spectra
    "NNBR_WRITE_MAX: 3", "SEPNBR_MAX: 5.0", "DNDB_SCALE: 2.0",
    "HOSTLIB_GALID_DUMP: 1", "GENRANGE_DMPEVENT: 1 2",
    "GENRANGE_DMPTREST: -5 5",
    # GENSOURCE: GRID and the keys that configure it
    "GRID_FORMAT: FITS", "NGRID_COLORLAW: 2", "NGRID_COLORPAR: 2",
    "NGRID_LOGZ: 10", "NGRID_SHAPEPAR: 4", "NGRID_LUMIPAR: 4", "NGRID_TREST: 20",
    # a submit_batch_jobs master file fed to the simulator
    "BATCH_INFO: sbatch x 20", "CONFIG: 1",
    # bug-compatibility switches for old analyses
    "RESTORE_BUGS_DES3YR: 1", "RESTORE_BUG_ZHEL: 1", "RESTORE_DES5YR: 1",
    "RESTORE_WRONG_VPEC: 1",
    # harness/debug knobs with no analogue in a vectorized batch simulator
    "CLEARPROMPT: 1", "TRACE_MAIN: 1", "UNIT_TEST: IGM",
    # the deleted legacy search-efficiency readers
    "REFAC_SEARCHEFF_MAP: 0",
    # keys configuring the two unimplemented models
    "LCLIB_CUTWIN: LOGMASS 5 15", "LCLIB_DEBUG_DAYSCALE: 2.0",
    "LCLIB_DEBUG_TOBS_OFFSET: 10.0", "LCLIB_DEBUG_ZERO_TEMPLATE_FLUX: 1",
    "LCLIB_DEBUG_FORCE_NREPEAT: 3", "STRONGLENS_FILE: lens.DAT",
    "NPSFSIGMA_MINSEP_DETECT: 1.0", "USE_SIMLIB_GALID: 0",
    # not a sim-input key; unusable smear keys; unguarded S/N targeting
    "SNTYPE: 120 220", "GENMAG_SMEAR_USRFUN: 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8",
    "GENMAG_SMEARPAR_OVERRIDE: BIMODAL_SIGMACOH 0.12",
    "GENMODEL_ARGLIST: 'FLAG 1'", "FUDGE_SNRMAX: 20", "FUDGE2_SNRMAX: 20",
    # model-interpolation grid and the two shared-random-list knobs
    "TGRIDSTEP_MODEL_INTERP: 2.0", "NSTREAM_RAN: 2",
    "RANLIST_START_GENSMEAR: 100",
])
def test_rejected_key_aborts(tmp_path, line):
    """Keys jax refuses abort naming the key, rather than silently dropping."""
    with pytest.raises(SNANAError) as exc:
        _parse_with_extra(tmp_path, line + "\n")
    assert line.split(":")[0] in str(exc.value)


def test_path_sndata_sim_is_the_output_directory(tmp_path):
    """PATH_SNDATA_SIM names where the GENVERSION folder is written
    (prep_simpath, snlc_sim.c:8551), with environment variables expanded;
    SNDATA_ROOT is not an input key in SNANA and is not one here."""
    cfg = _parse_with_extra(tmp_path, "PATH_SNDATA_SIM: $HOME/sims\n")
    assert cfg.output.path_sndata_sim == os.path.expandvars("$HOME/sims")
    assert _parse_with_extra(tmp_path).output.path_sndata_sim == ""
    assert not hasattr(cfg.run, "sndata_root")


def test_overrides_parse_without_sndata_root(tmp_path):
    """The command-line key set is built by a probe build; nothing in it
    needs $SNDATA_ROOT, so an unset one cannot turn a known override into
    'Unknown command-line override key'. The run aborts later, naming the
    variable, when something under it is first read."""
    saved = os.environ.pop("SNDATA_ROOT", None)
    try:
        cfg = parse_snana_input_file(_write_with_extra(tmp_path), ["NGENTOT_LC", "7"])
        assert cfg.run.ngentot_lc == 7
        with pytest.raises(SNANAError, match=r"\$SNDATA_ROOT is not set"):
            sndata_root()
    finally:
        if saved is not None:
            os.environ["SNDATA_ROOT"] = saved


def test_validation_reports_every_bad_key_at_once(tmp_path):
    """All the checks run before any of them aborts, so a user with three bad
    keys fixes three at a time rather than one per run."""
    with pytest.raises(SNANAError) as exc:
        _parse_with_extra(tmp_path, "GENSOURCE: GRID\nEXTINC_HOSTGAL: SJPAR\n"
                                    "GENRANGE_PEAKMAG: 10 20\n")
    message = str(exc.value)
    assert "3 invalid inputs" in message
    for key in ("GENSOURCE", "EXTINC_HOSTGAL", "GENRANGE_PEAKMAG"):
        assert key in message


def test_single_failure_keeps_its_own_abort(tmp_path):
    """One bad key still reports the check that caught it, not a tally."""
    with pytest.raises(SNANAError) as exc:
        _parse_with_extra(tmp_path, "GENSOURCE: GRID\n")
    assert "invalid inputs" not in str(exc.value)
    assert "_check_gensource" in str(exc.value)


def test_non_integer_bitmask_aborts_cleanly(tmp_path):
    """GENMAG_SMEAR_MSKOPT is coerced only by its check, and that check now runs
    while the known keys are derived — outside the handler that names
    the offending override. A bare int() would traceback from there."""
    with pytest.raises(SNANAError, match="not an integer bitmask"):
        _parse_with_extra(tmp_path, "GENMAG_SMEAR_MSKOPT: abc\n")


@pytest.mark.parametrize("alt, canonical", [
    ("GENALPHA_SALT2", "GENPEAK_SALT2ALPHA"),
    ("GENBETA_SALT2", "GENPEAK_SALT2BETA"),
])
def test_live_snana_spelling_resolves_to_canonical(tmp_path, alt, canonical):
    """SNANA accepts these spellings, so jax must too — they set the same field.

    Confirmed against snlc_sim.exe: each runs to completion, assigning
    GENGAUSS_SALT2{ALPHA,BETA}.PEAK (snlc_sim.c:2351-2362).

    The sentinel must differ from the fixture's own value, or the assertion
    holds whether or not the alias fired.
    """
    sentinel = 0.4242
    assert f"{canonical}: {sentinel}" not in open(_A_FIXTURE_INPUT).read()
    parsed = read_keys(_write_with_extra(tmp_path, f"{alt}: {sentinel}\n"))
    assert parsed[canonical] == sentinel
    assert alt not in parsed


_RUN_ADMIN_KEYS = (
    "GENPREFIX: MYPREFIX\n"
    "PRESCALE_SIMGEN_DUMP: 3\n"
    "WRSPEC_PRESCALE: 2\n"
    "INIT_ONLY: 2\n"
    "SIMLIB_SURVEY: MYSURVEY\n"
    "PATH_USER_INPUT: /some/search/dir\n"
    "FORCEVAL_PSF: 1.5\n"
)


def test_run_admin_keys_parse_into_typed_config(tmp_path):
    """The 7 implemented run-admin keys flow through to the typed SimulationConfig."""
    cfg = _parse_with_extra(tmp_path, _RUN_ADMIN_KEYS)
    assert cfg.output.genprefix == "MYPREFIX"
    assert cfg.output.prescale_simgen_dump == 3
    assert cfg.output.wrspec_prescale == 2
    assert cfg.run.init_only == 2
    assert cfg.survey.survey_override == "MYSURVEY"
    assert cfg.run.path_user_input == "/some/search/dir"
    assert cfg.survey.forceval_psf == 1.5


def test_simgen_dump_prescale_alias(tmp_path):
    """SIMGEN_DUMP_PRESCALE is the accepted alias of PRESCALE_SIMGEN_DUMP."""
    cfg = _parse_with_extra(tmp_path, "SIMGEN_DUMP_PRESCALE: 4\n")
    assert cfg.output.prescale_simgen_dump == 4


_DIAGNOSTIC_KEYS = (
    "SIMGEN_DUMP_NOISE: 2\n"
    "SIMGEN_DUMP_RATE: 1\n"
    "SIMGEN_DUMP_MWCL: 1\n"
    "SIMGEN_DUMP_TRAINSALT: 1\n"
    "WRFLAG_YAML_FILE: 1\n"
    "SIMLIB_DUMP: 3\n"
    "APPEND_SNID_SEDINDEX: 1\n"
    "PATH_NON1ASED: /some/non1a/dir\n"
)


def test_diagnostic_output_keys_parse_into_typed_config(tmp_path):
    """The 8 diagnostic-output keys flow through to the typed config."""
    cfg = _parse_with_extra(tmp_path, _DIAGNOSTIC_KEYS)
    assert cfg.output.simgen_dump_noise == 2
    assert cfg.output.simgen_dump_rate == 1
    assert cfg.output.simgen_dump_mwcl == 1
    assert cfg.output.simgen_dump_trainsalt == 1
    assert cfg.output.wrflag_yaml_file == 1
    assert cfg.survey.simlib_dump == 3
    assert cfg.output.append_snid_sedindex == 1
    assert cfg.model._model_input_values["PATH_NON1ASED"] == "/some/non1a/dir"


# ----------------------------------------------------------------------------
# Command-line overrides: `snlc_sim.py my.input NGENTOT_LC 500`.
# ----------------------------------------------------------------------------


def _known():
    return known_keys(read_keys(_A_FIXTURE_INPUT), [])


@pytest.mark.parametrize("tokens, expected", [
    (['NGENTOT_LC', '42'], [('NGENTOT_LC', ['42'])]),
    (['GENRANGE_REDSHIFT', '0.2', '0.9'],
     [('GENRANGE_REDSHIFT', ['0.2', '0.9'])]),
    (['NGENTOT_LC', '42', 'GENVERSION', 'RUN2'],
     [('NGENTOT_LC', ['42']), ('GENVERSION', ['RUN2'])]),
])
def test_group_override_tokens(tokens, expected):
    assert group_override_tokens(tokens, _known()) == expected


def test_values_continue_onto_the_lines_below_their_key(tmp_path):
    """SNANA reads the input as a stream of words rather than as lines, so a
    key's values may run on beneath it: NON1A_KEYS names its columns on the
    following line in every NON1ASED library SNDATA_ROOT ships
    (snlc_sim.c:3559-3585), and SIMGEN_DUMP does the same for its variables
    (:5046-5059)."""
    raw = read_keys(_write_with_extra(
        tmp_path,
        "NON1A_KEYS: 5\n"
        "     INDEX   WGT    MAGOFF   MAGSMEAR  SNTYPE\n"
        "SIMGEN_DUMP: 3\n"
        "     CID GENTYPE MWEBV\n"))
    assert raw.all('NON1A_KEYS') == [
        [5, 'INDEX', 'WGT', 'MAGOFF', 'MAGSMEAR', 'SNTYPE']]
    assert raw.all('SIMGEN_DUMP') == [[3, 'CID', 'GENTYPE', 'MWEBV']]


def test_a_line_continuing_nothing_aborts(tmp_path):
    """A line naming no key, with no key above it to continue, is malformed."""
    path = tmp_path / "orphan.input"
    path.write_text("INDEX   WGT    MAGOFF\nGENVERSION: T\n")
    with pytest.raises(SNANAError, match="nothing for it to continue"):
        read_keys(str(path))


def test_count_prefixed_key_swallows_a_value_named_like_a_key():
    """SIMGEN_DUMP declares how many values follow, so GENTYPE — which is both
    a dump variable and a real key — stays a value instead of starting a group.
    """
    tokens = ['SIMGEN_DUMP', '3', 'CID', 'GENTYPE', 'MWEBV']
    known = _known()
    assert group_override_tokens(tokens, known) == [
        ('SIMGEN_DUMP', ['3', 'CID', 'GENTYPE', 'MWEBV'])]

    # Without the count rule, splitting at the next known key would cut the
    # group in two. Shown here so the assertion above cannot pass vacuously.
    naive, current = [], None
    for token in tokens:
        if token in known:
            current = (token, [])
            naive.append(current)
        else:
            current[1].append(token)
    assert len(naive) == 2 and naive[1][0] == 'GENTYPE'


@pytest.mark.parametrize("tokens, message", [
    (['500', 'NGENTOT_LC'], "Unknown command-line override key '500'"),
    (['SIMGEN_DUMP', '2', 'CID', 'GENTYPE', 'GENRANGE_REDSHIT', '0.2'],
     'GENRANGE_REDSHIT'),
    (['NGENTOT_LC'], "has no value"),
    (['SIMGEN_DUMP', '9', 'CID'], "declares 9 values but only 1 follow"),
])
def test_group_override_tokens_rejects(tokens, message):
    """Every key is validated, not just the first: a typo after a counted group
    would otherwise become a key and silently drop the override."""
    with pytest.raises(SNANAError, match=message):
        group_override_tokens(tokens, _known())


def test_mistyped_key_names_the_override_it_broke(tmp_path):
    """A typo mid-sequence is swallowed as a value of the preceding key, so it
    cannot be caught while grouping. It surfaces at the cast instead, and the
    error names the override rather than being a bare TypeError."""
    with pytest.raises(SNANAError) as exc:
        parse_snana_input_file(_write_with_extra(tmp_path),
                               ['NGENTOT_LC', '42', 'GENRANGE_REDSHIT', '0.2'])
    assert 'GENRANGE_REDSHIT' in str(exc.value)
    assert 'mistyped key' in str(exc.value)


def test_an_error_on_a_key_not_overridden_carries_no_override_hint(tmp_path):
    """An abort about a key the command line did not touch is reported as it
    is, without the mistyped-override hint."""
    with pytest.raises(SNANAError) as exc:
        parse_snana_input_file(_write_with_extra(tmp_path, 'OPT_MWEBV: 1'),
                               ['NGENTOT_LC', '42'])
    assert 'GENSIGMA_MWEBV_RATIO' in str(exc.value)
    assert 'mistyped key' not in str(exc.value)


@pytest.mark.parametrize("key", [
    'NGENTOT_LC',        # read straight from the input
    'W0_LAMBDA',         # an alias: read_keys rewrites it before storing
    'CALIB_FILE',        # another alias
    'HOSTLIB_MSKOPT_ADD',  # command-line only
    'SNTYPE',            # rejected, but still a key the user may name
    # Open families: the parser consumes these by iterating the dict, so no
    # lookup ever records them and only the prefix identifies them as keys.
    'CUTWIN_REDSHIFT_FINAL',
    'RANSYSTPAR_GENMAG_OFF_ZP',
    'SEARCHEFF_SPEC_SHIFT(g)',
    'GENMAG_SMEAR_SCALE(g)',
    # Read by name, but only from inside validate()
    'GENSOURCE',
    'EXTINC_HOSTGAL',
    'GENMAG_SMEAR_MSKOPT',
    'GENRANGE_PEAKMAG',
])
def test_known_keys_cover_more_than_the_keys_the_build_read(key):
    assert is_key(key, _known())


def test_open_family_override_reaches_the_config(tmp_path):
    """Recognising the key is not enough — it has to survive to the config."""
    cfg = parse_snana_input_file(
        _write_with_extra(tmp_path),
        ['CUTWIN_REDSHIFT_FINAL', '0.05', '0.11',
         'RANSYSTPAR_GENMAG_OFF_ZP', 'g', '0.03'])
    assert cfg.cuts.generic_ranges['redshift_final'] == (0.05, 0.11)
    assert cfg.run.ransystpar['GENMAG_OFF_ZP'] == ['g', 0.03]


def test_open_family_values_do_not_end_a_group():
    """`_override_values` stops at the next known key, and the prefix
    rule widens what counts as one. A CUTWIN group's own filter-string and
    negative-phase values must still be read as values."""
    assert group_override_tokens(
        ['CUTWIN_SNRMAX', '5', '999', 'gri', '-20', '80', 'GENVERSION', 'X'],
        _known()) == [
            ('CUTWIN_SNRMAX', ['5', '999', 'gri', '-20', '80']),
            ('GENVERSION', ['X'])]


def test_override_changes_the_config_and_rederives_counts(tmp_path):
    path = _write_with_extra(tmp_path)
    base = parse_snana_input_file(path)
    changed = parse_snana_input_file(
        path, ['NGENTOT_LC', '42', 'GENRANGE_REDSHIFT', '0.2', '0.9'])
    assert resolve_counts(changed, changed.hz_map())[0] == 42 \
        != resolve_counts(base, base.hz_map())[0]
    assert changed.rates.redshift_range == (0.2, 0.9) != base.rates.redshift_range


def test_no_overrides_leaves_the_config_untouched(tmp_path):
    """The override path must be inert when nothing is passed."""
    import dataclasses
    path = _write_with_extra(tmp_path)
    assert (dataclasses.asdict(parse_snana_input_file(path))
            == dataclasses.asdict(parse_snana_input_file(path, [])))


def test_dndz_override_replaces_the_files_rate_model(tmp_path):
    """A command-line rate model clears the file's (snlc_sim.c:2712)."""
    path = _write_with_extra(tmp_path)
    assert parse_snana_input_file(path).rates.dndz_entries == [['POWERLAW', 2.5e-05, 1.5]]
    after = parse_snana_input_file(path, ['DNDZ', 'MD14', '2.5E-5']).rates.dndz_entries
    assert after == [['MD14', 2.5e-05]]


def test_dndz_file_override_replaces_the_files_rate_model(tmp_path):
    """The same clear covers DNDZ_FILE, so it may override a file's DNDZ."""
    rate = tmp_path / "rate.dat"
    rate.write_text("0.0 1.0e-5\n1.0 2.0e-5\n")
    path = _write_with_extra(tmp_path)
    after = parse_snana_input_file(path, ['DNDZ_FILE', str(rate)]).rates.dndz_entries
    assert [e[0] for e in after] == ['FILE']


def test_dndz_and_dndz_file_together_in_a_file_abort(tmp_path):
    with pytest.raises(SNANAError, match="cannot mix multiple rate models"):
        _parse_with_extra(tmp_path, "DNDZ_FILE: rate.dat\n")


def test_two_rate_model_names_in_a_file_abort(tmp_path):
    with pytest.raises(SNANAError, match="POWERLAW and MD14"):
        _parse_with_extra(tmp_path, "DNDZ: MD14 2.5E-5\n")


def test_a_repeated_rate_model_keeps_the_last(tmp_path):
    """Every model but POWERLAW2 sets NMODEL_ZRANGE = 1 (snlc_sim.c:2802)."""
    cfg = _parse_with_extra(tmp_path, "DNDZ: POWERLAW 1.0E-4 1.5\n")
    assert cfg.rates.dndz_entries == [['POWERLAW', 1.0e-4, 1.5]]


def test_powerlaw2_ranges_accumulate(tmp_path):
    """POWERLAW2 alone does NMODEL_ZRANGE++ — its repeats are one model's
    z-ranges (snlc_sim.c:2816)."""
    path = tmp_path / "in.input"
    path.write_text(open(_A_FIXTURE_INPUT).read().replace(
        "DNDZ:  POWERLAW  2.5E-5  1.5",
        "DNDZ: POWERLAW2 2.5E-5 1.5 0.02 0.10\n"
        "DNDZ: POWERLAW2 3.0E-5 1.5 0.10 0.15"))
    assert len(parse_snana_input_file(str(path)).rates.dndz_entries) == 2


def test_each_argv_rate_model_clears_the_one_before_it(tmp_path):
    """The clear fires per DNDZ override, so a multi-range POWERLAW2 cannot be
    built on the command line — only the last range survives. Confirmed against
    snlc_sim.exe, which then aborts in SNrate_model for a redshift the surviving
    range does not cover."""
    cfg = parse_snana_input_file(_write_with_extra(tmp_path), [
        'DNDZ', 'POWERLAW2', '2.5E-5', '1.5', '0.02', '0.10',
        'DNDZ', 'POWERLAW2', '3.0E-5', '1.5', '0.10', '0.15'])
    assert cfg.rates.dndz_entries == [['POWERLAW2', 3.0e-5, 1.5, 0.10, 0.15]]


def test_override_value_containing_a_hash_aborts(tmp_path):
    """'#' starts a comment in the input grammar, so it cannot round-trip."""
    with pytest.raises(SNANAError, match="#"):
        parse_snana_input_file(_write_with_extra(tmp_path),
                               ['GENVERSION', 'RUN#2'])


@pytest.mark.parametrize("key", _ARGV_ONLY_KEYS)
def test_argv_only_keys_are_rejected_in_a_file(tmp_path, key):
    with pytest.raises(SNANAError, match=key):
        _parse_with_extra(tmp_path, f"{key}: 1\n")


_A_NON1ASED_INPUT = 'tests/fixtures/snana/non1ased_v19_replay/input.input'


@pytest.mark.parametrize("alias", ['NONIA', 'NON1ASED', 'NONIASED'])
def test_non1ased_row_key_aliases_are_read(tmp_path, alias):
    """All four spellings set IS_NON1AKEY (snlc_sim.c:3528)."""
    expected = parse_snana_input_file(
        _A_NON1ASED_INPUT).model._structured_config['non1a_config']
    path = tmp_path / "in.input"
    path.write_text(open(_A_NON1ASED_INPUT).read().replace('NON1A:', f'{alias}:'))
    parsed = parse_snana_input_file(str(path)).model._structured_config['non1a_config']
    assert (parsed['index_per_template'] == expected['index_per_template']).all()


@pytest.mark.parametrize("override", [
    ['NON1A_KEYS', '2', 'INDEX', 'WGT'],
    ['NON1A', '201', '0.5'],
    ['NONIA', '201', '0.5'],           # via the alias
    ['NONIA_KEYS', '2', 'INDEX', 'WGT'],
    ['PECIA', '201', '0.5'],
])
def test_file_only_keys_are_rejected_on_the_command_line(override):
    """parse_input_NON1ASED matches the colon-bearing form only, so check_argv
    rejects these (snlc_sim.c:3475). A NON1ASED input is what puts them among
    the known keys in the first place."""
    parse_snana_input_file(_A_NON1ASED_INPUT)
    with pytest.raises(SNANAError, match="belongs in the sim-input file"):
        parse_snana_input_file(_A_NON1ASED_INPUT, override)


@pytest.mark.parametrize("override,expected", [
    (['-1', '0.10'], (0.02, 0.10)),      # zmax tightened, zmin untouched
    (['-1', '0.99'], (0.02, 0.15)),      # zmax not loosened
    (['0.10', '-1'], (0.10, 0.15)),      # zmin raised
    (['0.001', '-1'], (0.02, 0.15)),     # zmin not lowered
    (['0.05', '0.12'], (0.05, 0.12)),    # both given, both applied
])
def test_negative_redshift_limit_from_argv_keeps_the_files(tmp_path, override,
                                                           expected):
    """A negative limit means "leave this one alone"; the other applies only if
    it narrows the range (snlc_sim.c:2160). The file has 0.0200 0.1500."""
    cfg = parse_snana_input_file(_write_with_extra(tmp_path),
                                 ['GENRANGE_REDSHIFT'] + override)
    assert cfg.rates.redshift_range == pytest.approx(expected)


@pytest.mark.parametrize("value", ["0.10 -1", "-1 0.15"])
def test_negative_redshift_limit_in_a_file_aborts(tmp_path, value):
    path = tmp_path / "in.input"
    path.write_text(open(_A_FIXTURE_INPUT).read().replace(
        "GENRANGE_REDSHIFT:  0.0200  0.1500", f"GENRANGE_REDSHIFT: {value}"))
    with pytest.raises(SNANAError, match="command-line override"):
        parse_snana_input_file(str(path))


@pytest.mark.parametrize("key,bad,good", [
    ('OPT_MWCOLORLAW', '42', '99'),   # valid codes: OPT_MWCOLORLAW_NAMES
    ('OPT_MWEBV', '42', '2'),         # valid codes: 0-3 (MWgaldust.h:43)
])
def test_invalid_mw_dust_option_aborts(tmp_path, key, bad, good):
    """An unknown code is fatal in SNANA (MWgaldust.c:217, :240). jax used to
    fall back silently — to the default colour law, or to no MW extinction."""
    def parse(value):
        path = tmp_path / "in.input"
        path.write_text(re.sub(rf'^{key}:.*$', f'{key}: {value}',
                               open(_A_FIXTURE_INPUT).read(), flags=re.M))
        return parse_snana_input_file(str(path))

    parse(good)
    with pytest.raises(SNANAError, match=f"Invalid {key}"):
        parse(bad)


@pytest.mark.parametrize("line,expected", [
    ('', 1),                       # SNANA's default (snlc_sim.c:1358)
    ('CUTWIN_NEPOCH: 0 -10', 0),   # kept: NEPOCH < 0 never rejects (:296)
    ('CUTWIN_NEPOCH: 3 -10', 3),
])
def test_cutwin_nepoch_minimum(tmp_path, line, expected):
    path = tmp_path / "in.input"
    path.write_text(re.sub(r'^CUTWIN_NEPOCH:.*$', line,
                           open(_A_FIXTURE_INPUT).read(), flags=re.M))
    assert parse_snana_input_file(str(path)).cuts.nepoch_min == expected


@pytest.mark.parametrize("opt", [0, 8, 16, 32, 64, 24, 48, 80])
def test_opt_setpkmjd_is_a_mask_not_an_enum(tmp_path, opt):
    """SNANA tests each bit separately (sntools.c:1614-1620), so combinations
    like 8|16 and 16|32 are ordinary inputs, and bit 64 is defined but nothing
    consumes it. None of these may be rejected.

    GENSIGMA_PEAKMJD is stripped because a Gaussian smear forces the option to
    0 (snlc_sim.c:10105), which would mask the value under test."""
    src = "\n".join(l for l in open(_A_FIXTURE_INPUT).read().splitlines()
                    if not l.startswith('GENSIGMA_PEAKMJD'))
    path = tmp_path / "in.input"
    path.write_text(f"{src}\nOPT_SETPKMJD: {opt}\n")
    assert parse_snana_input_file(str(path)).rates.opt_setpkmjd == opt


def test_simgen_dumpadd_appends_to_the_dump_columns(tmp_path):
    path = _write_with_extra(tmp_path)
    base = parse_snana_input_file(path).output.simgen_dumpall_columns
    added = parse_snana_input_file(
        path, ['SIMGEN_DUMPADD', 'ZCMB,AV']).output.simgen_dumpall_columns
    assert [c for c in added if c not in base] == ['ZCMB', 'AV']


def test_hostlib_mskopt_add_ors_into_the_mask(tmp_path):
    path = _write_with_extra(tmp_path)
    cfg = parse_snana_input_file(
        path, ['HOSTLIB_MSKOPT', '8', 'HOSTLIB_MSKOPT_ADD', '64'])
    assert cfg.host.mskopt == 8 | 64


def test_hostlib_mskopt_add_needs_a_nonzero_mask(tmp_path):
    """SNANA forbids it when HOSTLIB_MSKOPT is 0 (snlc_sim.c:3955)."""
    with pytest.raises(SNANAError, match="HOSTLIB_MSKOPT"):
        parse_snana_input_file(_write_with_extra(tmp_path),
                               ['HOSTLIB_MSKOPT_ADD', '64'])


def _a_non1a_fixture():
    """The one bundled input with more than one NON1A template."""
    path = str(_FIXTURES / "bbc_ccprior_sim" / "input.input")
    config = parse_snana_input_file(path)
    non1a = config.model._structured_config['non1a_config']
    assert non1a['n_templates'] > 1, f"{path} no longer carries several NON1A templates"
    return path, config


def test_non1a_index_force_keeps_one_template_at_full_weight():
    """snlc_sim.c:3607 collapses the list to the forced index with WGT=1."""
    path, base = _a_non1a_fixture()
    forced = int(base.model._structured_config['non1a_config']
                 ['index_per_template'][1])
    forced_config = parse_snana_input_file(
        path, ['NON1A_INDEX_FORCE', str(forced)]
    ).model._structured_config['non1a_config']
    assert forced_config['n_templates'] == 1
    assert list(forced_config['index_per_template']) == [forced]
    assert list(forced_config['weight_per_template']) == [1.0]


def test_non1a_index_force_rejects_an_absent_index():
    path, _ = _a_non1a_fixture()
    with pytest.raises(SNANAError, match="not among the NON1A templates"):
        parse_snana_input_file(path, ['NON1A_INDEX_FORCE', '999999'])


def test_missing_genpdf_file_aborts(tmp_path):
    """SNANA is fatal when it cannot open GENPDF_FILE (sntools_genPDF.c:191)."""
    with pytest.raises(SNANAError, match="GENPDF_FILE not found"):
        _parse_with_extra(tmp_path, "GENPDF_FILE: /nonexistent/missing.DAT\n")


def test_genmag_smear_in_a_genpdf_file(tmp_path):
    """GENMAG_SMEAR may sit in the GENPDF file (sntools_genPDF.c:221); a sim
    input that also sets it aborts, as init_modelSmear does (snlc_sim.c:9944)."""
    genpdf = tmp_path / "SMEAR.DAT"
    genpdf.write_text("GENMAG_SMEAR: 0.13\n")
    cfg = _parse_with_extra(tmp_path, f"GENPDF_FILE: {genpdf}\n")
    assert cfg.model.genmag_smear == 0.13
    with pytest.raises(SNANAError, match="Cannot define GENMAG_SMEAR in two input files"):
        _parse_with_extra(tmp_path, f"GENPDF_FILE: {genpdf}\nGENMAG_SMEAR: 0.10\n")


@pytest.mark.parametrize("token", ['+HOSTMAGS', '+HOSTNBR', '+HOSTAPPEND', '+HOSTSELECT'])
def test_hostlib_rewrite_flags_are_rejected(tmp_path, token):
    """They go through the one rejection list, written bare as SNANA takes
    them (snlc_sim.c:3964-3990)."""
    with pytest.raises(SNANAError, match="Unsupported SNANA input key"):
        parse_snana_input_file(_write_with_extra(tmp_path), [token])


def test_required_key_may_come_from_the_command_line(tmp_path):
    """snlc_sim.exe accepts a required key supplied only on the command line;
    verified against the binary with the GENMODEL line deleted."""
    body = [l for l in open(_A_FIXTURE_INPUT).read().splitlines()
            if not l.startswith('GENMODEL')]
    path = tmp_path / "nomodel.input"
    path.write_text("\n".join(body) + "\n")
    model = next(l.split(':', 1)[1].strip()
                 for l in open(_A_FIXTURE_INPUT) if l.startswith('GENMODEL'))
    assert parse_snana_input_file(str(path), ['GENMODEL', model]).model.genmodel == model
    with pytest.raises(SNANAError, match="GENMODEL"):
        parse_snana_input_file(str(path))


def test_validation_runs_after_overrides(tmp_path):
    """SNANA validates only once the command line has been applied
    (snlc_sim.c:680-683), so an override can repair the input."""
    body = [l for l in open(_A_FIXTURE_INPUT).read().splitlines()
            if not l.startswith('GENRANGE_REDSHIFT')]
    path = tmp_path / "noz.input"
    path.write_text("\n".join(body) + "\n")
    cfg = parse_snana_input_file(str(path), ['GENRANGE_REDSHIFT', '0.2', '0.9'])
    assert cfg.rates.redshift_range == (0.2, 0.9)
    with pytest.raises(SNANAError, match="zmin"):
        parse_snana_input_file(str(path))


# ----------------------------------------------------------------------------
# INPUT_FILE_INCLUDE. Production code since the initial commit, exercised by no
# bundled input, so these are its first tests.
# ----------------------------------------------------------------------------


def _write_include_chain(tmp_path, primary_extra="", **files):
    """Write a primary input plus named include files; return the primary path.

    Keys in `files` are paths relative to the primary's directory, so a name
    containing '/' places the include in a subdirectory.
    """
    for rel, body in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    primary = tmp_path / "primary.input"
    primary.write_text(open(_A_FIXTURE_INPUT).read() + "\n" + primary_extra)
    return str(primary)


def test_include_keys_reach_the_config(tmp_path):
    """A key that appears only in the include is read."""
    path = _write_include_chain(
        tmp_path,
        primary_extra="INPUT_FILE_INCLUDE: extra.input\n",
        **{"extra.input": "GENPREFIX: FROM_INCLUDE\n"})
    assert parse_snana_input_file(path).output.genprefix == "FROM_INCLUDE"


def test_include_scalar_overrides_the_primary(tmp_path):
    """Scalars are last-read-wins, and the include is read where its key sits."""
    path = _write_include_chain(
        tmp_path,
        primary_extra="INPUT_FILE_INCLUDE: extra.input\n",
        **{"extra.input": "NGENTOT_LC: 77\n"})
    assert read_keys(path)["NGENTOT_LC"] == 77


def test_include_accumulating_key_merges(tmp_path):
    """Repeatable keys append across files: an include cannot replace the list.

    This is why SNANA needs EXCLUDE_INCLUDE_FILE to swap an include out
    (snlc_sim.c:3545 increments NON1ASED->NINDEX and never resets per file).
    """
    before = len(read_keys(_A_FIXTURE_INPUT)["DNDZ"])
    path = _write_include_chain(
        tmp_path,
        primary_extra="INPUT_FILE_INCLUDE: extra.input\n",
        **{"extra.input": "DNDZ: MD14 2.5E-5\n"})
    assert len(read_keys(path)["DNDZ"]) == before + 1


def test_include_resolves_relative_to_the_primary_even_when_nested(tmp_path):
    """A nested include's relative path is anchored to the PRIMARY's directory.

    `mid.input` lives in a subdirectory and includes plain `leaf.input`. That
    resolves only if the anchor stays the primary's directory; anchoring to the
    including file would look in `sub/` and fail.
    """
    path = _write_include_chain(
        tmp_path,
        primary_extra="INPUT_FILE_INCLUDE: sub/mid.input\n",
        **{"sub/mid.input": "INPUT_FILE_INCLUDE: leaf.input\n",
           "leaf.input": "GENPREFIX: FROM_LEAF\n"})
    assert parse_snana_input_file(path).output.genprefix == "FROM_LEAF"


def test_include_alternate_spelling(tmp_path):
    """SNANA accepts INPUT_INCLUDE_FILE as well as INPUT_FILE_INCLUDE."""
    path = _write_include_chain(
        tmp_path,
        primary_extra="INPUT_INCLUDE_FILE: extra.input\n",
        **{"extra.input": "GENPREFIX: FROM_ALT_SPELLING\n"})
    assert parse_snana_input_file(path).output.genprefix == "FROM_ALT_SPELLING"


def test_argv_include_is_read_after_the_primary_one(tmp_path):
    """A command-line include is read last, so it wins (snlc_sim.c:672)."""
    path = _write_include_chain(
        tmp_path,
        primary_extra="INPUT_FILE_INCLUDE: first.input\n",
        **{"first.input": "GENPREFIX: FROM_FILE\n",
           "second.input": "GENPREFIX: FROM_ARGV\n"})
    assert parse_snana_input_file(path).output.genprefix == "FROM_FILE"
    argv = ['INPUT_FILE_INCLUDE', str(tmp_path / "second.input")]
    assert parse_snana_input_file(path, argv).output.genprefix == "FROM_ARGV"


def test_argv_exclude_drops_an_include(tmp_path):
    """EXCLUDE_INCLUDE_FILE matches on a substring, as SNANA does with strstr
    (snlc_sim.c:654)."""
    path = _write_include_chain(
        tmp_path,
        primary_extra="INPUT_FILE_INCLUDE: dusty.input\n",
        **{"dusty.input": "GENPREFIX: FROM_INCLUDE\n"})
    assert parse_snana_input_file(path).output.genprefix == "FROM_INCLUDE"
    assert parse_snana_input_file(
        path, ['EXCLUDE_INCLUDE_FILE', 'dust']).output.genprefix == ""


def test_argv_exclude_then_include_swaps_an_include(tmp_path):
    """The documented idiom: command-line includes are additive, so replacing
    one means excluding the old first (snana_manual.tex:7503)."""
    path = _write_include_chain(
        tmp_path,
        primary_extra="INPUT_FILE_INCLUDE: dusty.input\n",
        **{"dusty.input": "GENPREFIX: OLD\n", "fresh.input": "GENPREFIX: NEW\n"})
    argv = ['EXCLUDE_INCLUDE_FILE', 'dusty',
            'INPUT_FILE_INCLUDE', str(tmp_path / "fresh.input")]
    assert parse_snana_input_file(path, argv).output.genprefix == "NEW"


def test_argv_include_directives_mix_with_key_overrides(tmp_path):
    """The directives are consumed before grouping, so they do not look like
    unknown keys to the override grouper."""
    path = _write_include_chain(
        tmp_path,
        primary_extra="INPUT_FILE_INCLUDE: dusty.input\n",
        **{"dusty.input": "GENPREFIX: FROM_INCLUDE\n"})
    cfg = parse_snana_input_file(
        path, ['EXCLUDE_INCLUDE_FILE', 'dusty', 'NGENTOT_LC', '7'])
    assert cfg.output.genprefix == ""
    assert resolve_counts(cfg, cfg.hz_map())[0] == 7


def test_argv_include_directive_without_a_filename_aborts(tmp_path):
    with pytest.raises(SNANAError, match="needs a filename"):
        parse_snana_input_file(_write_with_extra(tmp_path),
                               ['EXCLUDE_INCLUDE_FILE'])


def test_include_does_not_warn_as_unrecognised(tmp_path, caplog):
    """The INCLUDE key itself is consumed, not reported as an unknown key."""
    path = _write_include_chain(
        tmp_path,
        primary_extra="INPUT_FILE_INCLUDE: extra.input\n",
        **{"extra.input": "GENPREFIX: X\n"})
    with caplog.at_level(logging.WARNING, logger="jax_snlc_sim.config_parser"):
        parse_snana_input_file(path)
    assert "Unrecognized SNANA input key" not in caplog.text


def test_gensigma_peakmjd_forces_opt_setpkmjd_zero(tmp_path):
    """GENSIGMA_PEAKMJD>0 selects the legacy smear, forcing OPT_SETPKMJD=0
    even when the input sets it to the Fmax-clump default (snlc_sim.c:10105)."""
    cfg = _parse_with_extra(tmp_path, "GENSIGMA_PEAKMJD: 1.5\nOPT_SETPKMJD: 16\n")
    assert cfg.rates.gensigma_peakmjd == 1.5
    assert cfg.rates.opt_setpkmjd == 0


def test_override_rescues_a_value_the_file_got_wrong(tmp_path):
    """Classifying tokens must not require the file's values to parse, or an
    override could never fix a bad one."""
    path = _write_broken(tmp_path)
    assert parse_snana_input_file(
        path, overrides=['NGENTOT_LC', '50']).run.ngentot_lc == 50

    with pytest.raises(SNANAError, match="NGENTOT_LC: cannot read 'FIXME'"):
        parse_snana_input_file(path)


def test_a_broken_value_does_not_shrink_the_known_keys(tmp_path):
    """A partial set misclassifies keys rather than just missing them:
    an override naming a dropped key reads as a value of the one before it."""
    path = _write_broken(tmp_path)
    clean = known_keys(read_keys(_A_FIXTURE_INPUT), [])
    broken = known_keys(read_keys(path), ['NGENTOT_LC', '50'])
    assert not clean - broken, (
        f"{len(clean - broken)} keys lost when the probe build aborts")

    # CIDOFF is dropped by a partial set; GENFILTERS would swallow it.
    cfg = parse_snana_input_file(path, overrides=[
        'GENFILTERS', 'gri', 'CIDOFF', '7', 'NGENTOT_LC', '50'])
    assert cfg.survey.genfilters == 'gri'
    assert cfg.output.cidoff == 7


# ----------------------------------------------------------------------------
# KEY_PLUS_FILTER repeatability and pairing (parse_input_KEY_PLUS_FILTER,
# snlc_sim.c:3648-3741)
# ----------------------------------------------------------------------------


def test_key_plus_filter_one_line_per_band_accumulates(tmp_path):
    from jax_snlc_sim.config_parser import _key_plus_filter
    raw = read_keys(_write_with_extra(
        tmp_path,
        "EXPOSURE_TIME_FILTER: g 2.0\n"
        "EXPOSURE_TIME_FILTER: r 3.0\n"))
    out = _key_plus_filter(raw, 'EXPOSURE_TIME', 1.0, 'gri')
    assert out['g'] == 2.0
    assert out['r'] == 3.0
    assert out['i'] == 1.0


def test_key_plus_filter_no_comma_in_bands_reads_one_value(tmp_path):
    """Without a comma in the filter token, sscanf reads a single value for
    every named band (snlc_sim.c:3697-3703)."""
    from jax_snlc_sim.config_parser import _key_plus_filter
    raw = read_keys(_write_with_extra(
        tmp_path, "EXPOSURE_TIME_FILTER: gri -6.4,3.2,6.9\n"))
    out = _key_plus_filter(raw, 'EXPOSURE_TIME', 1.0, 'gri')
    assert out == {'g': -6.4, 'r': -6.4, 'i': -6.4}


def test_key_plus_filter_comma_bands_pair_values(tmp_path):
    from jax_snlc_sim.config_parser import _key_plus_filter
    raw = read_keys(_write_with_extra(
        tmp_path, "EXPOSURE_TIME_FILTER: g,r,i 2.0,3.0,4.0\n"))
    out = _key_plus_filter(raw, 'EXPOSURE_TIME', 1.0, 'gri')
    assert out == {'g': 2.0, 'r': 3.0, 'i': 4.0}


# ----------------------------------------------------------------------------
# GENPERFECT (snlc_sim.c:8725-8731, 8776-8782)
# ----------------------------------------------------------------------------


def test_genperfect_exposure_preserves_band_ratios(tmp_path):
    from jax_snlc_sim.config_parser import _apply_genperfect, _key_plus_filter
    raw = read_keys(_write_with_extra(
        tmp_path,
        "GENPERFECT: 2\n"
        "EXPOSURE_TIME_FILTER: g 2.0\n"))
    _apply_genperfect(raw)
    assert raw['EXPOSURE_TIME'] == 1.0e5
    out = _key_plus_filter(raw, 'EXPOSURE_TIME', 1.0, 'gr')
    assert out['g'] == pytest.approx(2.0e5)


def test_genperfect_smear_bit_clears_filter_smear(tmp_path):
    from jax_snlc_sim.config_parser import _apply_genperfect
    raw = read_keys(_write_with_extra(
        tmp_path,
        "GENPERFECT: 4\n"
        "GENMAG_SMEAR_FILTER: g 0.1\n"))
    _apply_genperfect(raw)
    assert 'GENMAG_SMEAR_FILTER' not in raw


# ----------------------------------------------------------------------------
# TAKE_SPECTRUM: NONE (snlc_sim.c:5608)
# ----------------------------------------------------------------------------


def test_take_spectrum_none_discards_previous_entries(tmp_path):
    raw = read_keys(_write_with_extra(
        tmp_path,
        "TAKE_SPECTRUM: TREST(-5:20) TEXPOSE_ZPOLY(3600,0)\n"
        "TAKE_SPECTRUM: NONE\n"))
    assert raw['TAKE_SPECTRUM'] == []


# ----------------------------------------------------------------------------
# WV07 activation (snlc_sim.c:7947-7948): 0 is a meaningful value
# ----------------------------------------------------------------------------


def test_wv07_rewgt_zero_activates_profile():
    from jax_snlc_sim.config_parser import _wv07_profile
    assert _wv07_profile({'WV07_REWGT_EXPAV': 0}) == 1
    assert _wv07_profile({'WV07_REWGT_EXPAV': 0.0}) == 1
    assert _wv07_profile({'WV07_REWGT_EXPAV': -9.0}) == 0
    assert _wv07_profile({}) == 0


def test_cutwin_snrmax_repeatable_with_logic(tmp_path):
    """Each CUTWIN_SNRMAX line is an independent AND-combined cut
    (snlc_sim.c:4338-4346); the logic token is AND, OR, or a validated
    integer (prep_user_CUTWIN, snlc_sim.c:7741-7760)."""
    from jax_snlc_sim.config_parser import _cuts
    raw = read_keys(_write_with_extra(
        tmp_path,
        "CUTWIN_SNRMAX: 5 griz OR -20 60\n"
        "CUTWIN_SNRMAX: 3 griz AND -20 60\n"))
    cuts = _cuts(raw)
    assert cuts.snrmax_cuts == [(5.0, 'griz', 1, -20.0, 60.0),
                                (3.0, 'griz', 4, -20.0, 60.0)]

    raw = read_keys(_write_with_extra(
        tmp_path, "CUTWIN_SNRMAX: 3 griz 9 -20 60\n"))
    with pytest.raises(SNANAError, match="NFILT_CUT"):
        _cuts(raw)


def test_simgen_dumpall_columns_continue_across_lines(tmp_path):
    """SNANA reads the declared count of dump names from the token stream,
    newlines included (PREP_SIMGEN_DUMP, snlc_sim.c:5046-5059)."""
    src = open(_A_FIXTURE_INPUT).read()
    src = "\n".join(l for l in src.splitlines()
                    if not l.startswith('SIMGEN_DUMP'))
    inp = tmp_path / "dump_multiline.input"
    inp.write_text(src + "\nSIMGEN_DUMPALL: 5\n"
                   "  CID GENZ\n"
                   "  LIBID RA DEC\n"
                   "GENSIGMA_MWEBV_RATIO: 0\n")
    cfg = parse_snana_input_file(str(inp))
    assert cfg.output.simgen_dumpall
    assert cfg.output.simgen_dumpall_columns == [
        'CID', 'GENZ', 'LIBID', 'RA', 'DEC', 'FLAG_ACCEPT']


@pytest.mark.parametrize("line,expected", [
    ("CUTWIN_EPOCHS_SNRMIN: 5.0", 99999.0),
    ("CUTWIN_EPOCHS_SNRMIN: 5.0 20.0 gri", 20.0),
])
def test_cutwin_epochs_time_above_max_defaults_to_no_upper_bound(
        tmp_path, line, expected):
    """CUTWIN_EPOCHS_TRANGE[1] is initialised to 99999 (snlc_sim.c:1378), so
    the short form imposes no upper bound on TIME_ABOVE_SNRMIN. A 0.0 default
    would reject every event, since the cut passes on TIME_ABOVE < max."""
    path = _write_with_extra(tmp_path, line + "\n")
    cut = parse_snana_input_file(path).cuts.epochs_time_cut
    assert cut["snr_min"] == 5.0
    assert cut["time_above_max"] == expected


def test_genperfect_leaves_ebv_host_dust_alone(tmp_path):
    """GENPERFECT bit 16 zeroes GENPROFILE_AV in place (snlc_sim.c:8735-8754)
    without activating it, so a run whose host dust is modelled through
    EBV_HOST parses and keeps its own profile instead of gaining an AV one
    and tripping the both-specified guard."""
    path = _write_with_extra(
        tmp_path, "GENPERFECT: 1\nGENRANGE_EBV_HOST: 0 1\nGENTAU_EBV_HOST: 0.1\n")
    pops = parse_snana_input_file(path).model.populations
    assert 'EBV_HOST' in pops
    assert 'AV' not in pops


def test_genperfect_zeroes_an_av_profile_that_exists(tmp_path):
    """With AV-family keys present, bit 16 still pins the profile to AV=0."""
    path = _write_with_extra(
        tmp_path, "GENPERFECT: 1\nGENRANGE_AV: 0 3\nGENTAU_AV: 0.4\n")
    pops = parse_snana_input_file(path).model.populations
    assert 'AV' in pops
    assert float(pops['AV'].distributions[0].exp_tau) == 0.0


def test_cutwin_snrmax_rejects_a_filter_the_survey_lacks(tmp_path):
    """An undefined filter in CUTWIN_SNRMAX aborts (snlc_sim.c:24351-24358).
    Dropping it silently would leave the entry with no filters, and the
    NFILT requirement would then reject every event."""
    from jax_snlc_sim.survey.cutwin import make_jax_cutwin
    from dataclasses import replace
    from jax_snlc_sim.simulation import Cuts
    cuts = replace(Cuts(), snrmax_cuts=[(5.0, 'gX', 1, -20.0, 60.0)])
    with pytest.raises(SNANAError, match="Invalid filter"):
        make_jax_cutwin(cuts, ['LSST-g', 'LSST-r'], byfield_masks=())


def test_unused_argv_only_key_aborts():
    """A command-line-only key nothing consumes is fatal, as in SNANA's
    check_argv (sntools.c:11358-11383); a SALT input has no NON1A table."""
    with pytest.raises(SNANAError, match="not used by this GENMODEL"):
        parse_snana_input_file('tests/fixtures/snana/salt3_k21_replay/input.input',
                               ['NON1A_INDEX_FORCE', '834'])


def test_non1a_index_force_declasses_and_matches_index_zero(tmp_path):
    """The forced row is NON1A-classed whatever it was, a table INDEX of 0
    matches any forced index, and the kept row takes the forced index
    (snlc_sim.c:3612-3620)."""
    path, base = _a_non1a_fixture()
    src = open(path).read()
    first = next(l for l in src.splitlines() if l.startswith('NON1A:'))
    tokens = first.split()
    pec1a = 'PEC1A: ' + ' '.join(tokens[1:])
    inp = tmp_path / "in.input"
    inp.write_text(src.replace(first, first + "\n" + pec1a
                               + "\nDNDZ_PEC1A: POWERLAW 1.0E-5 1.5"))
    forced_index = int(float(tokens[1]))
    cfg = parse_snana_input_file(str(inp), ['NON1A_INDEX_FORCE',
                                           str(forced_index)])
    non1a = cfg.model._structured_config['non1a_config']
    assert list(non1a['ispec1a_per_template']) == [False]

    wildcard = src.replace(first, 'NON1A: 0 ' + ' '.join(tokens[2:]))
    inp.write_text(wildcard)
    cfg = parse_snana_input_file(str(inp), ['NON1A_INDEX_FORCE', '424242'])
    non1a = cfg.model._structured_config['non1a_config']
    # the kept row takes the forced index, replacing the wildcard 0 (:3563)
    assert list(non1a['index_per_template']) == [424242]


def test_variation_polynomial_vanishes_at_its_pivot():
    """PIVOT anchors the relation, so the population key the entry modifies
    means its value there and the slope can change without moving it."""
    entry, = parse_variation_keys(
        ["GENPEAK_SALT2x1 LOGMASS POLY -0.30 PIVOT 10.0"])
    assert entry.parname == "GENPEAK_SALT2x1"
    assert entry.depends_on == "LOGMASS"
    shifts = shift_array(entry, jnp.array([9.0, 10.0, 11.0]))
    np.testing.assert_allclose(np.asarray(shifts), [0.3, 0.0, -0.3], atol=1e-6)


def test_variation_coefficients_start_at_the_linear_term():
    """Unlike ZPOLY, whose first coefficient is the constant: PIVOT already
    fixes that, so a constant here would be degenerate with it."""
    entry, = parse_variation_keys(
        ["GENPEAK_SALT2c LOGsSFR POLY -0.01 0.002 PIVOT -10.0"])
    x = np.array([-12.0, -10.0, -8.0])
    expected = -0.01 * (x + 10.0) + 0.002 * (x + 10.0) ** 2
    np.testing.assert_allclose(
        np.asarray(shift_array(entry, jnp.asarray(x))), expected, atol=1e-6)


@pytest.mark.parametrize("line", [
    "GENPEAK_SALT2x1 LOGMASS POLY -0.30",              # no PIVOT
    "GENPEAK_SALT2x1 LOGMASS -0.30 PIVOT 10.0",        # no POLY
    "GENPEAK_SALT2x1 LOGMASS POLY PIVOT 10.0",         # no coefficients
    "GENPEAK_SALT2x1 LOGMASS POLY -0.30 PIVOT 10.0 3", # trailing token
])
def test_malformed_variation_aborts(line):
    with pytest.raises(SNANAError, match="VARIATION"):
        parse_variation_keys([line])


_UNPROVEN_KEYS = (
    "HOSTLIB_FILE: $JAX_TEST_FIXTURE_ROOT/hostlib/JAX_TEST.HOSTLIB\n"
    "HOSTLIB_SNR_DETECT: 5.0,10.0,20.0\n"
    "HOSTLIB_MAG_DETECT: 24.0,22.5\n"
    "HOSTLIB_ZPHOTEFF_FILE: /some/zphoteff.dat\n"
    "HOSTLIB_SBRADIUS: 0.7\n"
    "VEL_CMBAPEX: 0.0\n"
    "EPCUTWIN_SNRMIN: -3.0 9999.0\n"
    "WEAKLENS_DMUERR_FRAC: 0.25\n"
)


@pytest.mark.parametrize("key,value", [("MJDWIN_SETPKMJD", "45.0"),
                                       ("SNRCUT_SETPKMJD", "6.5"),
                                       ("SNRCUT_BACKUP", "2.5")])
def test_the_hardwired_setpkmjd_knobs_are_refused(tmp_path, key, value):
    """SNANA hard-wires the Fmax-clump window and both SNR cuts and reads no
    input key for any of them (PARLIST_SETPKMJD, snlc_sim.c:10111-10113): it
    ignores these spellings, so accepting them would silently give a peak
    estimate SNANA cannot produce. OPT_SETPKMJD is the only key there is."""
    with pytest.raises(SNANAError, match=key):
        _parse_with_extra(tmp_path, f"{key}: {value}\n")


def test_host_peakmjd_and_lensing_keys_reach_their_fields(tmp_path):
    """Seven keys whose spelling no other test exercises. SNANA sorts the
    two GAL_DETECT ladders (parse_input_HOSTLIB_GAL_DETECT): SNR descending,
    magnitude ascending, so a threshold ladder is read in the order the
    detection walk expects."""
    cfg = _parse_with_extra(tmp_path, _UNPROVEN_KEYS)
    assert cfg.host.snr_detect == [20.0, 10.0, 5.0]
    assert cfg.host.mag_detect == [22.5, 24.0]
    assert cfg.host.zphoteff_file == "/some/zphoteff.dat"
    assert cfg.host.sbradius == 0.7
    assert cfg.cosmology.vel_cmbapex == 0.0
    assert cfg.cuts.epcutwin_snrmin == (-3.0, 9999.0)
    assert cfg.lensing.dmuerr_frac == 0.25
