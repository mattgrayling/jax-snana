"""RANSYSTPAR draw semantics (SNANA prep_RANSYSTPAR, snlc_sim.c:8135-8349).

Draws are random per job, so there is no SNANA reference to replay; the knob
EFFECTS are covered by the knobs' own parity fixtures. Here: drawn values
land in the right config knobs, the same RANSEED_SYST reproduces them,
IDSURVEY changes only the unsynced draws, and all draws respect the +/-3
sigma clip.
"""

import os
import re
from pathlib import Path

import numpy as np
import pytest

from jax_snlc_core.utils.error_handling import SNANAError
from jax_snlc_sim.config_parser import parse_snana_input_file

FIXTURE_INPUT = (Path(__file__).resolve().parents[1]
                 / "fixtures" / "snana" / "salt3_k21_replay" / "input.input")

RANSYSTPAR_BLOCK = """
RANSYSTPAR_RANSEED_SYST: 4242
{idsurvey_line}
RANSYSTPAR_SIGSCALE_MWEBV: 0.05
RANSYSTPAR_SIGSHIFT_MWRV: 0.2
RANSYSTPAR_SIGSHIFT_REDSHIFT: 0.001
RANSYSTPAR_SIGSHIFT_OMEGA_MATTER: 0.01
RANSYSTPAR_RANGESHIFT_W0: -0.05 0.05
RANSYSTPAR_SIGSCALE_FLUXERR: 0.05
RANSYSTPAR_SIGSCALE_FLUXERR2: 0.05
RANSYSTPAR_SIGSHIFT_ZP(gri): 0.01
RANSYSTPAR_SIGSHIFT_LAMFILT(z): 20.0
"""


def _config_with(tmp_path, idsurvey, tag):
    base = FIXTURE_INPUT.read_text()
    base = re.sub(r"GENVERSION:\s+\S+", "GENVERSION:  JAX_RSP_TEST", base)
    idsurvey_line = (f"RANSYSTPAR_IDSURVEY: {idsurvey}"
                     if idsurvey is not None else "")
    inp = tmp_path / f"rsp_{tag}.input"
    inp.write_text(base + RANSYSTPAR_BLOCK.format(idsurvey_line=idsurvey_line))
    return parse_snana_input_file(str(inp))


def test_ransystpar_draws(tmp_path):
    cfg = _config_with(tmp_path, idsurvey=10, tag="a")
    base = parse_snana_input_file(str(FIXTURE_INPUT))

    # every configured knob moved off its baseline
    assert cfg.mw_dust.scale != 1.0
    assert cfg.mw_dust.rv != base.mw_dust.rv
    assert cfg.rates.genbias_redshift != 0.0
    assert cfg.cosmology.omega_matter != base.cosmology.omega_matter
    assert cfg.cosmology.w0 != base.cosmology.w0
    assert set(cfg.instrument.fudgescale_fluxerr) and all(
        v != 1.0 for v in cfg.instrument.fudgescale_fluxerr.values())
    assert set(cfg.instrument.fudgescale_fluxerr2) and all(
        v != 1.0 for v in cfg.instrument.fudgescale_fluxerr2.values())
    assert sorted(k for k, v in cfg.instrument.genmag_off_zp.items() if v != 0.0) == ['g', 'i', 'r']
    assert [k for k, v in cfg.instrument.fudgeshift_lam.items() if v != 0.0] == ['z']

    # clip bounds
    assert abs(cfg.mw_dust.scale - 1.0) <= 0.05 * 3.0
    assert abs(cfg.rates.genbias_redshift) <= 0.001 * 3.0
    assert abs(cfg.cosmology.w0 - base.cosmology.w0) <= 0.05
    for c in 'gri':
        assert abs(cfg.instrument.genmag_off_zp[c]) <= 0.01 * 3.0

    # reproducible: same seeds, same everything
    cfg2 = _config_with(tmp_path, idsurvey=10, tag="b")
    assert cfg2.mw_dust.scale == cfg.mw_dust.scale
    assert cfg2.instrument.genmag_off_zp == cfg.instrument.genmag_off_zp
    assert cfg2.instrument.fudgeshift_lam == cfg.instrument.fudgeshift_lam

    # different IDSURVEY: synced draws identical, unsynced draws differ
    cfg3 = _config_with(tmp_path, idsurvey=11, tag="c")
    assert cfg3.mw_dust.scale == cfg.mw_dust.scale
    assert cfg3.cosmology.omega_matter == cfg.cosmology.omega_matter
    assert cfg3.rates.genbias_redshift == cfg.rates.genbias_redshift
    assert cfg3.instrument.genmag_off_zp != cfg.instrument.genmag_off_zp
    assert cfg3.instrument.fudgescale_fluxerr != cfg.instrument.fudgescale_fluxerr

    # no RANSYSTPAR_IDSURVEY: SIMLIB SURVEY (LSST) via SURVEY.DEF -> 12
    cfg4 = _config_with(tmp_path, idsurvey=None, tag="d")
    cfg12 = _config_with(tmp_path, idsurvey=12, tag="e")
    assert cfg4.instrument.genmag_off_zp == cfg12.instrument.genmag_off_zp
    assert cfg4.instrument.fudgescale_fluxerr == cfg12.instrument.fudgescale_fluxerr


GENPDF_CONTENT = """VARNAMES: SALT2c PROB
PDF: -0.3 0.001
PDF:  0.0 1.00
PDF:  0.3 0.001
"""


def test_ransystpar_wildcards(tmp_path):
    base = FIXTURE_INPUT.read_text()
    base = re.sub(r"GENVERSION:\s+\S+", "GENVERSION:  JAX_RSP_WC", base)
    genmodel = re.search(r"GENMODEL:\s+(\S+)", base).group(1)

    for tag in "AB":
        (tmp_path / f"genpdf_{tag}.dat").write_text(GENPDF_CONTENT)
    # GENMODEL wildcard: two symlinks to the real model dir, so the pick
    # is observable while everything downstream of the path still loads
    model_dir = Path(os.path.expandvars(genmodel))
    picks = []
    for tag in "AB":
        link = tmp_path / f"{model_dir.name}_SYS{tag}"
        link.symlink_to(model_dir)
        picks.append(str(link))

    inp = tmp_path / "rsp_wc.input"
    inp.write_text(base + f"""
RANSYSTPAR_RANSEED_SYST: 4242
RANSYSTPAR_GENMODEL_WILDCARD: {tmp_path}/{model_dir.name}_SYS*
RANSYSTPAR_GENPDF_FILE_WILDCARD: {tmp_path}/genpdf_*.dat
""")
    cfg = parse_snana_input_file(str(inp))

    assert cfg.model.genmodel in picks
    assert cfg.model.model_type == "salt"
    assert cfg.model.populations["c"].genpdf is not None

    # reproducible: same seed, same picks
    cfg2 = parse_snana_input_file(str(inp))
    assert cfg2.model.genmodel == cfg.model.genmodel

    # different seed can land on a different file; both picks span the list
    seeds_seen = {cfg.model.genmodel}
    for seed in (1, 2, 3, 4, 5):
        alt = tmp_path / f"rsp_wc_{seed}.input"
        alt.write_text(inp.read_text().replace("4242", str(seed)))
        seeds_seen.add(parse_snana_input_file(str(alt)).model.genmodel)
    assert seeds_seen == set(picks)

    # batch job (NJOBTOT>0): sequential pick by JOBID, not random
    for jobid, expect in ((1, picks[0]), (2, picks[1])):
        batch = tmp_path / f"rsp_wc_job{jobid}.input"
        batch.write_text(inp.read_text()
                         + f"JOBID: {jobid}\nNJOBTOT: 2\n")
        assert parse_snana_input_file(str(batch)).model.genmodel == expect

    # JOBID beyond the matched-file count aborts, as SNANA
    bad = tmp_path / "rsp_wc_job3.input"
    bad.write_text(inp.read_text() + "JOBID: 3\nNJOBTOT: 3\n")
    with pytest.raises(SNANAError, match="invalid JOBID"):
        parse_snana_input_file(str(bad))


def _season_input(tmp_path, name, extra):
    """The fixture switched to NGEN_SEASON, so the count follows the rate model
    and therefore the cosmology; NGENTOT_LC would not."""
    base = FIXTURE_INPUT.read_text()
    base = re.sub(r"GENVERSION:\s+\S+", "GENVERSION:  JAX_RSP_TEST", base)
    base = re.sub(r"NGENTOT_LC:\s+\S+", "NGEN_SEASON: 1.0", base)
    inp = tmp_path / name
    inp.write_text(base + extra)
    return parse_snana_input_file(str(inp))


def test_shifts_reach_the_entry_point_the_cli_uses(tmp_path):
    """The shifts used to live behind a second loader that took no overrides,
    so `snlc_sim.py`, which parses with overrides, silently skipped them and
    produced nominal output for every systematics job. One entry point now, and
    it has to apply them with overrides in play.
    """
    base = re.sub(r"GENVERSION:\s+\S+", "GENVERSION:  JAX_RSP_TEST",
                  FIXTURE_INPUT.read_text())
    inp = tmp_path / "cli.input"
    inp.write_text(base + RANSYSTPAR_BLOCK.format(idsurvey_line=""))

    cfg = parse_snana_input_file(str(inp), ['NGEN_SCALE', '1.0'])
    assert cfg.mw_dust.scale != 1.0
    assert cfg.cosmology.omega_matter != parse_snana_input_file(
        str(FIXTURE_INPUT)).cosmology.omega_matter


def test_shifts_land_before_the_counts_are_resolved(tmp_path):
    """SNANA draws the shifts before init_RateModel (snlc_sim.c:122-126), so
    the generated count comes from the shifted cosmology. Pinning RANSEED_SYST
    makes the draw reproducible; the control input hard-codes the drawn
    OMEGA_MATTER, so this compares one engine result against another rather
    than recomputing the rate integral here.
    """
    shifted = _season_input(
        tmp_path, "shift.input",
        "\nRANSYSTPAR_RANSEED_SYST: 4242"
        "\nRANSYSTPAR_SIGSHIFT_OMEGA_MATTER: 0.05\n")
    control = _season_input(
        tmp_path, "control.input",
        f"\nOMEGA_MATTER: {shifted.cosmology.omega_matter:.10f}\n")
    nominal = _season_input(tmp_path, "nominal.input", "")

    from jax_snlc_sim.builder import resolve_counts

    def n_simulated(cfg):
        return resolve_counts(cfg, cfg.hz_map())[0]

    assert n_simulated(shifted) == n_simulated(control)
    assert n_simulated(shifted) != n_simulated(nominal), "shift too small to bite"


def test_sigshift_zp_multi_group_reads_one_value_per_group(tmp_path):
    """SNANA reads one value per comma-separated filter group
    (parse_input_RANSYSTPAR, snlc_sim.c:4706-4738)."""
    from jax_snlc_sim.ransystpar import _filter_groups
    sig = _filter_groups({'SIGSHIFT_ZP(gr,iz)': ['0.01', '0.02']},
                         'SIGSHIFT_ZP', 'griz')
    assert sig['g'] == sig['r'] == 0.01
    assert sig['i'] == sig['z'] == 0.02


def test_sigshift_zp_bare_key_takes_one_value_per_filter(tmp_path):
    """The bare key expands to one group per GENFILTERS filter, so it takes
    NFILT values; a wrong count is fatal (snlc_sim.c:4706-4728)."""
    from jax_snlc_core.utils.error_handling import SNANAError
    from jax_snlc_sim.ransystpar import _filter_groups
    sig = _filter_groups({'SIGSHIFT_ZP': ['0.01', '0.02', '0.03']},
                         'SIGSHIFT_ZP', 'gri')
    assert sig == {'g': 0.01, 'r': 0.02, 'i': 0.03}
    with pytest.raises(SNANAError, match="expected 3 values"):
        _filter_groups({'SIGSHIFT_ZP': 0.01}, 'SIGSHIFT_ZP', 'gri')


def test_sigshift_zphot_host_keeps_higher_order_bias(tmp_path):
    """SNANA assigns only HOSTLIB_GENZPHOT_BIAS[0] (snlc_sim.c:8199-8205)."""
    import numpy as _np
    from jax_snlc_sim.ransystpar import prep_ransystpar
    cfg = _config_with(tmp_path, None, "zphot")
    cfg.host.genzphot_bias = [0.0, 0.5, -0.2]
    cfg.run.ransystpar = {'RANSEED_SYST': 4242,
                          'SIGSHIFT_zPHOT_HOST': 0.001}
    prep_ransystpar(cfg)
    assert cfg.host.genzphot_bias[1:] == [0.5, -0.2]
    assert cfg.host.genzphot_bias[0] != 0.0


def test_ranseed_gen_overrides_the_generation_seed(tmp_path):
    """RANSYSTPAR_RANSEED_GEN re-seeds generation after the systematic draws
    (prep_RANSYSTPAR, snlc_sim.c:8335-8341), so the run generates with it
    while the systematics keep whichever seed drew them. It reaches
    Run.random_seed, which every draw in the run now reads, with or without
    RANSEED_SYST beside it."""
    base = re.sub(r"GENVERSION:\s+\S+", "GENVERSION:  JAX_RSP_GEN",
                  FIXTURE_INPUT.read_text())
    nominal = parse_snana_input_file(str(FIXTURE_INPUT)).run.random_seed
    assert nominal != 777

    for tag, block in (("alone", "RANSYSTPAR_RANSEED_GEN: 777\n"),
                       ("with_syst", "RANSYSTPAR_RANSEED_SYST: 4242\n"
                                     "RANSYSTPAR_RANSEED_GEN: 777\n")):
        inp = tmp_path / f"rsp_gen_{tag}.input"
        inp.write_text(base + "\n" + block)
        assert parse_snana_input_file(str(inp)).run.random_seed == 777, tag


def test_ranseed_gen_is_not_warned_as_unrecognised(tmp_path, caplog):
    """The RANSYSTPAR_ family is an open prefix, so the key must not trip the
    unrecognised-key warning on its way to Run.ransystpar."""
    import logging
    base = re.sub(r"GENVERSION:\s+\S+", "GENVERSION:  JAX_RSP_GEN2",
                  FIXTURE_INPUT.read_text())
    inp = tmp_path / "rsp_gen_warn.input"
    inp.write_text(base + "\nRANSYSTPAR_RANSEED_GEN: 777\n")
    with caplog.at_level(logging.WARNING, logger="jax_snlc_sim.config_parser"):
        parse_snana_input_file(str(inp))
    assert "Unrecognized SNANA input key" not in caplog.text


def test_named_wildcards_select_files_before_preparation(tmp_path):
    from jax_snlc_sim.inputs import build_config, command_settings, parse_input

    model = tmp_path/'SALT3.K21_SYS'
    model.mkdir()
    pdf = tmp_path/'genpdf_A.dat'
    pdf.write_text(GENPDF_CONTENT)
    systematic = dict(IDSURVEY=1, RANSEED_SYST=42,
                      GENMODEL_WILDCARD=str(tmp_path/'SALT3.*'),
                      GENPDF_FILE_WILDCARD=str(tmp_path/'genpdf_*.dat'))
    settings = dict(genmodel='SALT3.K21', genfilters='griz', simlib_file='survey.simlib',
                    kcor_file='survey.fits', random_seed=42, redshift_range=[.1, .2],
                    trest_range=[-10, 40], peakmjd_range=[60000, 60010], ngentot_lc=2,
                    sigma_ratio=0., ransystpar=systematic)
    named = build_config(settings)
    assert named.model.genmodel == str(model)
    assert named.model.populations['c'].genpdf is not None
    cli, remaining = command_settings(['--ransystpar', 'RANSEED_SYST', '42'], settings)
    assert not remaining
    assert build_config({**settings, 'ransystpar': {**systematic, **cli['ransystpar']}}).model.genmodel == str(model)
    # A named override on a SNANA file uses the same resolver and preserves
    # the file's other random-systematic settings.
    inp = tmp_path/'simulation.input'
    inp.write_text(FIXTURE_INPUT.read_text() + '\nRANSYSTPAR_IDSURVEY: 1\n')
    mixed = parse_input(inp, {'ransystpar': systematic})
    assert mixed.model.genmodel == str(model)
    assert mixed.model.populations['c'].genpdf is not None
    assert settings['genmodel'] == 'SALT3.K21'
    for name in ('GENMODEL_WILDCARD', 'GENPDF_FILE_WILDCARD'):
        with pytest.raises(SNANAError, match='no files match'):
            build_config({**settings, 'ransystpar': {'IDSURVEY': 1, name: str(tmp_path/'missing-*')}})
