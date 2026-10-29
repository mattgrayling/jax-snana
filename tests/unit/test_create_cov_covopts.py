"""create_covariance config: covopt parsing, filter matching, cosmology.

All three are small but load-bearing. A covopt string selects which systematics
enter a covariance matrix, and a divergent reading silently builds it from the
wrong set rather than failing; the covopt cases below are the documented examples
from create_covariance.py's own help menu (:460). The cosmology decides the bin
redshifts of a rebinned Hubble diagram.
"""

import os

import numpy as np
import pandas as pd
import pytest

from jax_create_cov import apply_filter, load_config, parse_covopt
from jax_create_cov.covariance import covsys_from_covfile
from jax_create_cov.covariance import hubble_diagram_from
from jax_create_cov.covariance import build_covariance
from jax_snlc_core.logging import SNANAError


def test_binned_covariance_row_order():
    base = pd.DataFrame({"ROW": [1, 2], "zHD": [.1, .2], "MU": [38., 40.],
                         "MUREF": [38., 40.], "MUERR": [.1, .1]})
    variant = base.iloc[::-1].reset_index(drop=True).copy()
    variant["MU"] += [.2, .1]
    result = build_covariance(
        {"FITOPT000_MUOPT000": hubble_diagram_from(base, "binned"),
         "FITOPT000_MUOPT001": hubble_diagram_from(variant, "binned")},
        {0: ("DEFAULT", 1.)}, {0: ("DEFAULT", 1.), 1: ("SHIFT", 1.)}, [])
    np.testing.assert_allclose(result.covariances["ALL"]["matrix"], [[.01, .02], [.02, .04]])


@pytest.mark.parametrize("text,label,fitopt,muopt,scale", [
    ("[NOSYS]  [=DEFAULT,=DEFAULT]", "NOSYS", "=DEFAULT", "=DEFAULT", 1.0),
    ("[PS1CAL] [+PS1CAL,=DEFAULT]", "PS1CAL", "+PS1CAL", "=DEFAULT", 1.0),
    ("[NOCAL]  [-CAL,=DEFAULT]", "NOCAL", "-CAL", "=DEFAULT", 1.0),
    ("[NOCALONLY] [-CAL,]", "NOCALONLY", "-CAL", "", 1.0),
    ("[noFrag] [,-SCAT]", "noFrag", "", "-SCAT", 1.0),
    ("[ALL]    [,]", "ALL", "", "", 1.0),
])
def test_parses_documented_covopts(text, label, fitopt, muopt, scale):
    covopt = parse_covopt(text)
    assert (covopt.label, covopt.fitopt_filter, covopt.muopt_filter) == \
        (label, fitopt, muopt)
    assert covopt.scale == scale


def test_covopt_scale_is_squared_in_covariance(tmp_path):
    """'[...,1.2]' means errSys *= 1.2, so covSys *= 1.44 (:1436).

    Asserts the covariance, not just the parsed number: the earlier version of
    this test checked `.scale == 1.2` and passed for months while the square was
    never applied anywhere.
    """

    from jax_create_cov.covariance import Contribution, covsys_from_covopt

    assert parse_covopt("[SCAT] [=DEFAULT,+SCAT,1.2]").scale == pytest.approx(1.2)

    base = pd.DataFrame({"MU": [38.0, 39.0], "MUREF": [38.0, 39.0]})
    contribs = [Contribution("DEFAULT", "SCAT", diff=np.array([0.1, 0.2]))]
    plain = covsys_from_covopt(parse_covopt("[S] [,+SCAT]"), contribs, base)
    scaled = covsys_from_covopt(parse_covopt("[S] [,+SCAT,1.2]"), contribs, base)

    np.testing.assert_allclose(scaled, plain * 1.44)


@pytest.mark.parametrize("text", [
    "[ONLYONE]",                        # missing the second bracketed term
    "[A] [B]",                          # second term needs two comma-separated items
    "[A] [x,y,z,w]",                    # ... and no more than three
    "[A] [x,y,notanumber]",             # third item must be a scale
    "A [x,y]",                          # first term unbracketed
])
def test_malformed_covopts_are_rejected(text):
    with pytest.raises(SNANAError):
        parse_covopt(text)


@pytest.mark.parametrize("label,pattern,expected", [
    ("DEFAULT", "=DEFAULT", True),
    ("DEFAULTX", "=DEFAULT", False),
    ("PS1CAL", "+CAL", True),
    ("SCATTER", "+CAL", False),
    ("PS1CAL", "-CAL", False),
    ("SCATTER", "-CAL", True),
    ("ANYTHING", "", True),
])
def test_filter_matching(label, pattern, expected):
    assert apply_filter(label, pattern) is expected


def test_filter_matching_is_case_insensitive():
    assert apply_filter("ps1cal", "+CAL")
    assert apply_filter("PS1CAL", "+cal")


def test_a_grid_without_a_cosmology_aborts():
    """SNANA substitutes FlatLambdaCDM(H0=70, Om0=0.315) here (:982); bin
    redshifts come from inverting this grid, so jax asks for the cosmology."""
    from jax_create_cov.covariance import reference_grid

    with pytest.raises(SNANAError, match="set COSMOLOGY"):
        reference_grid(None)


def test_the_recorded_cosmology_and_the_input_must_agree():
    """BBC records the biasCor simulation's cosmology; COSMOLOGY stands in
    for an output without it, and may restate it but not contradict it."""
    from jax_create_cov.config import Cosmology
    from jax_create_cov.covariance import reference_cosmology

    header = {"SIM_OMEGA_MATTER(BIASCOR)": "0.2812",
              "SIM_OMEGA_LAMBDA(BIASCOR)": "0.7188",
              "SIM_w0_LAMBDA(BIASCOR)": "-1.0000", "SIM_wa_LAMBDA(BIASCOR)": "0",
              "SIM_MUSHIFT(BIASCOR)": "0.01:0.03"}
    recorded = Cosmology(0.2812, 0.7188, mushift=(0.01, 0.03))
    given = Cosmology(0.3, 0.7)

    assert reference_cosmology(None, header) == recorded
    assert reference_cosmology(recorded, header) == recorded
    assert reference_cosmology(given, {}) == given
    with pytest.raises(SNANAError, match="disagrees"):
        reference_cosmology(given, header)


def test_cosmology_block_is_read_from_the_config(tmp_path):
    config = tmp_path / "input.yml"
    config.write_text(
        "INPUT_DIR: bbc\nVERSION: V\n"
        "COSMOLOGY:\n  OMEGA_MATTER: 0.28\n  OMEGA_LAMBDA: 0.72\n"
        "  w0_LAMBDA: -0.9\n  wa_LAMBDA: 0.3\n"
    )
    cosmology = load_config(config).cosmology
    assert (cosmology.omega_matter, cosmology.omega_lambda) == (0.28, 0.72)
    assert (cosmology.w0, cosmology.wa) == (-0.9, 0.3)


def test_incomplete_cosmology_block_is_rejected(tmp_path):
    config = tmp_path / "input.yml"
    config.write_text("INPUT_DIR: bbc\nVERSION: V\n"
                      "COSMOLOGY:\n  OMEGA_MATTER: 0.3\n")
    with pytest.raises(SNANAError, match="omega_lambda"):
        load_config(config)


def test_extra_covs_weight_scales_the_covariance(tmp_path):
    """SNANA accepts this weight and never applies it.

    Deliberately not a parity test: applying it is the divergence, so SNANA and
    jax disagree for any weight other than 1.
    """

    from jax_create_cov.covariance import contributions

    cov_file = tmp_path / "extra.dat"
    cov_file.write_text("CID1 IDSURVEY1 CID2 IDSURVEY2 MU_COV\n"
                        "1 10 1 10 4.0e-04\n2 10 2 10 4.0e-04\n")
    base = pd.DataFrame({"CID": ["1", "2"], "IDSURVEY": [10, 10],
                         "MU": [38.0, 39.0], "MUREF": [38.0, 39.0],
                         "MUERR": [0.1, 0.1]})
    scales = {0: ("DEFAULT", 1.0)}

    def diagonal(weight):
        result = contributions({}, scales, scales,
                               [("VPEC_EXTRA", weight, str(cov_file))], base)
        return np.diagonal(result[0].matrix())

    np.testing.assert_allclose(diagonal(1.0), [4.0e-04, 4.0e-04])
    np.testing.assert_allclose(diagonal(2.0), [1.6e-03, 1.6e-03])


def test_an_npz_extra_cov_reads_as_the_text_form_with_both_halves(tmp_path):
    """The .npz form's upper triangle is the whole symmetric covariance, placed
    in diagram order; events the diagram lacks are skipped. The text form
    listing both halves gives the same matrix."""
    base = pd.DataFrame({"CID": ["1", "2", "3"], "IDSURVEY": [10, 10, 10]})
    npz = tmp_path / "extra.npz"
    # events in file order 3, 1, 9; upper triangle row by row
    np.savez(npz, CID=np.array([3, 1, 9]), IDSURVEY=np.array([10, 10, 10]),
             MU_COV=np.array([4e-4, 1e-4, 7e-4, 5e-4, 2e-4, 6e-4]))
    text = tmp_path / "extra.dat"
    text.write_text("CID1 IDSURVEY1 CID2 IDSURVEY2 MU_COV\n"
                    "3 10 3 10 4e-4\n3 10 1 10 1e-4\n1 10 3 10 1e-4\n"
                    "1 10 1 10 5e-4\n")
    from_npz = covsys_from_covfile(base, str(npz))
    np.testing.assert_allclose(from_npz, covsys_from_covfile(base, str(text)))
    np.testing.assert_allclose(from_npz, from_npz.T)


def test_muopt_selection_keeps_one_muopt_as_the_reference(tmp_path):
    """SNANA aborts here, so this is jax-only.

    Selecting a MUOPT makes it the reference the FITOPTs are differenced against,
    which is what relabelling it DEFAULT was reaching for.
    """
    from jax_create_cov import create_covariance

    fixture = os.path.join("tests", "fixtures", "snana", "create_cov")
    config = load_config(os.path.join(fixture, "input.yml"))
    config.muopt = 1
    # covopts selecting the MUOPT this filters out have nothing left to match
    config.covopts = [c for c in config.covopts
                      if "SCAT" not in c.muopt_filter]
    config.outdir = tmp_path
    result = create_covariance(config)

    assert list(result.covariances) == [
        "ALL", "NOSYS", "CAL", "NOCAL", "NOCALONLY"]
    assert (tmp_path / "hubble_diagram.txt").is_file()


def test_muopt_selection_rejects_a_covopt_that_needs_a_dropped_muopt(tmp_path):
    from jax_create_cov import create_covariance

    fixture = os.path.join("tests", "fixtures", "snana", "create_cov")
    config = load_config(os.path.join(fixture, "input.yml"))
    config.muopt = 1
    config.outdir = tmp_path
    with pytest.raises(SNANAError, match="SCAT"):
        create_covariance(config)


def test_label_cov_rows_writes_every_element(tmp_path):
    """SNANA writes only the row count here: its
    labelled branch flushes on a full 50,000-element batch and never afterwards."""
    import gzip

    from jax_create_cov import create_covariance

    fixture = os.path.join("tests", "fixtures", "snana", "create_cov")
    config = load_config(os.path.join(fixture, "input.yml"))
    config.mode = "unbinned"
    config.label_cov_rows = True
    config.write_mask_cov = 1
    config.outdir = tmp_path
    create_covariance(config)

    text = gzip.decompress(
        (tmp_path / "covsys_000.txt.gz").read_bytes()).decode().splitlines()
    size = int(text[0])
    assert len(text) == 1 + size * size, "not every element was written"
    assert "CID,IDSURVEY,zHD,index" in text[1]
    assert text[1].rstrip().endswith("diag"), "first element is on the diagonal"


def test_cosmology_mushift_is_read_and_reported(tmp_path):
    """MUSHIFT is one of the five keys a cosmology fitter needs from INFO.YML."""
    config = tmp_path / "input.yml"
    config.write_text(
        "INPUT_DIR: bbc\nVERSION: V\n"
        "COSMOLOGY:\n  OMEGA_MATTER: 0.3\n  OMEGA_LAMBDA: 0.7\n  MUSHIFT: 0.02\n"
    )
    assert load_config(config).cosmology.mushift == (0.02, 0.02)


@pytest.mark.parametrize("argv,attr,expected", [
    (["--write_mask_cov", "7"], "write_mask_cov", 7),
    (["--write_format_cov", "text"], "write_format_cov", "text"),
    (["--nosys"], "nosys", True),
    (["--label_cov_rows"], "label_cov_rows", True),
    (["--systematic_HD"], "systematic_hd", True),
    (["-u"], "mode", "unbinned"),
    (["-s"], "mode", "unbinned"),
    (["--nbin_x1", "2", "--nbin_c", "2"], "mode", "rebin"),
    # 0 == False, so a truth test would silently drop this one
    (["--muopt", "0"], "muopt", 0),
    (["--muopt", "2"], "muopt", 2),
])
def test_cli_flags_reach_the_config(tmp_path, argv, attr, expected):
    from create_covariance import config_from_args

    config_file = tmp_path / "input.yml"
    config_file.write_text("INPUT_DIR: bbc\nVERSION: V\n")
    config = config_from_args([str(config_file), "-o", str(tmp_path)] + argv)

    assert getattr(config, attr) == expected


def test_input_dir_and_version_may_come_from_the_command_line(tmp_path):
    """The YAML need not name INPUT_DIR and VERSION when -i and -v do; the
    parse aborts naming whichever is still missing after the flags are
    merged."""
    from create_covariance import config_from_args

    config_file = tmp_path / "input.yml"
    config_file.write_text("WRITE_MASK_COV: 7\n")
    config = config_from_args([str(config_file), "-i", "bbc", "-v", "V",
                               "-o", str(tmp_path)])
    assert (str(config.input_dir), config.version) == ("bbc", "V")
    with pytest.raises(SNANAError, match="no VERSION, and none given with -v"):
        config_from_args([str(config_file), "-i", "bbc", "-o", str(tmp_path)])


def test_obsolete_muopt_scales_key_is_rejected(tmp_path):
    """SNANA exits with the same advice; silently ignoring it would drop the
    user's MUOPT systematic scales."""
    config = tmp_path / "input.yml"
    config.write_text("INPUT_DIR: bbc\nVERSION: V\nMUOPT_SCALES:\n  SCAT: 1.2\n")
    with pytest.raises(SNANAError, match="SYS_SCALE_FILE"):
        load_config(config)


def test_unrecognised_keys_are_reported(tmp_path, caplog):
    import logging

    config = tmp_path / "input.yml"
    config.write_text("INPUT_DIR: bbc\nVERSION: V\nNOT_A_KEY: 1\nMETHOD: BBC\n")
    with caplog.at_level(logging.WARNING):
        load_config(config)
    assert "NOT_A_KEY" in caplog.text and "METHOD" in caplog.text


def test_conditioning_check_is_skipped_above_the_size_guard():
    """np.linalg.cond is an SVD; SNANA skips it for large HDs and so do we."""
    from jax_create_cov.covariance import cov_invert

    n = 6
    # invertible, but with eigenvalues spanning 1e36 -- far past 1/eps
    cov = np.diag([1e18] + [0.0] * (n - 1))
    muerr = np.full(n, 1e-9)

    with pytest.raises(SNANAError, match="ill-conditioned"):
        cov_invert(cov, muerr, mxsize_test_posdef=n)
    # above the guard the same matrix passes through unchecked
    assert cov_invert(cov, muerr, mxsize_test_posdef=n - 1).shape == (n, n)


def test_environment_variables_in_paths_are_expanded(tmp_path):
    """SNANA expandvars INPUT_DIR, OUTDIR and SYS_SCALE_FILE; real configs are
    written in terms of $PIPPIN_OUTPUT, $DES_ROOT and the like."""
    (tmp_path / "grid").mkdir()
    config = tmp_path / "input.yml"
    config.write_text("INPUT_DIR: $CREATE_COV_TEST_ROOT/grid\nVERSION: V\n"
                      "OUTDIR: $CREATE_COV_TEST_ROOT/out\n")

    previous = os.environ.get("CREATE_COV_TEST_ROOT")
    os.environ["CREATE_COV_TEST_ROOT"] = str(tmp_path)
    try:
        parsed = load_config(config)
        assert parsed.input_dir == tmp_path / "grid"
        assert parsed.outdir == tmp_path / "out"
        assert "$" not in str(parsed.data_dir)
    finally:
        if previous is None:
            del os.environ["CREATE_COV_TEST_ROOT"]
        else:
            os.environ["CREATE_COV_TEST_ROOT"] = previous


def test_a_row_without_a_finite_error_names_its_bin():
    """A BBC bin with no events carries MUDIFERR 999 (MUDIFERR_EMPTY,
    SALT2mu.c:501), read as NaN, and one too thin to fit leaves NaN; the
    inversion cannot take either, so the abort names the bins rather than
    letting numpy fail on the matrix."""

    from jax_create_cov.covariance import require_finite_errors

    diagram = hubble_diagram_from(pd.DataFrame({
        "zHD": [0.05, 0.15, 0.25, 0.35], "MUREF": [36.0, 39.0, 40.0, 41.0],
        "MUDIF": [0.0, 0.0, 0.0, 0.0], "MUDIFERR": [0.1, 999.0, np.nan, 0.1],
        "NFIT": [40, 0, 3, 50], "ROW": [1, 2, 3, 4]}), "binned")
    with pytest.raises(SNANAError, match=r"2 of 4 .* zHD=0.1500 \(NFIT=0\), "
                                          r"zHD=0.2500 \(NFIT=3\)"):
        require_finite_errors(diagram)
    require_finite_errors(diagram[np.isfinite(diagram["MUERR"])])


def test_non_positive_definite_inverse_is_reported(caplog):
    """SNANA warns rather than aborting here; silently skipping the check would
    hand a cosmology fit a matrix nobody vouched for."""
    import logging

    from jax_create_cov.covariance import cov_invert

    n = 4
    # symmetric, invertible, and indefinite: one negative eigenvalue
    cov = np.diag([1.0, 1.0, 1.0, -3.0])
    muerr = np.full(n, 1.0)

    with caplog.at_level(logging.WARNING):
        cov_invert(cov, muerr, label="ALL")
    assert "positive definite" in caplog.text


def test_well_behaved_covariance_warns_about_nothing(caplog):
    import logging

    from jax_create_cov.covariance import cov_invert

    cov = np.zeros((4, 4))
    with caplog.at_level(logging.WARNING):
        cov_invert(cov, np.full(4, 0.1), label="ALL")
    assert "not unitary" not in caplog.text
    assert "positive definite" not in caplog.text


@pytest.mark.parametrize("selection, warn", [
    ("[NOSYS] [=DEFAULT,=DEFAULT]", False),
    ("[CAL] [+CAL,=DEFAULT]", True),
    ("[DISABLED] [+CAL,=DEFAULT,0]", False),
])
def test_zero_covariance_warns_only_for_an_active_systematic(selection, warn, caplog):
    from jax_create_cov.covariance import Contribution, covsys_from_covopt

    contributions = [
        Contribution("DEFAULT", "DEFAULT", np.zeros(2)),
        Contribution("CAL", "DEFAULT", np.zeros(2)),
    ]
    cov = covsys_from_covopt(parse_covopt(selection), contributions, pd.DataFrame())
    assert not np.any(cov)
    assert ("every element" in caplog.text) == warn


@pytest.mark.parametrize("factorized", [False, True])
@pytest.mark.parametrize("nosys,options,label", [
    (False, ["[ALL] [=DEFAULT,=DEFAULT]"], "ALL"),
    (True, ["[=DEFAULT] [,]"], "=DEFAULT"),
    (False, ["[CAL] [,+CAL]", "[CAL] [,-CAL]"], "CAL"),
])
def test_duplicate_covopt_labels_fail_before_building(factorized, nosys, options, label):
    # Empty inputs deliberately verify rejection before any HD or covariance work.
    with pytest.raises(SNANAError, match=f"duplicate COVOPT label {label!r}"):
        build_covariance({}, {}, {}, [parse_covopt(s) for s in options],
                         factorized=factorized, nosys=nosys)


@pytest.mark.parametrize("settings", [{}, {"radiation": True}, {"radiation": False},
                                      {"RADIATION": False}])
def test_reference_distances_follow_radiation_setting(settings):
    """Check the public configuration path against independent quadrature."""
    from scipy.integrate import quad
    from jax_create_cov import compute_covariance

    result = compute_covariance(
        diagrams={"FITOPT000_MUOPT000": pd.DataFrame({
            "ROW": ["A"], "zHD": [1.0], "MU": [44.0],
            "MUREF": [44.0], "MUERR": [0.1]})},
        fitopt_scales={0: ("DEFAULT", 1.0)},
        muopt_scales={0: ("DEFAULT", 1.0)},
        cosmology={"omega_matter": .315, "omega_lambda": .685},
        isdata_real=0, **settings)
    z, mu = (np.asarray(a) for a in result.mu_grid)
    omega_r = 9e-5 if settings.get("radiation", settings.get("RADIATION", True)) else 0.
    for i in (0, 1500, 3000, 3999):
        integral = quad(lambda x: 1 / np.sqrt(
            (1 - omega_r) * (.315 * (1 + x)**3 + .685)
            + omega_r * (1 + x)**4), 0, z[i], epsabs=1e-12)[0]
        expected = 25 + 5 * np.log10((299792.458 / 70) * (1 + z[i]) * integral)
        np.testing.assert_allclose(mu[i], expected, rtol=0, atol=1e-10)


@pytest.mark.parametrize('axis', ['FITOPT', 'MUOPT'])
@pytest.mark.parametrize('factorized', [False, True])
def test_shared_systematic_labels_keep_last_contribution(axis, factorized):
    """SNANA overwrites earlier contributions with the same FITOPT|MUOPT label."""
    base = pd.DataFrame({'ROW': [1, 2], 'zHD': [.1, .2], 'MU': [38., 40.],
                         'MUREF': [38., 40.], 'MUERR': [.1, .1]})
    diagrams = {}
    for number, shift in enumerate([0., .1, .2]):
        name = (f'FITOPT{number:03d}_MUOPT000' if axis == 'FITOPT'
                else f'FITOPT000_MUOPT{number:03d}')
        diagrams[name] = hubble_diagram_from(base.assign(MU=base.MU + shift), 'binned')
    reference = {0: ('DEFAULT', 1.)}
    variants = {**reference, 1: ('CAL', 1.), 2: ('CAL', 1.)}
    selection = '[CAL] [=CAL,]' if axis == 'FITOPT' else '[CAL] [,=CAL]'
    result = build_covariance(
        diagrams, variants if axis == 'FITOPT' else reference,
        variants if axis == 'MUOPT' else reference,
        [parse_covopt(selection), parse_covopt('[STAT] [=DEFAULT,=DEFAULT]')],
        factorized=factorized)
    for label in ['ALL', 'CAL', 'STAT']:
        covariance = result.covariances[label]
        matrix = (covariance['U'] @ covariance['U'].T if factorized
                  else covariance['matrix'])
        np.testing.assert_allclose(matrix, 0. if label == 'STAT' else .04, atol=1e-14)


@pytest.mark.parametrize('factorized', [False, True])
def test_external_covariances_with_shared_labels_replace_previous(tmp_path, factorized):
    """The last external matrix replaces earlier matrices and a same-labelled MUOPT."""
    path = tmp_path / 'extra.dat'
    path.write_text('CID1 IDSURVEY1 CID2 IDSURVEY2 MU_COV\n1 10 1 10 .03\n')
    base = pd.DataFrame({'CID': ['1'], 'IDSURVEY': [10], 'zHD': [.1],
                         'MU': [38.], 'MUREF': [38.], 'MUERR': [.1]})
    result = build_covariance(
        {'FITOPT000_MUOPT000': base, 'FITOPT000_MUOPT001': base.assign(MU=38.1)},
        {0: ('DEFAULT', 1.)}, {0: ('DEFAULT', 1.), 1: ('CAL', 1.)},
        [parse_covopt('[CAL] [,=CAL]')], factorized=factorized,
        extra_covs=[('CAL', 1., str(path)), ('CAL', 2., str(path))])
    for label in ['ALL', 'CAL']:
        np.testing.assert_allclose(result.covariances[label]['matrix'], [[.12]])


@pytest.mark.parametrize('source', ['FITOPT', 'MUOPT', 'EXTRA_COVS'])
def test_default_label_replaces_reference_contribution(tmp_path, source):
    """DEFAULT is an ordinary label; a later contribution overwrites its reference."""
    from jax_create_cov.covariance import contributions

    base = pd.DataFrame({'CID': ['1'], 'IDSURVEY': [10], 'MU': [38.], 'MUREF': [38.]})
    diagrams = {'FITOPT000_MUOPT000': base}
    reference = {0: ('DEFAULT', 1.)}
    fitopts, muopts, extra = dict(reference), dict(reference), []
    if source == 'FITOPT':
        diagrams['FITOPT001_MUOPT000'] = base.assign(MU=38.2)
        fitopts[1] = ('DEFAULT', 1.)
    elif source == 'MUOPT':
        diagrams['FITOPT000_MUOPT001'] = base.assign(MU=38.2)
        muopts[1] = ('DEFAULT', 1.)
    else:
        path = tmp_path / 'extra.dat'
        path.write_text('CID1 IDSURVEY1 CID2 IDSURVEY2 MU_COV\n1 10 1 10 .04\n')
        extra = [('DEFAULT', 1., str(path))]
    result = contributions(diagrams, fitopts, muopts, extra, base=base)
    assert len(result) == 1
    np.testing.assert_allclose(result[0].matrix(), [[.04]])


@pytest.mark.parametrize("rows", [[1, 3], [2, 1]])
@pytest.mark.parametrize("factorized", [False, True])
def test_binned_covariance_rejects_mismatched_bin_labels(rows, factorized):
    """Equal row counts cannot make different redshift bins comparable."""
    from jax_create_cov import compute_covariance

    base = pd.DataFrame({"ROW": [1, 2], "zHD": [.1, .2], "MU": [38., 40.],
                         "MUREF": [38., 40.], "MUERR": [.1, .1]})
    variant = base.assign(ROW=rows, MU=[38.01, 40.5])
    with pytest.raises(SNANAError, match="different ROW bin labels"):
        compute_covariance(
            diagrams={"FITOPT000_MUOPT000": base, "FITOPT001_MUOPT000": variant},
            fitopt_scales={0: ("DEFAULT", 1.), 1: ("CAL", 1.)},
            muopt_scales={0: ("DEFAULT", 1.)}, write_mask_cov=8 if factorized else 1)


def test_binned_diagram_requires_row_labels():
    """Missing bin identities must not be replaced by positional labels."""
    diagram = pd.DataFrame({"zHD": [.1], "MU": [38.], "MUERR": [.1]})
    with pytest.raises(SNANAError, match="ROW"):
        hubble_diagram_from(diagram, "binned")
