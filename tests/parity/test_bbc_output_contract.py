"""Every field jax writes means what SNANA's means, across every output mode.

The BBC writers produced plausible-looking files that had never been diffed
against SALT2mu.exe's. Eight defects survived that way: a .YAML key SNANA does not
use, three counts that tallied the wrong thing, warning lines computed but never
written to the files a reader sees, passthrough columns reformatted so the input's
stated precision was lost, an extra column under MU-biasCor, and the biasCor
summary absent on the 1D path.

Six of the eight were invisible to a single-config check: they only appear under a
1D biasCor, a CC prior, BayeSN, blinding, or no biasCor at all. So this is
parametrised over the modes rather than run on the 5D SALT case alone -- including
two defects introduced *while* fixing the others, both of which matched the 5D
reference by coincidence.

References and their exact SALT2mu.exe commands are in the fixture README.
"""


import numpy as np
import pytest
import yaml

from jax_salt2mu.config import parse_bbc_input
from jax_salt2mu.output import write_cov, write_fitres, write_m0_csv, write_m0dif, write_yaml
from jax_salt2mu.run import run_bbc, run_bbc_fit
from jax_snlc_core.snana_files import open_text, read_fitres

_REF = "tests/fixtures/snana/bbc_output_contract"
_BC = "tests/fixtures/snana/bbc_des_beams"
_PZ = "tests/fixtures/snana/bbc_des_beams"
_BS = "tests/fixtures/snana/bbc_des_bayesn_data"
_BSBC = "tests/fixtures/snana/bbc_des_bayesn_biascor"

_COMMON = ["nzbin=10", "zmin=0.1", "zmax=1.0", "u1=1", "u2=1", "p9=0.7", "p11=-1"]
# The configs that inflate muerr least -- no biasCor, and 1D -- leave the low-z bins
# with a per-event scatter under SNANA's hardcoded 0.1 mag, and three such bins trigger
# a refit (NFIT_EXEC=3) that jax does not implement. Starting at z=0.25 drops those
# bins; the file format under test is unaffected.
_COMMON_HIZ = ["nzbin=10", "zmin=0.25", "zmax=1.0", "u1=1", "u2=1", "p9=0.7", "p11=-1"]
_5D = ["opt_biascor=112", "min_per_cell_biascor=3", "snrmin_sigint_biascor=60",
       "surveygroup_biascor=DES(zbin=0.15:x1bin=2.0:cbin=0.2)"]

# name -> overrides. Each exercises a branch the others do not.
CONFIGS = {
    "salt_5d": _COMMON + _5D + [
        f"datafile={_BC}/DATA.FITRES", f"simfile_biascor={_BC}/BIASCOR.FITRES",
        "blindflag=0"],
    "salt_nobiascor": _COMMON_HIZ + [f"datafile={_BC}/DATA.FITRES", "blindflag=0"],
    # no blindflag key: the headers report the flag SALT2mu resolves at setup,
    # not the parsed default
    "salt_default_blindflag": _COMMON_HIZ + [f"datafile={_BC}/DATA.FITRES"],
    # typeIa_ccprior=999: CONTAM_DATA has no TYPE column, so it reads -9 == the
    # default and force_probcc0 would pin every pIa to 1, leaving the CC term
    # weightless (SALT2mu.c:22789)
    "salt_ccprior": _COMMON + _5D + [
        f"datafile={_BC}/CONTAM_DATA.FITRES",
        f"simfile_biascor={_BC}/BIASCOR.FITRES",
        f"simfile_ccprior={_BC}/CCPRIOR.FITRES",
        "p13=1", "opt_ccprior=5", "varname_pIa=PROBIA",
        "typeIa_ccprior=999", "blindflag=0"],
    # force_realdata activates blinding, which sims are exempt from by default
    "salt_blinded": _COMMON + _5D + [
        f"datafile={_BC}/DATA.FITRES", f"simfile_biascor={_BC}/BIASCOR.FITRES",
        "force_realdata=1", "blindflag=2"],
    # 1DZAVG|SAMPLE with a photo-z split: two IDSAMPLEs on the 1D bias path
    "salt_1d_split": _COMMON_HIZ + [
        f"datafile={_PZ}/DATA_PHOTOZ.FITRES",
        "opt_biascor=65", "zspec_errmax_idsample=0.05", "blindflag=0"],
    # MU-biasCor: no per-parameter bias columns, and no alpha/beta grid note
    "bayesn": ["model_lcfit=BAYESN", "opt_biascor=144", "sigint_biascor=0.10",
               "nzbin=4", "zmin=0.05", "zmax=0.21", "nbin_logmass=1",
               "snrmin_sigint_biascor=6.0", "p9=0.7", "p11=-1", "blindflag=0",
               f"datafile={_BS}/FITRES.TEXT",
               f"simfile_biascor={_BSBC}/FITRES.TEXT.gz"],
}

# Lines carrying each program's own identity or timing, which cannot match:
# the version stamp, the program name, the error method (SNANA runs MINUIT, jax
# takes the Hessian from Newton-LM), MINUIT's call count, and wall time.
_OWN_IDENTITY = ("SNANA_VERSION", "PROGRAM", "MIGRAD", "HESSIAN", "NCALL_FCN", "CPU")

CHI2GRID_NBIN = 5   # write_chi2grid=5 in the reference commands



def _header(path):
    """Provenance, notes and warnings, minus each program's own identity.

    Stops at "SALT2mu fit results:", where the formats deliberately part: SNANA
    writes that block after VARNAMES (write_fitres_driver:23145), jax before.
    """
    out = []
    for line in open_text(path):
        if line.startswith("VARNAMES:") or "SALT2mu fit results" in line:
            break
        if not any(k in line for k in _OWN_IDENTITY):
            out.append(line.rstrip("\n").rstrip())
    return out


def _yaml_fields(path):
    """SNANA fields; timing and the separately checked JAX diagnostics differ."""
    jax_diagnostics = {'FIT_DIAGNOSTICS', 'max_gradient', 'max_projected_gradient'}
    return {k.strip(): v.strip()
            for k, v in (ln.split(":", 1) for ln in open_text(path)
                         if ":" in ln and not ln.startswith("  -"))
            if k.strip() != "CPU_MINUTES" and k.strip() not in jax_diagnostics}


def _biascor_summary(path):
    """The NSIM/sigint(<N>D-BIASCOR) lines: present, absent, and labelled per mode."""
    return [ln.rstrip() for ln in open_text(path)
            if "NSIM(" in ln or "sigint(" in ln and "BIASCOR" in ln]


@pytest.fixture(scope="module", params=sorted(CONFIGS))
def written(bbc_biascor_variants, request, tmp_path_factory):
    """jax's outputs for one config, beside SNANA's committed reference."""
    name = request.param
    tmp = tmp_path_factory.mktemp(name)
    prefix = str(tmp / "JAX")
    # run_bbc_fit writes the CHI2GRID itself, from config.prefix
    overrides = CONFIGS[name]
    if name == "salt_1d_split":
        overrides = overrides + [f"simfile_biascor={bbc_biascor_variants}/BIASCOR_PHOTOZ.FITRES"]
    config = parse_bbc_input(None, overrides=overrides + [
        "radiation=0", f"prefix={prefix}", f"write_chi2grid={CHI2GRID_NBIN}"])
    result, output, zbins, tv, accept = run_bbc_fit(config)
    config = output.config    # resolved, and blinded where asked
    passthrough = {n: np.asarray(a)[accept] for n, a in tv.passthrough.items()}
    write_m0dif(f"{prefix}.M0DIF", zbins, output, result, config,
                warnings=output.warnings)
    write_fitres(f"{prefix}.FITRES", output, zbins, result, config,
                 passthrough=passthrough, warnings=output.warnings)
    write_yaml(f"{prefix}.YAML", output.yaml, result, config)
    write_cov(f"{prefix}.COV", result.cov_m0)
    write_m0_csv(f"{prefix}.CSV", zbins, output, config)
    return name, prefix, output


# Columns built from the fitted sigint/muCOVscale, which amplify the ~1e-5
# optimiser-level difference rather than tracking it. Measured max on the 5D
# config: MUERR 2.3e-4. Everything else agrees to its last printed digit.
_COMPOUNDING = frozenset({
    "MUERR", "MUERR_RAW", "MUERR_RENORM", "MUPULL", "CHI2", "CHI2_BEAMS",
    "PROBCC_BEAMS", "biasCorErr_mu", "biasCor_muCOVSCALE", "biasCor_muCOVADD",
    "M0DIFERR",
})
# Below this a probability is dead under both hypotheses, and a relative bound means
# nothing: PROBCC_BEAMS reaches 1e-308, where 1e-5 in mures is 1% once exponentiated.
_NEGLIGIBLE = 1.0e-12


# A column passes if its bulk matches and the exceptions are rare and small. Set from
# the measured SALT2mu integrator glitch: 2 rows of 1473 (0.14%) at 6.2e-4 mag.
# 6e-3 rather than the 6.2e-4 seen in MU/MUMODEL: the same glitch reaches 6.0e-3 in
# MUPULL, which divides the residual by muerr (0.11 on the worst row).
_OUTLIER_FRAC = 0.005
_OUTLIER_MAX = 6.0e-3


def _column_tol(want, col):
    """The per-token tolerance _same_number would apply, for the median check."""
    decimals = _decimals(want)
    ulp = 10.0 ** -decimals if decimals > 0 else 0.0
    return max(2e-3 * abs(float(want)), 2.5 * ulp) if col in _COMPOUNDING else 2.5 * ulp


def _decimals(want):
    """Places in the last printed digit, for %f and %e alike. Counting the characters
    past the "." makes 1.48841590e-02 look like 12 decimals rather than 10."""
    mant, _, exp = want.lower().partition("e")
    d = len(mant.partition(".")[2])
    return d - int(exp) if exp else d


def _same_number(got, want, col=""):
    """Equal to the reference's own last printed place.

    Each appended column has its own width (write_fitres_line_append,
    SALT2mu.c:25281): MUPULL is %6.3f, MU is %8.5f. A single absolute tolerance
    is either too loose for one or too tight for the other, so it comes from the
    reference; _COMPOUNDING columns get a relative bound instead.
    """
    decimals = _decimals(want)
    if decimals <= 0:
        return got.strip() == want.strip()
    a, b = float(got), float(want)
    ulp = 10.0 ** -decimals
    if col in _COMPOUNDING:
        return abs(a - b) <= max(2e-3 * abs(b), 2.5 * ulp, _NEGLIGIBLE)
    # 2 ulp is the measured maximum (MU, %8.5f, one row of 334); 1 would assert
    # closer agreement than the two programs actually have
    return abs(a - b) <= 2.5 * ulp


def _same_line(a, b, tol=5e-4):
    """Token-wise: numbers within tolerance, everything else exact.

    Header lines carry fitted values (`FIT alpha0 : 0.1642 +- 0.0082`), which
    differ from SNANA in the last printed digit for the same reason every other
    value does. Comparing them as text would make this a duplicate of the
    per-feature parity tests, and a flaky one. The tolerance also covers the
    blinded case, where MINUIT's parabolic error is numerically unstable and jax's
    is not (3e-4 on alpha0).
    """
    ta, tb = a.split(), b.split()
    if len(ta) != len(tb):
        return False
    for x, y in zip(ta, tb):
        if x == y:
            continue
        try:
            if abs(float(x) - float(y)) <= tol:
                continue
        except ValueError:
            pass
        return False
    return True


@pytest.mark.parametrize("ext", ["M0DIF", "FITRES"])
def test_header_matches_salt2mu(written, ext):
    """Line for line: every provenance, note and warning line, in order."""
    name, prefix, _ = written
    mine, ref = _header(f"{prefix}.{ext}"), _header(f"{_REF}/{name}.{ext}")
    if name == "bayesn":
        # JAX carries the generating cosmology; SALT2mu does not write it.
        cosmology = [line for line in mine if line.startswith("# SIM_")]
        assert len(cosmology) == 5
        assert {line.split()[1]: float(line.split()[2]) for line in cosmology} == {
            "SIM_OMEGA_MATTER(BIASCOR):": 0.3,
            "SIM_OMEGA_LAMBDA(BIASCOR):": 0.7,
            "SIM_w0_LAMBDA(BIASCOR):": -1.0,
            "SIM_wa_LAMBDA(BIASCOR):": 0.0,
            "SIM_MUSHIFT(BIASCOR):": 0.0,
        }
        mine = [line for line in mine if line not in cosmology]
    assert len(mine) == len(ref), f"{name}.{ext}: {len(mine)} lines vs {len(ref)}"
    bad = [(m, r) for m, r in zip(mine, ref) if not _same_line(m, r)]
    assert not bad, f"{name}.{ext}: {bad[:3]}"


def test_yaml_matches_salt2mu(written):
    name, prefix, _ = written
    assert _yaml_fields(f"{prefix}.YAML") == _yaml_fields(f"{_REF}/{name}.YAML")
    with open_text(f"{prefix}.YAML") as stream:
        diagnostics = yaml.safe_load(stream)['FIT_DIAGNOSTICS']
    assert set(diagnostics) == {'max_gradient', 'max_projected_gradient'}
    assert np.isfinite(diagnostics['max_gradient'])
    assert 0 <= diagnostics['max_projected_gradient'] <= diagnostics['max_gradient']


def test_column_list_matches_salt2mu(written):
    """SNANA column order, with a trailing IZBIN also for uncorrected fits."""
    name, prefix, output = written
    mine = read_fitres(f"{prefix}.FITRES", as_text=True)
    ref = read_fitres(f"{_REF}/{name}.FITRES", as_text=True)
    expected = list(ref.varnames)
    if name in ("salt_nobiascor", "salt_default_blindflag"):
        expected.append("IZBIN")
    assert mine.varnames == expected
    np.testing.assert_array_equal(np.asarray(mine.get("IZBIN"), dtype=int),
                                  output.izbin)


def test_m0dif_redshift_columns_have_salt2mu_precision(written):
    """write_M0_fitres prints the three redshift columns as %7.5f
    (SALT2mu.c:23947). zHD is what wfit and create_covariance read back, so a
    %.4f column truncates the cosmology fit's redshifts at 1e-5.
    """
    name, prefix, _ = written
    mine = read_fitres(f"{prefix}.M0DIF", as_text=True)
    ref = read_fitres(f"{_REF}/{name}.M0DIF", as_text=True)
    for col in ("zHDMIN", "zHDMAX", "zHD"):
        assert [_decimals(v) for v in mine.get(col)] == \
            [_decimals(v) for v in ref.get(col)], f"{name}.M0DIF: {col}"


def test_biascor_summary_matches_salt2mu(written):
    """NSIM/sigint carry the biasCor dimensionality and are absent without one;
    the alpha/beta grid note is SALT-only (SALT2mu.c:24953)."""
    name, prefix, _ = written
    assert _biascor_summary(f"{prefix}.FITRES") == \
        _biascor_summary(f"{_REF}/{name}.FITRES")


def test_bbc_columns_match_salt2mu(written):
    """The appended block's values, not just its column names.

    Names alone let IDSAMPLE ship as all-zeros while the real partition had two
    samples: the column was taken from the survey-level assignment rather than the
    biasCor partition, and only a value comparison catches that.
    """
    name, prefix, _ = written
    mine = read_fitres(f"{prefix}.FITRES", as_text=True)
    ref = read_fitres(f"{_REF}/{name}.FITRES", as_text=True)
    row_of = {c: i for i, c in enumerate(ref.get("CID"))}
    cols = ref.varnames[ref.varnames.index("CUTMASK"):]
    for col in cols:
        got, want = mine.get(col), ref.get(col)
        bad = [(c, got[i], want[row_of[c]])
               for i, c in enumerate(mine.get("CID"))
               if not _same_number(got[i], want[row_of[c]], col)]
        if not bad:
            continue
        # SALT2mu's distance integrator can terminate early at isolated redshifts:
        # rombint (SALT2mu.c:25789) stops on the agreement between two successive
        # Romberg extrapolations, which can be coincidental, leaving ~6e-4 mag
        # instead of its 1e-6 tolerance. So a handful of rows can disagree while
        # the column as a whole is right. Judge the column
        # by its distribution: the bulk must agree to the printed digit, and the
        # stragglers must be few and small. A jax defect moves the median, not two
        # rows in a thousand.
        # the tokens are decimals, so their difference is rounded back to one
        # before it meets a decimal bound (0.006 - 0.000 is not 6e-3 in binary)
        d = np.round(np.abs(np.array([float(got[i]) - float(want[row_of[c]])
                                      for i, c in enumerate(mine.get("CID"))])), 10)
        n = len(d)
        assert np.median(d) <= _column_tol(want[0], col), \
            f"{name}:{col} median|d|={np.median(d):.2e} -- systematic, not outliers"
        assert len(bad) <= max(2, _OUTLIER_FRAC * n), \
            f"{name}:{col} {len(bad)} of {n} rows disagree ({100*len(bad)/n:.2f}%)"
        assert d.max() <= _OUTLIER_MAX, \
            f"{name}:{col} worst outlier {d.max():.2e} exceeds {_OUTLIER_MAX:.0e}"


def test_passthrough_columns_are_token_identical(written):
    """SNANA re-emits the input words (write_fitres_line, SALT2mu.c:24501). A
    numeric comparison passes while "0.00100" silently becomes "0.001"."""
    name, prefix, _ = written
    mine = read_fitres(f"{prefix}.FITRES", as_text=True)
    ref = read_fitres(f"{_REF}/{name}.FITRES", as_text=True)
    cols = ref.varnames[:ref.varnames.index("CUTMASK")]
    row_of = {c: i for i, c in enumerate(ref.get("CID"))}
    for col in cols:
        got, want = mine.get(col), ref.get(col)
        assert [got[i] for i in range(mine.size)] == \
            [want[row_of[c]] for c in mine.get("CID")], f"{name}:{col}"


def _numeric_lines(path, skip=0):
    """Whitespace-separated numbers per line, for the free-format outputs."""
    out = []
    for line in list(open_text(path))[skip:]:
        if line.startswith(("#", "VARNAMES:")) or not line.strip():
            continue
        out.append(line.split())
    return out


def test_cov_matches_salt2mu(written):
    """write_M0_cov: nzbin then the row-major M0 covariance."""
    name, prefix, _ = written
    mine = _numeric_lines(f"{prefix}.COV")
    ref = _numeric_lines(f"{_REF}/{name}.COV")
    assert len(mine) == len(ref), f"{name}.COV: {len(mine)} vs {len(ref)} entries"
    for m, r in zip(mine, ref):
        a, b = float(m[0]), float(r[0])
        # the diagonal agrees to ~5e-5 relative; the off-diagonals are ~1e-6 in
        # absolute terms, where a relative bound means nothing
        assert abs(a - b) <= max(2e-3 * abs(b), 1e-5), f"{name}.COV: {a} vs {b}"


def test_csv_matches_salt2mu(written):
    """write_csv: the binned Hubble diagram, printed to 4 dp."""
    name, prefix, _ = written
    mine, ref = _numeric_lines(f"{prefix}.CSV"), _numeric_lines(f"{_REF}/{name}.CSV")
    assert len(mine) == len(ref)
    for m, r in zip(mine, ref):
        assert len(m) == len(r)
        # the z columns inverting mu(z) carry MUREF's ~1e-5 difference, which the
        # sparse high-z bin amplifies; mu/muerr are printed to 4 dp
        for x, y in zip(m[1:], r[1:]):
            assert _same_number(x, y, "MUERR"), f"{name}.CSV: {x} vs {y}"


def test_chi2grid_matches_salt2mu(written):
    """write_chi2grid: the alpha/beta chi2 scan. Grid points must be exact; the
    chi2 values carry the fit difference.

    SNANA scans fcn, so under a CC prior the y-axis is the BEAMS sum rather than the
    Ia chi2, evaluated after the fit has restored muerrsq_last to its own errors.
    Scanning the Ia chi2 instead put salt_ccprior 17% out.
    """
    name, prefix, _ = written
    mine = _numeric_lines(f"{prefix}.CHI2GRID")
    ref = _numeric_lines(f"{_REF}/{name}.CHI2GRID")
    assert len(mine) == len(ref), f"{name}.CHI2GRID: {len(mine)} vs {len(ref)} rows"
    for m, r in zip(mine, ref):
        assert m[1] == r[1] and m[2] == r[2], f"{name}.CHI2GRID grid: {m} vs {r}"
        for x, y in zip(m[3:], r[3:]):
            assert abs(float(x) - float(y)) <= 2e-3 * abs(float(y)), \
                f"{name}.CHI2GRID: {x} vs {y}"


# cutmask_write=-9 asks for the fit results without the per-SN rows. It writes a
# different file set from every config above, so it is its own test rather than a
# CONFIGS entry: the reference is bbc_des_beams/SALT2mu_cutmask_write_m9.FITRES.
_M9 = ["radiation=0",
       "nzbin=10", "zmin=0.1", "zmax=1.0", "u1=1", "u2=1", "p9=0.7", "p11=-1",
       f"datafile={_BC}/DATA.FITRES", f"simfile_biascor={_BC}/BIASCOR.FITRES",
       "blindflag=0", "cutmask_write=-9", "write_csv=1"] + _5D


@pytest.fixture(scope="module")
def cutmask_write_m9(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cutmask_write_m9")
    run_bbc(parse_bbc_input(None, overrides=_M9 + [f"prefix={tmp / 'JAX'}"]))
    return tmp


def test_cutmask_write_m9_writes_only_the_fitres(cutmask_write_m9):
    """SNANA returns before opening the M0DIF, CSV and COV (write_M0_fitres
    SALT2mu.c:23865, write_M0_csv :23980, write_M0_cov :24037) and still writes
    the FITRES, so the FITRES alone is the whole output.

    Checked against SALT2mu.exe run with the same keys plus write_csv=1: it too
    left one file.
    """
    assert sorted(p.name for p in cutmask_write_m9.iterdir()) == ["JAX.FITRES"]


def test_cutmask_write_m9_fitres_has_no_rows(cutmask_write_m9):
    """No SN rows, and hence no VARNAMES line either (write_fitres_driver,
    SALT2mu.c:24254, :24399)."""
    text = (cutmask_write_m9 / "JAX.FITRES").read_text()
    assert "VARNAMES:" not in text
    assert "SN:" not in text


def test_cutmask_write_m9_header_matches_salt2mu(cutmask_write_m9):
    """The provenance and warning block, including the cutmask_write warning
    SNANA writes into the FITRES and not the M0DIF (write_NWARN, :24752).

    What follows it, the `SALT2mu fit results:` block, is compared value by value
    by the per-feature parity tests rather than as text, for the reason
    `_same_line` gives. Two of its lines are a known gap and are not compared
    anywhere: `m0-M0avg chi2/dof` (`:23221`) and the reduced COV matrix
    `write_fitres_driver->printCOVMAT` writes after it (`:23237`), neither of
    which jax produces.
    """
    mine = _header(cutmask_write_m9 / "JAX.FITRES")
    ref = _header(f"{_BC}/SALT2mu_cutmask_write_m9.FITRES")
    # The warning is shared; the reference also points to SNANA source code.
    ref = [line.removesuffix(" (grep CUTBIT SALT2mu.c | grep define)")
           for line in ref]
    assert len(mine) == len(ref), f"{len(mine)} lines vs {len(ref)}"
    bad = [(m, r) for m, r in zip(mine, ref) if not _same_line(m, r)]
    assert not bad, bad[:3]


def test_mupull_is_capped_at_the_printed_width(tmp_path):
    """write_fitres_line_append caps the written pull at 99.999 immediately before
    formatting it (SALT2mu.c:25296). The cap is one-sided and applies to the column
    only: CHI2 keeps the uncapped square. Without it a %6.3f field holding 1e5
    widens and shifts every column after it on that row.
    """
    config = parse_bbc_input(None, overrides=_COMMON_HIZ + [
        f"datafile={_BC}/DATA.FITRES", "blindflag=0"])
    result, output, zbins, tv, accept = run_bbc_fit(config)
    # one event driven past the cap, the way an outlier reaches the writer
    output.mupull[0], output.mupull[1] = 1.0e5, -1.0e5
    path = str(tmp_path / "JAX.FITRES")
    write_fitres(path, output, zbins, result, config,
                 passthrough={"CID": np.asarray(tv.name)[accept]})

    written = read_fitres(path, as_text=True).get("MUPULL")
    assert written[0] == "99.999"
    assert float(written[1]) == -1.0e5      # SNANA does not cap the other side
