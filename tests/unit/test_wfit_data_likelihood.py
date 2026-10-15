"""WFIT HD loading, covariance semantics, and profile-likelihood tests."""

from pathlib import Path

import numpy as np
import pytest

from jax_snlc_core.physics.constants import LIGHT_KM
from jax_snlc_core.physics.cosmology import distance_modulus

_HUBBLE_DISTANCE_MPC = LIGHT_KM / 70.0
from jax_snlc_core.utils import SNANAError
from jax_wfit.config import WFITConfig
from jax_wfit.data import load_hubble_diagrams, stack_arrays
from jax_wfit.likelihood import evaluate_likelihood, hdibc_observed_mu
from jax_wfit.priors import prepare_priors


def _write_hd(path: Path) -> None:
    path.write_text(
        "VARNAMES: ROW zHD MUDIF MUDIFERR MUREF NFIT zHDERR\n"
        "ROW: 1 0.10 0.01 0.10 38.30 3 0.001\n"
        "ROW: 2 0.20 -0.02 0.20 40.00 1 0.002\n"
        "ROW: 3 0.30 0.03 0.30 40.90 4 0.003\n"
    )


def _write_covariance(path: Path, matrix: np.ndarray) -> None:
    lines = [str(matrix.shape[0])]
    lines.extend(f"{value:.12e}" for value in matrix.ravel())
    path.write_text("\n".join(lines) + "\n")


def _chi2(params, diagrams, config, priors):
    """Evaluate the likelihood over stacked diagrams, as the backends do."""
    first = diagrams[0]
    return evaluate_likelihood(
        params,
        *stack_arrays(diagrams)[:4],
        priors["omega_m_mean"],
        priors["cmb_r"],
        config,
        priors,
        hdibc=(
            None if first.mu_second is None
            else (first.mu_second, first.bias0, first.bias1)
        ),
    )


def test_muref_nfit_and_zerr(tmp_path):
    hd_path = tmp_path / "HD.M0DIF"
    _write_hd(hd_path)
    config = WFITConfig()
    config.hubble_diagram = str(hd_path)
    data = load_hubble_diagrams(config)
    assert len(data) == 1
    assert data[0].n_rows == 2
    np.testing.assert_array_equal(data[0].cid, ["1", "3"])
    np.testing.assert_allclose(data[0].mu, [38.31, 40.93])
    np.testing.assert_allclose(data[0].zerr, [0.001, 0.003])
    assert data[0].precision.shape == (2,)


def test_systematics_and_total_covariance_are_distinct(tmp_path):
    hd_path = tmp_path / "HD.M0DIF"
    cov_path = tmp_path / "HD.COV"
    _write_hd(hd_path)
    covariance = np.diag([0.04, 0.09, 0.16])
    _write_covariance(cov_path, covariance)

    systematics = WFITConfig()
    systematics.hubble_diagram = str(hd_path)
    systematics.covariance = str(cov_path)
    systematics.covariance_kind = "systematics"
    data_sys = load_hubble_diagrams(systematics)
    expected_sys = covariance[np.ix_([0, 2], [0, 2])] + np.diag([0.01, 0.09])
    np.testing.assert_allclose(
        data_sys[0].covariance, expected_sys,
    )

    total = WFITConfig()
    total.hubble_diagram = str(hd_path)
    total.covariance = str(cov_path)
    total.covariance_kind = "total"
    data_total = load_hubble_diagrams(total)
    expected_total = covariance[np.ix_([0, 2], [0, 2])]
    np.testing.assert_allclose(
        data_total[0].covariance, expected_total,
    )


def test_realisations_can_use_distinct_covariance_values(tmp_path):
    hd_path = tmp_path / "HD.M0DIF"
    cov_a_path = tmp_path / "HD_A.COV"
    cov_b_path = tmp_path / "HD_B.COV"
    _write_hd(hd_path)
    covariance_a = np.diag([0.04, 0.09, 0.16])
    covariance_b = np.diag([0.08, 0.18, 0.32])
    _write_covariance(cov_a_path, covariance_a)
    _write_covariance(cov_b_path, covariance_b)

    config = WFITConfig()
    config.hubble_diagram = [str(hd_path), str(hd_path)]
    config.covariance = [str(cov_a_path), str(cov_b_path)]
    config.covariance_kind = "systematics"
    data = load_hubble_diagrams(config)

    statistical = np.diag([0.01, 0.09])
    np.testing.assert_allclose(
        data[0].covariance,
        covariance_a[np.ix_([0, 2], [0, 2])] + statistical,
    )
    np.testing.assert_allclose(
        data[1].covariance,
        covariance_b[np.ix_([0, 2], [0, 2])] + statistical,
    )


def test_dense_covariance_realisations_are_padded_and_masked(tmp_path):
    hd_a = tmp_path / "HD_A.FITRES"
    hd_b = tmp_path / "HD_B.FITRES"
    cov_a = tmp_path / "HD_A.COV"
    cov_b = tmp_path / "HD_B.COV"
    hd_a.write_text(
        "VARNAMES: CID zHD MU MUERR\n"
        "SN: A 0.10 38.40 0.10\n"
        "SN: B 0.20 40.05 0.12\n"
        "SN: C 0.30 40.95 0.15\n"
    )
    hd_b.write_text(
        hd_a.read_text()
        + "SN: D 0.40 41.60 0.18\n"
    )
    _write_covariance(cov_a, np.diag([0.01, 0.02, 0.03]))
    _write_covariance(cov_b, np.diag([0.01, 0.02, 0.03, 0.04]))

    config = WFITConfig()
    config.hubble_diagram = [str(hd_a), str(hd_b)]
    config.covariance = [str(cov_a), str(cov_b)]
    config.covariance_kind = "systematics"
    data = load_hubble_diagrams(config)
    priors = prepare_priors(config, [None], None)
    together = _chi2({"omega_m": 0.3, "w0": -1.0}, data, config, priors)

    _, _, precision, valid = stack_arrays(data)
    assert precision.shape == (2, 4, 4)
    np.testing.assert_array_equal(
        valid,
        [[True, True, True, False], [True, True, True, True]],
    )
    for index, (hd_path, cov_path) in enumerate(
        ((hd_a, cov_a), (hd_b, cov_b))
    ):
        config.hubble_diagram = str(hd_path)
        config.covariance = str(cov_path)
        separate_data = load_hubble_diagrams(config)
        separate = _chi2({"omega_m": 0.3, "w0": -1.0}, separate_data, config, priors)
        assert float(together["hd_chi2"][index]) == pytest.approx(
            float(separate["hd_chi2"][0]), rel=1.0e-12,
        )


def test_npz_total_covariance(tmp_path):
    hd_path = tmp_path / "HD.M0DIF"
    cov_path = tmp_path / "HD_TOTAL.npz"
    _write_hd(hd_path)
    covariance = np.diag([0.04, 0.09, 0.16])
    np.savez(cov_path, cov=covariance)

    config = WFITConfig()
    config.hubble_diagram = str(hd_path)
    config.covariance = str(cov_path)
    config.covariance_kind = "total"
    data = load_hubble_diagrams(config)

    expected = covariance[np.ix_([0, 2], [0, 2])]
    np.testing.assert_allclose(data[0].covariance, expected)


def _write_hd_rows(path, n):
    """An HD whose rows all survive: the nfit > 1 cut, and no z cut."""
    rows = "".join(
        f"ROW: {i + 1} {0.1 + 0.05 * i:.2f} 0.01 0.10 {38.0 + i:.2f} 3\n"
        for i in range(n)
    )
    path.write_text("VARNAMES: ROW zHD MUDIF MUDIFERR MUREF NFIT\n" + rows)


def test_snana_packed_npz_is_read_as_a_precision_matrix(tmp_path):
    """create_covariance's default output is covtot_inv: an NPZ holding only the
    upper triangle. n=8 is the size whose triangle is itself a perfect square."""
    n = 8
    hd_path = tmp_path / "HD.M0DIF"
    cov_path = tmp_path / "covtot_inv_000.npz"
    _write_hd_rows(hd_path, n)
    rng = np.random.default_rng(0)
    root = rng.normal(size=(n, n))
    # a precision matrix the loader will accept: symmetric and positive definite
    precision = root @ root.T + n * np.eye(n)
    np.savez(cov_path, nsn=[n],
             cov=precision[np.triu_indices(n)].astype(np.float32))

    config = WFITConfig()
    config.hubble_diagram = str(hd_path)
    config.covariance = str(cov_path)
    config.covariance_kind = "total_inverse"
    data = load_hubble_diagrams(config)

    assert data[0].n_rows == n
    np.testing.assert_allclose(data[0].precision, precision, rtol=0, atol=1e-6)


def test_gzipped_covariance_text_is_read(tmp_path):
    """create_covariance writes covsys as .txt.gz (SUFFIX_COV_DICT)."""
    import gzip

    n = 3
    hd_path = tmp_path / "HD.M0DIF"
    cov_path = tmp_path / "covsys_000.txt.gz"
    _write_hd_rows(hd_path, n)
    cov = np.diag([0.04, 0.09, 0.16])
    body = f"{n}\n" + "".join(f"{v:13.6e}\n" for v in cov.flatten())
    cov_path.write_bytes(gzip.compress(body.encode()))

    config = WFITConfig()
    config.hubble_diagram = str(hd_path)
    config.covariance = str(cov_path)
    config.covariance_kind = "total"
    data = load_hubble_diagrams(config)

    np.testing.assert_allclose(data[0].covariance, cov)


def test_muerr_force_and_snrms_set_diagonal_precision(tmp_path):
    hd_path = tmp_path / "HD.M0DIF"
    _write_hd(hd_path)
    config = WFITConfig()
    config.hubble_diagram = str(hd_path)
    config.muerr_force = 0.05
    config.snrms = 0.03
    data = load_hubble_diagrams(config)

    np.testing.assert_allclose(data[0].muerr, [0.05, 0.05])
    np.testing.assert_allclose(
        data[0].precision,
        np.full(2, 1.0 / (0.05**2 + 0.03**2)),
    )


def test_idealised_data_use_truth_polynomial_and_realisation_keys(tmp_path):
    hd_path = tmp_path / "HD.M0DIF"
    _write_hd(hd_path)
    config = WFITConfig()
    config.hubble_diagram = [str(hd_path), str(hd_path)]
    config.om_sim = 0.3
    config.w0_sim = -0.9
    config.wa_sim = 0.2
    config.muerr_ideal = True
    config.muerr_ideal_polynomial = (0.08, 0.04)

    data = load_hubble_diagrams(config)

    redshift = np.asarray(data[0].redshift)
    expected_sigma = 0.08 + 0.04 * redshift
    expected_model = distance_modulus(
        redshift, redshift, 0.3, 0.7, _HUBBLE_DISTANCE_MPC, -0.9, 0.2,
    )

    np.testing.assert_allclose(data[0].muerr, expected_sigma)
    # noise is drawn around the truth model with the polynomial sigma
    resid = (np.asarray(data[0].mu) - np.asarray(expected_model))
    assert np.all(np.abs(resid) < 6.0 * expected_sigma)
    assert not np.array_equal(data[0].mu, expected_model)
    assert not np.array_equal(data[0].mu, data[1].mu)
    np.testing.assert_allclose(
        data[0].precision,
        1.0 / expected_sigma**2,
    )

    repeated = load_hubble_diagrams(config)
    np.testing.assert_array_equal(repeated[0].mu, data[0].mu)


def test_idealised_data_can_reuse_input_uncertainties(tmp_path):
    hd_path = tmp_path / "HD.M0DIF"
    _write_hd(hd_path)
    config = WFITConfig()
    config.hubble_diagram = str(hd_path)
    config.om_sim = 0.3
    config.w0_sim = -1.0
    config.wa_sim = 0.0
    config.muerr_ideal = True
    config.muerr_ideal_use_input = True

    data = load_hubble_diagrams(config)

    np.testing.assert_allclose(data[0].muerr, [0.10, 0.30])


def test_profile_likelihood_agrees_between_diagonal_and_dense_precision(tmp_path):
    """The profile takes an einsum for a dense precision and an elementwise
    product for a diagonal one; the same covariance must give the same answer.

    The chi-square value itself is pinned against SNANA's `wfit.exe` by
    tests/parity/test_wfit_frozen.py::test_cospar_matches_frozen_snana.
    """
    hd_path = tmp_path / "HD.FITRES"
    hd_path.write_text(
        "VARNAMES: CID zHD MU MUERR\n"
        "SN: A 0.10 38.40 0.10\n"
        "SN: B 0.20 40.05 0.12\n"
        "SN: C 0.30 40.95 0.15\n"
    )
    cov_path = tmp_path / "HD.COV"
    _write_covariance(cov_path, np.diag([0.10, 0.12, 0.15]) ** 2)

    params = {"omega_m": 0.3, "w0": -1.0}
    config = WFITConfig()
    config.hubble_diagram = str(hd_path)
    diagonal_data = load_hubble_diagrams(config)
    assert np.asarray(diagonal_data[0].precision).ndim == 1
    diagonal = _chi2(params, diagonal_data, config, prepare_priors(config, [None], None))

    config.covariance = str(cov_path)
    config.covariance_kind = "total"
    dense_data = load_hubble_diagrams(config)
    assert np.asarray(dense_data[0].precision).ndim == 2
    dense = _chi2(params, dense_data, config, prepare_priors(config, [None], None))

    assert float(dense["intercept"][0]) == pytest.approx(
        float(diagonal["intercept"][0]), rel=1.0e-11,
    )
    assert float(dense["hd_chi2"][0]) == pytest.approx(
        float(diagonal["hd_chi2"][0]), rel=1.0e-10,
    )


def test_hdibc_refuses_a_mushift_range(tmp_path):
    """HDIBC adds one constant MUSHIFT to every bias-correction distance. A
    simulation that drew it per event from lo:hi has no such constant, where
    SNANA's wfit would read the first number."""
    for name in ("HD0", "HD1"):
        (tmp_path / f"{name}.FITRES").write_text(
            "VARNAMES: CID zHD MU MUERR\nSN: A 0.1 38.3 0.1\n")
        (tmp_path / f"{name}.YML").write_text(
            "MUSHIFT: 0.0:0.2\nOMEGA_MATTER: 0.3\nOMEGA_LAMBDA: 0.7\n"
            "w0_LAMBDA: -1.0\nwa_LAMBDA: 0.0\n")
    config = WFITConfig()
    config.hubble_diagram = [str(tmp_path / "HD0.FITRES"), str(tmp_path / "HD1.FITRES")]
    config.hdibc = True
    config.hdibc_info = [str(tmp_path / "HD0.YML"), str(tmp_path / "HD1.YML")]

    with pytest.raises(SNANAError, match="MUSHIFT from 0.0:0.2"):
        load_hubble_diagrams(config)


@pytest.mark.parametrize("inverse", [False, True])
def test_hdibc_interpolates_common_cids_at_trial_cosmology(tmp_path, inverse):
    z_by_cid = {"A": 0.1, "B": 0.2, "C": 0.3, "D": 0.4}
    order0 = ("A", "B", "C", "D")
    order1 = ("C", "A", "D")
    hd0 = tmp_path / "HD0.FITRES"
    hd1 = tmp_path / "HD1.FITRES"
    info0 = tmp_path / "INFO0.YML"
    info1 = tmp_path / "INFO1.YML"

    def write_hd(path, order, omega_m, residual):
        redshift = np.asarray([z_by_cid[cid] for cid in order])
        distance = np.asarray(distance_modulus(
            redshift, redshift, omega_m, 1.0 - omega_m, _HUBBLE_DISTANCE_MPC, -1.0, 0.0,
        ))
        lines = ["VARNAMES: CID zHD MU MUERR"]
        lines.extend(
            f"SN: {cid} {z:.8f} {mu + residual:.12f} 0.10"
            for cid, z, mu in zip(order, redshift, distance)
        )
        path.write_text("\n".join(lines) + "\n")

    def write_info(path, omega_m, omega_lambda):
        path.write_text(
            "MUSHIFT: 0.0\n"
            f"OMEGA_MATTER: {omega_m}\n"
            f"OMEGA_LAMBDA: {omega_lambda}\n"
            "w0_LAMBDA: -1.0\n"
            "wa_LAMBDA: 0.0\n"
        )

    write_hd(hd0, order0, 0.2, 0.01)
    write_hd(hd1, order1, 0.4, 0.03)
    # info0 is flat, so a trial at omega_m=0.2 lands exactly on bias0 and the
    # interpolation fraction is zero.
    write_info(info0, 0.2, 0.8)
    write_info(info1, 0.4, 0.75)

    config = WFITConfig()
    config.hubble_diagram = [str(hd0), str(hd1)]
    config.model = "lcdm"
    config.hdibc = True
    config.hdibc_info = [str(info0), str(info1)]
    if inverse:
        matrix = np.eye(4) + .2 * np.ones((4, 4))
        cov_path = tmp_path / "inverse.npz"
        np.savez(cov_path, cov=np.linalg.inv(matrix))
        config.covariance = str(cov_path)
        config.covariance_kind = "total_inverse"
        with pytest.raises(SNANAError, match="matching HDIBC event IDs"):
            load_hubble_diagrams(config)
        return
    data = load_hubble_diagrams(config)
    priors = prepare_priors(config, [None], None)
    components = _chi2({"omega_m": 0.3}, data, config, priors)

    assert data[0].mu_second is not None
    np.testing.assert_array_equal(data[0].cid, ["A", "C", "D"])
    np.testing.assert_allclose(
        data[0].bias0,
        distance_modulus(data[0].redshift, data[0].redshift, 0.2, 0.8, _HUBBLE_DISTANCE_MPC),
    )
    np.testing.assert_allclose(
        data[0].bias1,
        distance_modulus(data[0].redshift, data[0].redshift, 0.4, 0.75, _HUBBLE_DISTANCE_MPC),
    )
    # At the bias0 cosmology the fraction is zero, so the blended distances are
    # the first diagram's and the chi-square is the one the plain path gives on
    # the same rows. Away from it the blend moves the answer, so the identity
    # is not vacuous.
    def plain(omega_m):
        return evaluate_likelihood(
            {"omega_m": omega_m}, *stack_arrays(data)[:4],
            priors["omega_m_mean"], priors["cmb_r"], config, priors,
            hdibc=None,
        )

    at_bias0 = _chi2({"omega_m": 0.2}, data, config, priors)
    assert float(at_bias0["hd_chi2"][0]) == pytest.approx(
        float(plain(0.2)["hd_chi2"][0]), rel=1.0e-10,
    )
    assert float(components["hd_chi2"][0]) != pytest.approx(
        float(plain(0.3)["hd_chi2"][0]), rel=1.0e-6,
    )


_HDIBC = Path(__file__).parent.parent / "fixtures" / "wfit" / "hdibc"
_HDIBC_REFERENCES = ("ref_plain", "ref_clip_hi", "ref_clip_lo")


def _hdibc_reference(name):
    """wfit.exe's per-event columns and its best-fit omega_m for one reference."""
    lines = (_HDIBC / f"{name}.resid").read_text().splitlines()
    varnames = next(l.split()[1:] for l in lines if l.startswith("VARNAMES:"))
    rows = [l.split()[1:] for l in lines if l.startswith("SN:")]
    columns = {name: np.array([float(row[index]) for row in rows])
               for index, name in enumerate(varnames) if name != "CID"}
    columns["CID"] = [row[varnames.index("CID")] for row in rows]
    cospar = [line for line in (_HDIBC / f"{name}.cospar").read_text().splitlines()
              if not line.startswith("#")][0].split()
    return columns, float(cospar[2])


def _hdibc_config():
    config = WFITConfig()
    config.hubble_diagram = [str(_HDIBC / "HD0" / "HD.FITRES"),
                             str(_HDIBC / "HD1" / "HD.FITRES")]
    config.model = "lcdm"
    config.hdibc = True
    config.hdibc_info = [str(_HDIBC / "HD0" / "INFO.YML"),
                         str(_HDIBC / "HD1" / "INFO.YML")]
    return config


@pytest.mark.parametrize("name", _HDIBC_REFERENCES)
def test_hdibc_blend_matches_wfit_per_event(name):
    """Both bias legs, the interpolation fraction and the blended distance are
    wfit.exe's, event by event.

    `wfit.exe` writes mu_bcor0/1_hdibc and f_interp_hdibc per event with
    -outfile_resid (wfit.c:5366-5408); tests/fixtures/wfit/hdibc/README.md
    records the runs. wfit removes the mean residual from the mu it prints
    (wfit.c:5377), so the blended distance is compared about its own mean.
    """
    columns, omega_m = _hdibc_reference(name)
    data = load_hubble_diagrams(_hdibc_config())
    assert list(data[0].cid) == columns["CID"]

    # jax integrates the comoving distance by Gauss-Legendre where wfit's
    # codist uses a trapezoid, so the legs agree to a few parts in 10,000 of a
    # magnitude; wfit prints them to 1e-4 and the fraction to 1e-3.
    np.testing.assert_allclose(np.asarray(data[0].bias0),
                               columns["mu_bcor0_hdibc"], atol=1.0e-3)
    np.testing.assert_allclose(np.asarray(data[0].bias1),
                               columns["mu_bcor1_hdibc"], atol=1.0e-3)

    model = np.asarray(distance_modulus(
        data[0].redshift, data[0].redshift, omega_m, 1.0 - omega_m,
        _HUBBLE_DISTANCE_MPC, -1.0, 0.0))
    observed, fraction = hdibc_observed_mu(
        model, np.asarray(data[0].mu), np.asarray(data[0].mu_second),
        np.asarray(data[0].bias0), np.asarray(data[0].bias1))
    observed, fraction = np.asarray(observed), np.asarray(fraction)

    np.testing.assert_allclose(fraction, columns["f_interp_hdibc"], atol=1.0e-3)
    np.testing.assert_allclose(observed - observed.mean(),
                               columns["mu_obs"] - columns["mu_obs"].mean(),
                               atol=2.0e-4)


def test_the_hdibc_references_span_the_window_and_both_clip_bounds():
    """Otherwise one side of the clip would go unpinned."""
    fractions = {name: _hdibc_reference(name)[0]["f_interp_hdibc"]
                 for name in _HDIBC_REFERENCES}
    inside = fractions["ref_plain"]
    assert np.all((inside > -0.5) & (inside < 1.5)) and np.ptp(inside) > 0.01
    assert np.all(fractions["ref_clip_hi"] == 1.5)
    assert np.all(fractions["ref_clip_lo"] == -0.5)


def test_covariance_npz_without_nsn_is_refused(tmp_path):
    """A packed upper triangle needs its order; guessing it is silently wrong.

    N=8 is one of the sizes where N(N+1)/2 is itself square, so the old
    sqrt-reshape succeeded and returned a 6x6 matrix built from 8x8 data.
    """
    from jax_wfit.data import _load_matrix

    path = tmp_path / "cov.npz"
    np.savez(path, cov=np.arange(36, dtype=np.float32))
    with pytest.raises(SNANAError, match="no 'nsn'"):
        _load_matrix(str(path))


def test_labelled_covariance_text_roundtrips(tmp_path):
    """--label_cov_rows appends a comment to each value line; wfit reads the
    first token only (wfit.c:2011), so the file still loads."""
    from jax_create_cov.output import write_covariance_text
    from jax_snlc_core.io.cov_matrix import read_covariance_text

    matrix = np.array([[4.0, 1.0], [1.0, 3.0]])
    rows = (np.array(["1234", "5678"]), np.array([10, 10]),
            np.array([0.1, 0.2]))
    path = tmp_path / "covsys_000.txt"
    write_covariance_text(path, matrix, rows)

    np.testing.assert_allclose(read_covariance_text(path), matrix, rtol=1e-6)


@pytest.mark.parametrize("kind", ["diagonal", "systematics", "total"])
@pytest.mark.parametrize("zmin", [0.0, 0.2])
@pytest.mark.parametrize("initial_scatter", [0.0, 0.15])
@pytest.mark.parametrize("scatter", [0.0, 0.1])
def test_supplied_covariance_matches_file(tmp_path, kind, zmin,
                                          initial_scatter, scatter):
    from dataclasses import replace
    from jax_wfit.run import _cut_supplied_diagram

    hd_path, cov_path = tmp_path / "HD.M0DIF", tmp_path / "HD.COV"
    _write_hd(hd_path)
    covariance = np.array([[.04, .018, .012], [.018, .09, .02], [.012, .02, .04]])
    _write_covariance(cov_path, covariance)
    config = WFITConfig(hubble_diagram=str(hd_path),
                        covariance=str(cov_path) if kind != "diagonal" else None,
                        covariance_kind=kind if kind != "diagonal" else None,
                        snrms=initial_scatter)
    original = load_hubble_diagrams(config)[0]
    target = replace(config, zmin=zmin, snrms=scatter)
    expected = load_hubble_diagrams(target)[0]
    supplied = _cut_supplied_diagram(target, original)
    np.testing.assert_array_equal(supplied.cid, expected.cid)
    np.testing.assert_array_equal(supplied.raw_indices, expected.raw_indices)
    np.testing.assert_allclose(supplied.precision, expected.precision, rtol=1e-6)
    if kind != "diagonal":
        np.testing.assert_allclose(supplied.covariance, expected.covariance, rtol=1e-6)
        np.testing.assert_allclose(supplied.precision @ supplied.covariance,
                                   np.eye(supplied.n_rows), atol=1e-6)
    repeated = _cut_supplied_diagram(target, supplied)
    np.testing.assert_array_equal(repeated.precision, supplied.precision)
    np.testing.assert_array_equal(repeated.raw_indices, supplied.raw_indices)


@pytest.mark.parametrize('column', ['zHD', 'MU', 'MUERR'])
def test_file_and_supplied_diagrams_reject_nonfinite_values(tmp_path, column):
    from jax_wfit.data import HubbleDiagram
    from jax_wfit.run import _cut_supplied_diagram
    values = dict(zHD=.1, MU=38., MUERR=.1)
    values[column] = np.nan
    path = tmp_path / 'bad.txt'
    path.write_text('VARNAMES: CID zHD MU MUERR\nSN: A ' +
                    ' '.join(str(values[k]) for k in ('zHD', 'MU', 'MUERR')) + '\n')
    config = WFITConfig(hubble_diagram=str(path))
    diagram = HubbleDiagram(cid=np.array(['A']), redshift=np.array([values['zHD']]),
        mu=np.array([values['MU']]), muerr=np.array([values['MUERR']]),
        precision=np.array([100.]), covariance_kind='diagonal')
    with pytest.raises(SNANAError, match='non-finite required'):
        load_hubble_diagrams(config)
    with pytest.raises(SNANAError, match='non-finite required'):
        _cut_supplied_diagram(config, diagram)


@pytest.mark.parametrize('kind', ['diagonal', 'total', 'systematics', 'total_inverse'])
def test_unchanged_prepared_diagram_reuses_precision(kind, monkeypatch):
    from jax_wfit.data import HubbleDiagram
    from jax_wfit.run import _cut_supplied_diagram
    precision = np.ones(2) * 100 if kind == 'diagonal' else np.eye(2) * 100
    covariance = np.eye(2) * .01 if kind in ('total', 'systematics') else None
    diagram = HubbleDiagram(cid=np.array(['A', 'B']), redshift=np.array([.1, .2]),
        mu=np.array([38., 40.]), muerr=np.array([.1, .1]), precision=precision,
        covariance_kind=kind, covariance=covariance)
    def unexpected(*args, **kwargs):
        pytest.fail('An unchanged prepared diagram must not reconstruct its precision')
    monkeypatch.setattr('jax_wfit.data._precision_from_covariance', unexpected)
    assert _cut_supplied_diagram(WFITConfig(), diagram) is diagram
    precision.flat[0] = np.nan
    with pytest.raises(SNANAError, match='non-finite|finite'):
        _cut_supplied_diagram(WFITConfig(), diagram)


def test_covariance_identity_selection_reuses_storage():
    from jax_wfit.data import selected_covariance
    matrix = np.array([[2., .2], [.2, 3.]])
    assert selected_covariance(matrix, np.arange(2), 2) is matrix
    np.testing.assert_array_equal(selected_covariance(matrix, np.array([1, 0]), 2),
                                  matrix[::-1, ::-1])
    matrix[0, 0] = np.nan
    with pytest.raises(SNANAError, match='non-finite'):
        selected_covariance(matrix, np.arange(2), 2)


@pytest.mark.parametrize('kind', ['total', 'systematics', 'total_inverse'])
@pytest.mark.parametrize('precision', [np.array([[1., 2.], [2., 1.]]),
                                       np.ones((2, 2))])
def test_prepared_diagram_rejects_indefinite_or_singular_precision(kind, precision):
    from jax_wfit.data import HubbleDiagram
    from jax_wfit.run import _cut_supplied_diagram
    diagram = HubbleDiagram(
        cid=np.array(['A', 'B']), redshift=np.array([.1, .2]),
        mu=np.array([38., 40.]), muerr=np.array([.1, .1]),
        precision=precision, covariance_kind=kind,
        covariance=np.eye(2) if kind != 'total_inverse' else None)
    with pytest.raises(SNANAError, match='positive definite'):
        _cut_supplied_diagram(WFITConfig(), diagram)
