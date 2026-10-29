"""Unit tests for the SNANA MAGLIMIT_calculator port."""

from jax_snlc_sim.diagnostics import maglimit_calculator


def test_returns_zero_for_crazy_args():
    assert maglimit_calculator(1.0, 0.5, 21.0, 5.0) == 0.0    # zpt <= 1.01
    assert maglimit_calculator(30.0, 0.0, 21.0, 5.0) == 0.0   # psf <= 0.001
    assert maglimit_calculator(30.0, 0.5, 0.5, 5.0) == 0.0    # skymag <= 1.0


def test_higher_snr_gives_brighter_limit():
    m5 = maglimit_calculator(30.5, 0.9, 22.0, 5.0)
    m10 = maglimit_calculator(30.5, 0.9, 22.0, 10.0)
    assert m10 < m5  # requiring SNR=10 yields a brighter (smaller mag) limit


def test_typical_value_is_sensible():
    # LSST-ish r-band single visit: ~24-25 mag 5-sigma limit
    m5 = maglimit_calculator(30.5, 0.9, 22.0, 5.0)
    assert 22.0 < m5 < 27.0
