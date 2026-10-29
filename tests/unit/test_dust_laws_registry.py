"""Smoke + invariance tests for every dust law in DUST_LAWS.

For each registered dust law:
- ``extinction_curve(λ, Rv)`` returns finite values shaped like ``λ``.
- The V-band sanity check: A(λ_V)/A(V) ≈ 1 by definition of A(V), within
  the polynomial-precision tolerances of each parameterisation.
- Loading via ``load_dust_law(name)`` round-trips back to the same class.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_snlc_core.dust import DUST_LAWS, load_dust_law
from jax_snlc_core.logging import SNANAError


WAVE_V = 5495.0  # CCM89/Fitzpatrick definition of Johnson V

# PARLIST_MWCOLORLAW values for the laws that require them, and the fixed
# RV Gordon03 insists on; every other law takes RV=3.1
_PARLIST = {'GOOBAR08': (-1.5, 0.9), 'GORD16': (3.1, 0.5),
            'SOMM26': (1.0, 1.0, 0.5, 1.5)}
_RV = {'GORD03': 2.74}


def _prepared(name, wavelength):
    """Construct a law and prep its wavelength grid (no-op for laws that
    don't override the hook)."""
    law = load_dust_law(name)
    if name in _PARLIST:
        law.set_params(_PARLIST[name])
    law.prepare_wavelength_grid(np.asarray(wavelength))
    return law


@pytest.mark.parametrize("name", list(DUST_LAWS))
def test_registered_law_returns_finite_curve(name):
    """Every registered dust law evaluates to a finite array with the input shape."""
    wavelength = np.array([3000.0, 4000.0, WAVE_V, 7000.0, 10000.0])
    law = _prepared(name, wavelength)
    curve = np.asarray(law.extinction_curve(wavelength, rv=_RV.get(name, 3.1)))
    assert curve.shape == wavelength.shape, name
    assert np.all(np.isfinite(curve)), f"{name} produced non-finite values: {curve}"
    # Simulation kernels capture the prepared law and its wavelength grid.
    compiled = jax.jit(lambda av: law.apply_extinction(
        1., wavelength, av, _RV.get(name, 3.1)))
    for av in (0.1, 0.4):
        expected = law.apply_extinction(1., wavelength, av, _RV.get(name, 3.1))
        np.testing.assert_allclose(compiled(av), expected, rtol=1e-12)


@pytest.mark.parametrize(
    "name", [n for n in DUST_LAWS if n != "OFF"])
def test_registered_law_v_band_normalisation(name):
    """A(V)/A(V) ≈ 1 at the V-band reference wavelength.

    Tolerance is loose because each parameterisation hits the V-band only
    to its polynomial precision; the test guards against
    misimplementations that drift far from unity.
    """
    wavelength = np.array([WAVE_V])
    law = _prepared(name, wavelength)
    av_at_v = float(np.asarray(
        law.extinction_curve(wavelength, rv=_RV.get(name, 3.1)))[0])
    assert abs(av_at_v - 1.0) < 0.05, (
        f"{name}: A(λ_V)/A(V) = {av_at_v:.4f} differs from 1.0 by > 0.05"
    )


# The validity range each law's SNANA routine enforces, in Angstroms.
# GORD03, FITZ04 and GORD16 all reach GALextinct_Fitz99_exact, which aborts
# outside 912-35000 A whatever the OPT (MWgaldust.c:715-720, :482-485, :549).
_WAVE_RANGE = {
    # CCM89 and O'Donnell94 are polynomials SNANA evaluates at any wavelength;
    # FITZ99_APPROX enforces its own 25000 A bound inside extinction_curve
    # (MWgaldust.c:655), so it declares none here
    'OFF': None, 'CCM89': None, 'ODONNELL94': None, 'FITZ99_APPROX': None,
    'FITZ99_EXACT': (912.0, 35000.0), 'GORD03': (912.0, 35000.0),
    'FITZ04': (912.0, 35000.0), 'GORD16': (912.0, 35000.0),
    'GOOBAR08': (2000.0, 22000.0), 'MAIZ14': (3000.0, 35000.0),
    'FITZ19': (1150.0, 35000.0), 'FITZ19_LIN': (1150.0, 35000.0),
    'GORD23': (912.0, 320000.0), 'SOMM25': (700.0, 9134.0),
    'SOMM26': (1000.0, 10000.0),
}


@pytest.mark.parametrize("name", list(DUST_LAWS))
def test_the_declared_validity_range_is_the_one_snana_enforces(name):
    law = load_dust_law(name)
    assert law.wave_range == _WAVE_RANGE[name], name
    if law.wave_range is None:
        return                      # CCM89, O'Donnell94 and F99-approx
    lo, hi = law.wave_range
    law.check_wave_range(lo, hi)    # the range itself is accepted
    with pytest.raises(SNANAError, match="only valid from"):
        law.check_wave_range(lo - 1.0, hi)
    with pytest.raises(SNANAError, match="only valid from"):
        law.check_wave_range(lo, hi + 1.0)


@pytest.mark.parametrize("spell", [float, np.float64, np.float32, jnp.asarray])
def test_gordon03_rv_validation_across_scalar_types(spell):
    """Gordon (2003) is defined only at RV = 2.74 and SNANA aborts on anything
    else (MWgaldust.c:710-714). A numpy float32 or a device scalar used to skip
    the test and silently get the RV = 2.74 curve."""
    wavelength = np.array([3000.0, WAVE_V])
    law = _prepared('GORD03', wavelength)
    with pytest.raises(SNANAError, match="only valid for RV=2.74"):
        law.extinction_curve(wavelength, rv=spell(3.1))
    # 2.74 itself, as the input file's double, is the one value it takes
    assert np.all(np.isfinite(np.asarray(
        law.extinction_curve(wavelength, rv=2.74))))
