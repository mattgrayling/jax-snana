"""Host noise: the HOSTNOISE corrections, which need a unique map for each
observed field and band, and the host term of the fit's recomputed error."""

import numpy as np
import pytest

from jax_snlc_core.logging import SNANAError
from jax_snlc_core.noise import fluxcal_errcalc
from jax_snlc_fit.calibration import hostnoise_errscale


def test_host_noise_maps():
    deep = dict(band='gr', field='DEEP', mag=np.array([20., 30.]),
                noise=np.array([2., 3.]))
    wide = dict(band='gr', field='WIDE', mag=np.array([20., 30.]),
                noise=np.array([1., 2.]))
    np.testing.assert_allclose(
        hostnoise_errscale([deep, wide], ['r', 'g'], ['DEEP', 'WIDE'],
                           [10., 10.], 27.5), [2.5, 1.5])
    with pytest.raises(SNANAError, match="band 'r', field 'WIDE'; found 0"):
        hostnoise_errscale([deep], ['r'], ['WIDE'], [10.], 27.5)
    with pytest.raises(SNANAError, match="band 'r', field 'DEEP'; found 2"):
        hostnoise_errscale([deep, deep], ['r'], ['DEEP'], [10.], 27.5)


def test_errcalc_host_term_needs_a_sky_noise():
    """SNANA's ERRCALC host term scales the sky noise by SBFLUXADU/SKY_AVG_ADU
    with SKY_AVG_ADU = GAIN*SQ1, so it vanishes with the sky noise
    (snana.F90:20476-20477, :20549-20552); with sky noise it adds a variance
    in proportion to the host's surface brightness."""
    def errcalc(skysig, sb):
        return fluxcal_errcalc(fluxcal=100.0, fluxcal_err=10.0, zeropt=30.0, zpterr=0.0,
                               skysig=skysig, skysig_t=0.0, psf1=1.5, psf2=0.0,
                               psf_ratio=0.0, gain=2.0, zp_fluxcal=27.5, sb_fluxcal=sb,
                               pixsize=0.2)
    assert errcalc(0.0, 50.0) == errcalc(0.0, 0.0)
    host_variance = [errcalc(5.0, sb) ** 2 - errcalc(5.0, 0.0) ** 2 for sb in (50.0, 100.0)]
    assert host_variance[0] > 0.0
    assert host_variance[1] == pytest.approx(2.0 * host_variance[0], rel=1e-9)
