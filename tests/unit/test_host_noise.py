"""Host-noise corrections require a unique map for each observed field/band."""

import numpy as np
import pytest

from jax_snlc_core.utils.error_handling import SNANAError
from jax_snlc_fit.host_noise import hostnoise_errscale


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
