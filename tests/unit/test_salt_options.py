"""SALT generation options affect the SED or validate its inputs."""

import shutil

import jax.numpy as jnp
import numpy as np
import pytest

from jax_snlc_core.models import build_model
from jax_snlc_core.models.salt_model import load_salt_data
from jax_snlc_core.utils.error_handling import SNANAError
from jax_snlc_core.utils.extrap import modelflux_extrap
from jax_snlc_fit.fitter import fit_static_data


def test_negative_sed_clipping(data_root):
    path = str(data_root / 'models' / 'SALT3.K21')
    plain = build_model(path, {})
    clipped = build_model(path, {'GENMODEL_MSKOPT': 16})
    static = fit_static_data(clipped, None)
    params = {'t0': 0., 'redshift': 0., 'x0': 1., 'x1': 10., 'c': 0.}
    sed = plain.generate_sed(0., params, static)
    assert np.any(sed < 0) and np.any(sed > 0)
    np.testing.assert_array_equal(clipped.generate_sed(0., params, static),
                                  jnp.maximum(sed, 0.))
    # A broad band contains both signs, so clipping after integration differs.
    weights = jnp.ones((1, len(sed)))
    static.update(filter_ref_fluxes=jnp.ones(1), zp_fluxcal=0.)
    flux = clipped.evaluate(params, jnp.zeros(1), jnp.zeros(1, dtype=int),
                            jnp.ones(1, dtype=bool), weights, static)
    np.testing.assert_allclose(flux[0], jnp.maximum(sed, 0.).sum() * static['fluxcal_scale_mag_offset'],
                               rtol=1e-12)

    # Outside the training phases, extrapolate the clipped band integral.
    edge = float(clipped.phases[-1])
    phase = edge + 9.
    band_flux = lambda t: (clipped.generate_sed(t, params, static).sum()
                          * static['fluxcal_scale_mag_offset'])
    expected = modelflux_extrap(
        phase, edge, band_flux(edge), (band_flux(edge) - band_flux(edge - 3.)) / 3.)
    extrapolated = clipped.evaluate(
        params, jnp.array([phase]), jnp.zeros(1, dtype=int),
        jnp.ones(1, dtype=bool), weights, static)
    np.testing.assert_allclose(extrapolated[0], expected, rtol=1e-12)


@pytest.mark.parametrize('error_map', ['salt3_color_dispersion.dat',
                                      'salt3_lc_variance_0.dat'])
def test_salt_option_validation(data_root, tmp_path, error_map):
    path = tmp_path / 'SALT3.K21'
    shutil.copytree(data_root / 'models' / 'SALT3.K21', path)
    with pytest.raises(SNANAError, match='not supported'):
        build_model(str(path), {'GENMODEL_MSKOPT': 32})
    info = path / 'SALT3.INFO'
    text = info.read_text()
    info.write_text('DOCUMENTATION:\n  PURPOSE: Test model\nDOCUMENTATION_END:\n' + text)
    load_salt_data(str(path), 2)
    info.write_text(text)
    if not text.startswith('DOCUMENTATION:'):
        with pytest.raises(SNANAError, match='DOCUMENTATION'):
            load_salt_data(str(path), 2)
    files = ([path / error_map] if 'dispersion' in error_map
             else list(path.glob('salt3_lc_*.dat*')))
    for error_file in files:
        data = np.loadtxt(error_file)
        axis = np.unique(data[:, 0])
        keep = (data[:, 0] >= axis[2]) & (data[:, 0] <= axis[-3])
        np.savetxt(error_file, data[keep])
    with pytest.raises(SNANAError, match='short of the SED range'):
        load_salt_data(str(path), 64)
