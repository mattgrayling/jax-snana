"""File and in-memory fits share preparation and numerical postprocessing."""

import gzip
import numpy as np
import pytest
from astropy.io import fits

from jax_snlc_fit import LightCurveFitter
from jax_snlc_fit.config_parser import parse_fit_input_file
from jax_snlc_fit.run import run_fitting


@pytest.mark.parametrize('zp', [27.5, 31.4])
def test_python_and_file_fit_agree_with_covariance_repair(tmp_path, snana_fixture, data_root, zp):
    fixture = snana_fixture('salt3_fit_dense')
    version = 'comparison'
    directory = tmp_path / version
    directory.mkdir()
    for suffix in ('HEAD', 'PHOT'):
        (directory / f'{version}_{suffix}.FITS').write_bytes(
            gzip.decompress((fixture.root / f'{suffix}.FITS.gz').read_bytes()))
    with fits.open(directory / f'{version}_HEAD.FITS', mode='update') as hdus:
        hdus[0].header['ZP_FLUXCAL'] = zp
    (directory / f'{version}.LIST').write_text(f'{version}_HEAD.FITS\n')
    config = parse_fit_input_file(fixture.root / 'fit.nml', {
        'version_photometry': version, 'private_data_path': str(tmp_path),
        'fitmodel_name': str(data_root / 'models/SALT3.K21'),
        'kcor_file': str(data_root / 'kcor/JAX_TEST_LSST.fits'),
        'output_name': str(tmp_path / 'comparison.FITRES.TEXT'),
        'batch_size': 1, 'max_fit': 1, 'opt_covar_lcfit': 2,
    })
    file_result, _ = run_fitting(config)
    assert file_result.n_sne == 1
    with fits.open(fixture.head_fits) as hdus:
        head = hdus[1].data[0]
        start, stop = int(head['PTROBS_MIN']) - 1, int(head['PTROBS_MAX'])
        metadata = {name: np.asarray(head[name]).item() if np.ndim(head[name]) == 0 else head[name]
                    for name in hdus[1].columns.names
                    if name not in ('SNID', 'PTROBS_MIN', 'PTROBS_MAX')}
        redshift = float(head['REDSHIFT_HELIO'])
        metadata.update(redshift_err=float(head['REDSHIFT_HELIO_ERR']),
                        peakmjd=float(head['PEAKMJD']))
    with fits.open(fixture.phot_fits) as hdus:
        phot = hdus[1].data[start:stop]
        data = {name: np.array(phot[name]).astype(phot[name].dtype.newbyteorder('='))
                for name in hdus[1].columns.names}
    fitter = LightCurveFitter(config.model.fitmodel_name, config=config)
    python_result = fitter.fit(
        mjd=data['MJD'], flux=data['FLUXCAL'], flux_err=data['FLUXCALERR'],
        band=np.char.strip(data['BAND'].astype(str)),
        band_map={b: 'LSST-' + b for b in 'ugrizY'},
        redshift=redshift, metadata=metadata, zp=zp, apply_selection=True, progress=False)
    assert python_result.n_sne == 1
    # The Python adapter expresses fluxes at 27.5; rescaling the optimisation
    # problem from 31.4 changes convergence only at numerical precision.
    rtol = 1e-9 if zp == 27.5 else 1e-6
    atol = 1e-12
    for name in file_result.params:
        np.testing.assert_allclose(python_result.params[name], file_result.params[name], rtol=rtol, atol=atol)
        np.testing.assert_allclose(python_result.param_errors[name], file_result.param_errors[name], rtol=rtol, atol=atol)
    np.testing.assert_allclose(python_result.param_covariances, file_result.param_covariances, rtol=rtol, atol=atol)
    np.testing.assert_allclose(python_result.chi_squared, file_result.chi_squared, rtol=rtol)
