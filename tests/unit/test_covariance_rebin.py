"""Named-coordinate grouping, weights and covariance alignment."""

import numpy as np
import pandas as pd
import pytest

from jax_create_cov import build_covariance
from jax_create_cov.config import Cosmology, resolve_rebin_bins
from jax_create_cov.config_parser import build_config
from jax_snlc_core.utils import SNANAError


@pytest.mark.parametrize('bins', [{}, {'shape': 2}, {'shape': 2, 'dust': 3, 'velocity': 2}])
def test_named_rebin(bins):
    # Two events per coordinate cell; the missing redshift bin must not drop data.
    base = pd.DataFrame({
        'IZBIN': [0, 0, 0, 0, 2, 2, 2, 2],
        'shape': [0, 0, 1, 1, 0, 0, 1, 1],
        'dust': [0, 0, 1, 1, 0, 0, 2, 2],
        'velocity': [1, 1, 0, 0, 0, 0, 1, 1],
        'MU': np.arange(8.) + 38., 'MUREF': np.arange(8.) * .1 + 38.,
        'MUERR': [.1, .2] * 4, 'MUERR_RENORM': [.2, .3] * 4,
    })
    shifted = base.copy()
    shift = np.arange(8.) * .01
    shifted['MU'] += shift
    # Variants are matched by event ID, not their original order or coordinates.
    shifted['shape'] = 99.
    result = build_covariance(
        {'FITOPT000_MUOPT000': base, 'FITOPT001_MUOPT000': shifted.iloc[::-1]},
        {0: ('DEFAULT', 1.), 1: ('CALIB', 1.)}, {0: ('DEFAULT', 1.)}, [],
        mode='rebin', rebin_bins=bins, cosmology=Cosmology(.3, .7))
    groups = list(base.groupby(['IZBIN', *bins], sort=True))
    assert result.hubble_diagram['NEVT'].sum() == len(base)
    offsets = []
    for (_, group), (_, row) in zip(groups, result.hubble_diagram.iterrows()):
        w0 = 1. / group.MUERR.to_numpy() ** 2
        w1 = 1. / group.MUERR_RENORM.to_numpy() ** 2
        assert row.MU == pytest.approx(np.average(group.MU, weights=w1))
        assert row.MUREF == pytest.approx(np.average(group.MUREF, weights=w0))
        assert row.MUERR == pytest.approx(1. / np.sqrt(w1.sum()))
        offsets.append(np.average(shift[group.index], weights=w1))
    assert len(result.hubble_diagram) == len(groups)
    np.testing.assert_allclose(result.covariances["ALL"]["matrix"], np.outer(offsets, offsets), atol=1e-14)
    assert list(result.rebin[1]) == list(bins)


def test_rebin_config(tmp_path):
    from create_covariance import config_from_args

    bins = {'theta': 3, 'AV': 2}
    config = build_config({'REBIN_BINS': bins})
    assert config.mode == 'rebin' and config.rebin_bins == bins
    source = tmp_path / 'cov.yml'
    source.write_text('INPUT_DIR: bbc\nVERSION: V\n')
    cli = config_from_args([str(source), '-o', str(tmp_path), '--rebin', 'theta=3', 'AV=2'])
    assert cli.rebin_bins == bins and cli.mode == 'rebin'
    with pytest.raises(SNANAError, match='not both'):
        config_from_args([str(source), '-o', str(tmp_path), '--rebin', 'theta=3',
                          '--nbin_x1', '2', '--nbin_c', '2'])


@pytest.mark.parametrize('bins', [{'shape': 0}, {'shape': 2.5}, {'shape': True}, {'IZBIN': 3}])
def test_invalid_rebin_bins(bins):
    with pytest.raises(SNANAError):
        resolve_rebin_bins(bins)
