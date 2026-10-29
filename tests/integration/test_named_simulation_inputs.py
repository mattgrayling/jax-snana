"""Named population inputs preserve the same seeded simulation."""
from dataclasses import replace

import numpy as np

from jax_snlc_sim import run, simulate
from jax_snlc_sim.inputs import build, validate


def test_named_and_snana_simulations_match(data_root):
    shared=dict(GENMODEL=str(data_root/'models/SALT3.K21'),GENFILTERS='ugrizY',
        KCOR_FILE=str(data_root/'kcor/JAX_TEST_LSST.fits'),
        SIMLIB_FILE=str(data_root/'simlib/JAX_TEST_LSST_DENSE.SIMLIB'),
        RANSEED=122459, NGENTOT_LC=2, GENRANGE_REDSHIFT=[.02,.4],
        GENRANGE_TREST=[-20,50], GENRANGE_PEAKMJD=[60540,61460], OPT_MWEBV=0,
        GENPEAK_SALT2x1=0.,GENSIGMA_SALT2x1=[1.,1.],GENRANGE_SALT2x1=[-2,2],
        GENPEAK_SALT2c=0.,GENRANGE_SALT2c=[0.,0.],
        GENPEAK_SALT2ALPHA=.14,GENRANGE_SALT2ALPHA=[.14,.14],
        GENPEAK_SALT2BETA=3.1,GENRANGE_SALT2BETA=[3.1,3.1],
        DNDZ=[['POWERLAW',2.5e-5,1.5]])
    legacy=build(shared)
    validate(legacy,shared)
    named=dict(genmodel=shared['GENMODEL'], genfilters='ugrizY',
        kcor_file=shared['KCOR_FILE'],simlib_file=shared['SIMLIB_FILE'],random_seed=122459,
        ngentot_lc=2,redshift_range=[.02,.4],trest_range=[-20,50],
        peakmjd_range=[60540,61460],opt_mwebv=0,
        dndz_entries=[['POWERLAW',2.5e-5,1.5]],
        populations={'SALT2x1':{'distribution':'normal','peak':0.,'sigma':1.,'range':[-2,2]},
            'SALT2c':{'peak':0.,'range':[0.,0.]},
            'SALT2ALPHA':{'peak':.14,'range':[.14,.14]},
            'SALT2BETA':{'peak':3.1,'range':[3.1,3.1]}})
    legacy.numerical = replace(legacy.numerical, batch_size=2)
    compiled = run.prepare(legacy)
    expected, expected_selection, _ = run.run_simulation(compiled)
    actual, actual_selection, _ = simulate(**named, batch_size=2)
    np.testing.assert_array_equal(expected_selection['passing_mask'],
                                  actual_selection['passing_mask'])
    outputs = [expected, actual]
    for key in ('fluxcals','fluxcalerrs','magnitudes','peakmags'):
        np.testing.assert_array_equal(outputs[0][key],outputs[1][key])
    for key in ('x1','c','peakmjd_true','redshift'):
        if key in outputs[0]['parameters']:
            np.testing.assert_array_equal(outputs[0]['parameters'][key],outputs[1]['parameters'][key])
