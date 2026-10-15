"""Host-galaxy magnitude step (gamma_DM) of the BBC distance fit.

SNANA ``get_gammadm_host`` (SALT2mu.c:16745): a Fermi-function step in a host
property (log stellar mass by default, any ``varname_gamma`` variable in
general), which a model's standardisation subtracts from its distance.
"""

import jax.numpy as jnp


def gammadm(gamma0, gamma1, logmass_cen, logmass_tau, z, host_var, offset,
            use_gamma0):
    """Fermi-function host mass step, less the biasCor SIM_gammaDM offset.

    ``gamma = gamma0 + gamma1*z``; the step is
    ``gamma * (0.5 - 1/(1 + exp(-(host_var - logmass_cen)/logmass_tau)))``.
    ``offset`` is GAMMADM_OFFSET, the midpoint of the biasCor sample's
    SIM_gammaDM range (SALT2mu.c:16228), subtracted at the end
    (SALT2mu.c:16782). ``use_gamma0`` is SNANA's USE_GAMMA0: with it off the
    routine returns zero before evaluating anything (SALT2mu.c:16776), so
    neither ``gamma1`` nor ``logmass_tau`` is read.
    """
    if not use_gamma0:
        return 0.0
    gamma = gamma0 + gamma1 * z
    fermi = 1.0 / (1.0 + jnp.exp(-(host_var - logmass_cen) / logmass_tau))
    return gamma * (0.5 - fermi) - offset
