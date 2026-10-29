"""Distance modulus and its derivatives with respect to the LC parameters.

``mu_at`` is the reference cosmology; the rest project a model's standardisation
onto a distance and its coefficients, which the biasCor, CC-prior and likelihood
all need.
"""

import jax
import jax.numpy as jnp
import numpy as np

from jax_snlc_core.cosmology import distance_modulus
from jax_snlc_core.utils import LIGHT_KM, LN10

from .config import MASK_BIASCOR_MU, parval, use_gamma0

# the bias vector under opt_biascor bit MASK_BIASCOR_MU: mu alone
BIAS_ON_MU = ("mu",)


def mu_at(zref, config, *, n_pad=None):
    """Reference distance modulus, or muref_zpoly standing in for the cosmology.

    cosmodl's polynomial branch (SALT2mu.c:25755) returns dl = 10^(0.2*muref - 5)
    and leaves before the (1+zhel) factor at :25784, so mu = 5log10(dl) + 25 is the
    polynomial itself and zhel never enters.
    """
    if config.muref_zpoly:
        return np.polynomial.polynomial.polyval(np.asarray(zref), config.muref_zpoly)
    count = len(zref) if n_pad is not None else None
    if n_pad is not None and n_pad != count:
        if n_pad < count:
            raise ValueError("cosmology capacity is smaller than the input redshift count")
        zref = np.pad(zref, (0, n_pad - count), mode="edge") if count else np.full(n_pad, .1)
    zref = jnp.asarray(zref)
    result = np.asarray(distance_modulus(
        zref, zref, 1.0 - config.omega_l - config.omega_k, config.omega_l,
        LIGHT_KM / config.h0, config.w0, config.wa, radiation=config.radiation))
    return result if count is None else result[:count]


def mu_at_sim(model, tv, sim, coeffs, config, gammadm_offset):
    """muFit of muresid_biasCor (SALT2mu.c:13596-13676) for a biasCor event.

    The LC parameters standardised at the event's own SIM coefficients, plus the
    host mass step. SNANA builds that step from the input parval, not from the
    event's SIM_gammaDM (:13628-13634; the per-event line is commented out at
    :13622), leaves it out entirely unless USE_GAMMA0 (:16776), and adds it where
    fcn subtracts it (:13676 against :5477). With the bias on mu every
    coefficient keeps the input parval too (:13658-13661).
    """
    gate = use_gamma0(model, config)
    nuis = {n: parval(config, n) for n in model.nuisance_spec}
    own_coeffs = bias_parameters(model, config) != BIAS_ON_MU
    for name, sim_col, _ in coeffs:
        if own_coeffs and name != model.gammadm_coeff:  # the step keeps SNANA's parval
            nuis[name] = sim[sim_col]
    lcpar = {n: tv.lcpar[n] for n in model.lcpar_names}
    host = {"z": tv.zhd, "gamma_var": tv.logmass,
            "gammadm_offset": gammadm_offset, "use_gamma0": gate}
    mu = np.asarray(model.compute_mu(lcpar, nuis, host, config._u_flags,
                                     apply_mass_step=False))
    if not gate:
        return mu
    # the mass step is the value of the coefficient axis it is gridded on
    axis = [name for name, _, _ in model.biascor_coeffs].index(model.gammadm_coeff)
    dmhost = model.biascor_coeff_values(nuis, host, config._u_flags)[axis]
    return mu + np.asarray(dmhost)


def bias_parameters(model, config) -> tuple:
    """The parameters the biasCor corrects: the LC parameters, or mu alone under
    opt_biascor bit MASK_BIASCOR_MU (ILCPAR_MIN/MAX, SALT2mu.c:11315). A model
    whose one LC parameter is the distance corrects it either way."""
    if config.opt_biascor & MASK_BIASCOR_MU and len(model.lcpar_names) > 1:
        return BIAS_ON_MU
    return model.lcpar_names


def bias_values(model, biascor_tv, config):
    """Each biasCor event's fitted and true bias parameters, ``{name: array}``
    each. For the bias on mu the fitted value is mu at the event's own SIM
    coefficients less M0 and its SIM mass step, the true one SIM_MUz
    (compute_more_TABLEVAR, SALT2mu.c:9149-9161)."""
    if bias_parameters(model, config) != BIAS_ON_MU:
        return dict(biascor_tv.lcpar), model.sim_lcpar(biascor_tv)
    sim = biascor_tv.sim_truth
    truth = {name: sim_col for name, sim_col, _ in model.biascor_coeffs}
    nuis = {name: v[0] for name, v in model.nuisance_spec.items()}
    for name, sim_col in truth.items():
        if name != model.gammadm_coeff:      # the step is subtracted as its truth
            nuis[name] = sim[sim_col]
    lcpar = {n: biascor_tv.lcpar[n] for n in model.lcpar_names}
    host = {"z": biascor_tv.zhd, "gamma_var": biascor_tv.logmass}
    mu = np.asarray(model.compute_mu(lcpar, nuis, host, config._u_flags,
                                     apply_mass_step=False))
    mass_step = (sim[truth[model.gammadm_coeff]]
                 if model.gammadm_coeff is not None else 0.0)
    return ({"mu": mu - model.m0_reference - mass_step},
            {"mu": np.asarray(biascor_tv.mumodel)})


def bias_coef(v, names, xp=np):
    """The coefficient of each bias parameter in mu, ``[..., k]``: the
    LC-parameter jacobian ``v``, or 1 for the bias on mu itself (get_muBias
    MUCOEF, SALT2mu.c:16578-16581)."""
    if names == BIAS_ON_MU:
        return xp.ones(v.shape[:-1] + (1,), dtype=v.dtype)
    return v


def mu_lcpar_coef(model, names, nuis, z, logmass, u_flags):
    """d(mu)/d(lcpar) per event via autodiff of compute_mu, vmapped over
    (z, logmass) -> ``[n, k]``. The single gradient kernel for the loss, the
    biasCor build, and the output columns. A nuisance value given as an
    array of length n is that event's own, and is vmapped over too."""
    per_event = {k: v for k, v in nuis.items() if jnp.ndim(v)}
    shared = {k: v for k, v in nuis.items() if not jnp.ndim(v)}

    def mu_of_vec(vec, zi, mi, own):
        lcdict = {name: vec[i] for i, name in enumerate(names)}
        # the mass step does not depend on the LC parameters, so it drops
        # out of this jacobian
        host = {"z": zi, "gamma_var": mi}
        return model.compute_mu(lcdict, {**shared, **own}, host, u_flags,
                                apply_mass_step=False)

    n = z.shape[0]
    vec0 = jnp.zeros((n, len(names)))
    return jax.vmap(jax.jacfwd(mu_of_vec))(vec0, z, logmass, per_event)


def mu_variance_scaled(v, cov, zhd, lensing_zpar, sigint_sq=0.0, xp=np):
    """The part of fcn_muerrsq (SALT2mu.c:6134) that muCOVscale multiplies: LC fit +
    intrinsic + lensing. The vpec term is the part it leaves alone (:5468), so that
    stays separate: add ``muerr_vpec_sq`` for the whole mu-variance.

    ``v`` is ``d(mu)/d(lcpar)`` as ``[n, k]`` from ``mu_lcpar_coef`` or
    ``sim_mu_coef``. sigInt lands on the mB diagonal of COVINT, so it reaches mu
    weighted by that coefficient (get_COVINT_model, :23186).

    ``xp`` is the array module for the one non-polymorphic call, the einsum: numpy for
    the init-time builds, jnp for the loss, where the arrays are tracers. Defaulting it
    to jnp instead would silently narrow the init-time maths to float32 wherever x64 is
    off, so it is explicit rather than inferred from the argument type.
    """
    return (xp.einsum("ni,nij,nj->n", v, cov, v)
            + sigint_sq * v[:, 0] ** 2
            + (lensing_zpar * zhd) ** 2)


def muerr_vpec_sq(zhd, zmuerr):
    """SNANA fcn_muerrz OPT=1 low-z form, squared (zmuerr = vpecerr/c)."""
    muerr = (5.0 / LN10) * (zmuerr / zhd) * (1.0 + zhd) / (1.0 + zhd / 2.0)
    return muerr ** 2

def mu_coef(model, nuis, u_flags) -> np.ndarray:
    """numpy d(mu)/d(lcpar) at scalar nuisance values, z=0 (get_muBias MUCOEF)."""
    names = model.lcpar_names
    return np.asarray(mu_lcpar_coef(model, names, nuis, jnp.zeros(1),
                                    jnp.zeros(1), u_flags)[0])

def sim_mu_coef(model, sim, n, u_flags) -> np.ndarray:
    """d(mu)/d(lcpar) at each event's sim coefficients and z=0, ``[n, n_lcpar]``:
    the VEC for ``muerrsq = VEC . COV . VEC`` (get_muBias MUCOEF)."""
    nuis = {name: v[0] for name, v in model.nuisance_spec.items()}
    for name, sim_col, _ in model.biascor_coeffs:
        nuis[name] = jnp.asarray(sim[sim_col])
    zeros = jnp.zeros(n)
    return np.asarray(mu_lcpar_coef(model, model.lcpar_names, nuis, zeros, zeros,
                                    u_flags))
