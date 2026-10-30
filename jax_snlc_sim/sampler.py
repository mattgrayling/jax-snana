"""Per-SN parameter sampling for the simulator, in two draws either side of
the host.

The placement (:func:`draw_placement`) is where and when an event happens:

    - peak MJD, and zCMB by dN/dz CDF inversion, with the SIMLIB entry's say
    - the SIMLIB entry, and so the SN's position
    - VPEC + galactic-motion redshift corrections (zCMB to zHEL)
    - the distance modulus, weak lensing and MUSHIFT (gen_distanceMag)
    - MWEBV map lookup + smearing (SFD98, resolved at load)

The population is what the event is:

    - the SUBPOP class and the model's parameters (via the SNModel hook),
      with the scatter pre-draws (:func:`draw_model`)
    - AV / RV / EBV sampling (:func:`draw_dust`)
    - the GENMAG_OFF_GLOBAL grey offset (:func:`draw_grey_offset`)

Choosing the host galaxy and everything that follows from it is
:mod:`jax_snlc_sim.hostlib.association`.

Sampling holds no state: every random quantity's per-SN keys come from
:mod:`jax_snlc_core.random`, so the same SN draws the same values
regardless of batch boundaries and streams never interfere. The per-field
and dN/dz tables read here are built once by ``build_simulation``.
"""

import logging
from collections.abc import Mapping
from dataclasses import replace
from functools import partial
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from jax_snlc_core.cosmology import (ANISOTROPY_V04_PARS, cos_sep_cmb_apex, cosmo_params,
                                     distance_modulus, zcmb_to_zhelio)
from jax_snlc_core.dust import ebv_on_device
from jax_snlc_core.logging import SNANAError
from jax_snlc_core.snana_files import SNANA_NULLFLOAT, canonical_property_name
from jax_snlc_core.utils import LIGHT_KM
from jax_snlc_sim.population import HOST_DUST_PARAMETERS
from jax_snlc_sim.distributions import (GenExpHalfGauss, GenGaussAsym, distribution_shifts,
                                        pdf_exp_halfgauss, pdf_gengauss_asym,
                                        sample_exp_halfgauss, sample_gengauss_asym,
                                        sample_truncated_gaussian)
from jax_snlc_sim.genpdf import pdf_genpdf, sample_genpdf, sample_genpdf_conditional
from jax_snlc_sim.lensing import lensing_offset
from jax_snlc_sim.population import (checked_genpdf_maps, component_names, outside_map,
                                     stream_name, validate_call_settings)
from jax_snlc_sim.scatter import (draw_phase_scatter, draw_coherent_scatter,
                                   draw_filter_scatter, draw_model_scatter,
                                   scale_model_scatter)
from jax_snlc_core.random import event_keys
from jax_snlc_sim.simlib import ZMIN_SIMLIB_REGEN, draw_simlib_entry, simlib_field_index
from jax_snlc_sim.zvariation import REDSHIFT, shift_array

logger = logging.getLogger(__name__)


def prepare_population_functions(simulator, names, registrations):
    """Resolve each function's dependencies against the simulator's population draw order.

    ``registrations`` maps each kind ('settings', 'sample', 'relation') to
    ``{population name: (function, dependencies)}``, the names canonical.
    Dependency names may use input aliases. The model and dust can precede
    host selection when its weights require them.
    """
    functions = []
    registered = set()
    populations = simulator.model_settings.populations
    model_order = list(simulator.model.drawn_parameters(populations))
    order = [*model_order, *[name for name in ('RV', 'AV', 'EBV_HOST') if name in populations],
             'GENMAG_OFF_GLOBAL']
    redshifts = {'REDSHIFT_CMB', 'REDSHIFT_HELIO'}
    for kind, supplied in registrations.items():
        for name, (function, dependencies) in supplied.items():
            if name in registered:
                raise SNANAError(f"Multiple population functions for {name}",
                                 function_name="build_sbi_pipeline")
            if name == 'GENMAG_OFF_GLOBAL' and kind != 'relation':
                raise SNANAError("GENMAG_OFF_GLOBAL accepts a deterministic relation",
                                 function_name="build_sbi_pipeline")
            deps = []
            if name in model_order:
                before_host = simulator.model_before_host
            else:
                before_host = name != 'GENMAG_OFF_GLOBAL' and simulator.dust_before_host
            earlier = order[:order.index(name)]
            # EBV is converted into AV by the completed dust stage.
            if name == 'GENMAG_OFF_GLOBAL':
                earlier = [quantity for quantity in earlier if quantity != 'EBV_HOST']
            if before_host and not simulator.model_before_host:
                earlier = [quantity for quantity in earlier if quantity not in model_order]
            for dep in dependencies:
                if dep.startswith('HOST.'):
                    column = dep.removeprefix('HOST.')
                    if (before_host or simulator.hostlib is None
                            or column not in simulator.hostlib.selector.varnames):
                        raise SNANAError(f"{name} requires unavailable host column {column}; "
                                         "host dependencies must be loaded and drawn after host selection",
                                         function_name="build_sbi_pipeline")
                elif dep not in redshifts:
                    dep = names.get(dep.lower(), dep)
                    if dep not in earlier:
                        raise SNANAError(f"{name} depends on {dep}, which is not drawn before it",
                                         function_name="build_sbi_pipeline")
                deps.append(dep)
            if name == 'GENMAG_OFF_GLOBAL' and simulator.model_settings._offset_shifts.get(name):
                raise SNANAError(f"A relation and configured variations both set {name}",
                                 function_name="build_sbi_pipeline")
            functions.append((name, kind, function, tuple(deps)))
            registered.add(name)
    return tuple(functions)


@partial(jax.jit, static_argnums=(1,))
def _generate_redshifts_jit(base_key, n_sne, global_start_idx, cdf_vals, cdf_z):
    """JIT-compiled redshift generation via numerical CDF inversion."""
    global_indices = jnp.arange(n_sne) + global_start_idx
    keys = event_keys(base_key, 'REDSHIFT', global_indices)
    uniforms = jax.vmap(jax.random.uniform)(keys)
    return jnp.interp(uniforms, cdf_vals, cdf_z)


def coord_shift_from_polar(r, phi, dec):
    """(RA, DEC) offsets for a polar displacement (r, phi) at declination `dec`.

    SNANA takes one COSDEC from the SN position and applies the resulting
    offsets to the SN and every host (snlc_sim.c:13139-13153).
    """
    return {'ra': r * jnp.cos(phi) / jnp.cos(jnp.radians(dec)),
            'dec': r * jnp.sin(phi), 'radius': r, 'phi': phi}


# why an event was rejected before its light curve was simulated
PREFILTER_REASON_CODES = {
    'PASS': 0,
    'NEPOCH': 1,
    'HOSTLIB_NOHOST': 4,
}


def _ebv_igm_shift(model, redshifts_cmb, n_sne):
    """EBV_IGM z-shift added to the base MWEBV (snlc_sim.c:13654);
    not applied in the GENRANGE_MWEBV branch."""
    entry = model._ebv_igm_shift
    if entry is None or redshifts_cmb is None:
        return jnp.zeros(n_sne, dtype=jnp.float32)
    return jnp.asarray(shift_array(entry, redshifts_cmb), dtype=jnp.float32)


def _quantity_values(entry, redshifts, host_properties, drawn=None):
    """The per-SN values of the quantity a shift depends on: the redshift, a
    HOSTLIB column, or a population parameter drawn before it, in the
    model's declaration order (the general form of `SALT2BETA_cPOLY`,
    snlc_sim.c:14071)."""
    if entry.depends_on == REDSHIFT:
        return redshifts
    if drawn and entry.depends_on in drawn:
        return drawn[entry.depends_on]
    column = canonical_property_name(entry.depends_on)
    if host_properties is not None and column in host_properties:
        return host_properties[column]
    available = sorted(host_properties) if host_properties else 'none'
    drawn_so_far = sorted(drawn) if drawn else 'none'
    raise SNANAError(
        f"The shift on {entry.parname} needs '{entry.depends_on}', which is "
        f"neither a host column this run carries nor a parameter drawn "
        f"before it. Host columns: {available}. Drawn so far: "
        f"{drawn_so_far}.",
        function_name="_quantity_values")


def _shift_arrays(param_name, shifts, redshifts, host_properties, drawn):
    """The per-SN shift of each component, and the static set of components
    the shifts touch. Several shifts on one component sum, so redshift
    evolution and an environmental relation compose."""
    field_map, fields = component_names(param_name)
    arrays, active = [], set()
    for f in fields:
        total = jnp.zeros_like(redshifts)
        # the sum is over floats, so the order is part of the result
        for name in field_map.get(f, ()):
            for entry in shifts:
                if entry.parname == name:
                    total = total + shift_array(entry, _quantity_values(
                        entry, redshifts, host_properties, drawn))
                    active.add(f)
        arrays.append(total)
    return tuple(arrays), frozenset(active)


def _generate_redshifts(simulator, n_sne: int,
                        global_start_idx: int,
                        template_budget=None, *, key) -> jnp.ndarray:
    """Generate redshifts via CDF inversion (comoving-volume-weighted).

    With DNDZ_PEC1A the peculiar-Ia events invert their own CDF; the same
    uniform draw feeds both, so only the curve read differs.
    """
    z = _generate_redshifts_jit(
        key, n_sne, global_start_idx,
        simulator.redshift_cdf_vals, simulator.redshift_cdf_z)
    if simulator.redshift_cdf_vals_pec1a is None:
        return z

    z_pec1a = _generate_redshifts_jit(
        key, n_sne, global_start_idx,
        simulator.redshift_cdf_vals_pec1a, simulator.redshift_cdf_z)
    return jnp.where(
        simulator.model.pec1a_flags(n_sne, global_start_idx,
                                        template_budget),
        z_pec1a, z)


def field_coords(simulator, params):
    """Per-SN SIMLIB pointing (ra, dec) in degrees, from ``field_idx``."""
    idx = params['field_idx']
    ras = jnp.asarray(simulator.field_data['ra'])
    decs = jnp.asarray(simulator.field_data['dec'])
    return jnp.take(ras, idx), jnp.take(decs, idx)


def _lookup_sfd98(simulator, field_indices: np.ndarray, sky_coords=None
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Look up SFD98 E(B-V) for SIMLIB fields.

    Reads the per-field values resolved at load. ``sky_coords`` overrides
    the field position with per-SN (ra, dec) under SN2GAL_RADEC, which needs
    the sky map itself; it becomes device-resident on first use, so a run
    without that bit never pays for it.

    Returns ``(ebv, ebv_err)`` arrays of shape (n_sne,).
    """
    if sky_coords is not None:
        ra, dec = sky_coords
        ebv = ebv_on_device(simulator.sfd_data, ra, dec).astype(jnp.float32)
        return ebv, ebv / 6.0
    ebv = jnp.take(jnp.asarray(simulator.field_sfd_ebv), field_indices)
    return ebv, ebv / 6.0


def _gen_mwebv(simulator, global_indices: jnp.ndarray,
               field_indices: jnp.ndarray, redshifts_cmb: jnp.ndarray,
               sky_coords=None, *, key
               ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Generate MWEBV following SNANA's gen_MWEBV hierarchy.

    Returns ``(mwebv_map, mwebv_err, mwebv_smear)``, each shape (n_sne,).
    ``mwebv_map`` is the measured map/file value (MWEBV column),
    ``mwebv_err`` is the uncertainty, and ``mwebv_smear`` is the
    true value applied to extinction (SIM_MWEBV).
    """
    mw_dust = simulator.mw_dust
    opt_mwebv = mw_dust.opt_mwebv
    sigma_ratio = mw_dust.sigma_ratio
    sigma_fixed = mw_dust.sigma
    shift = mw_dust.shift
    scale = mw_dust.scale
    genrange = mw_dust.genrange

    n_sne = field_indices.shape[0]
    zeros = jnp.zeros(n_sne, dtype=jnp.float32)

    # with no extinction the SIMLIB value is still reported
    if not mw_dust.extinction_applied:
        mwebv = (jnp.take(jnp.asarray(simulator.field_mwebv), field_indices)
                 if opt_mwebv == 1 else zeros)
        return mwebv.astype(jnp.float32), zeros, zeros

    # GENRANGE_MWEBV replaces only the map value (snlc_sim.c:13632-13685)
    if genrange[0] >= 0:
        mwebv = jax.vmap(
            lambda k: jax.random.uniform(
                k, minval=genrange[0], maxval=genrange[1]))(
            event_keys(key, 'MWEBV', global_indices)
        ).astype(jnp.float32)
        mwebv_err = zeros
    elif opt_mwebv == 1:
        # SIMLIB field MWEBV, SFD98 where it is <= 0
        mwebv = jnp.take(jnp.asarray(simulator.field_mwebv), field_indices)
        if simulator.field_sfd_ebv is not None:
            sfd_ebv, _ = _lookup_sfd98(simulator, field_indices, sky_coords)
            mwebv = jnp.where(mwebv <= 0, sfd_ebv, mwebv)
        mwebv = mwebv + _ebv_igm_shift(simulator.model_settings, redshifts_cmb, n_sne)
        mwebv_err = zeros
    elif opt_mwebv == 2:
        mwebv, mwebv_err = _lookup_sfd98(simulator, field_indices, sky_coords)
        mwebv = mwebv + _ebv_igm_shift(simulator.model_settings, redshifts_cmb, n_sne)
    else:
        # OPT_MWEBV 3, Schlafly+11: 0.86 × SFD98
        sfd_ebv, _ = _lookup_sfd98(simulator, field_indices, sky_coords)
        mwebv = (0.86 * sfd_ebv).astype(jnp.float32)
        mwebv_err = (0.05 * mwebv).astype(jnp.float32)
        mwebv = mwebv + _ebv_igm_shift(simulator.model_settings, redshifts_cmb, n_sne)

    # error chain (snlc_sim.c:13663-13685)
    if opt_mwebv == 1:
        mwebv_err = jnp.sqrt(
            sigma_fixed**2 + (sigma_ratio * mwebv)**2).astype(jnp.float32)
    elif sigma_fixed > 0:
        mwebv_err = jnp.full(n_sne, sigma_fixed, dtype=jnp.float32)
    elif sigma_ratio == 0 and sigma_fixed == 0:
        mwebv_err = zeros

    gauss_ran = jax.vmap(
        lambda k: jax.random.truncated_normal(k, -3.0, 3.0))(
        event_keys(key, 'MWEBV_SMEAR', global_indices)
    ).astype(jnp.float32)
    mwebv_smear = (mwebv + mwebv_err * gauss_ran + shift) * scale

    return (mwebv.astype(jnp.float32), mwebv_err.astype(jnp.float32),
            mwebv_smear.astype(jnp.float32))


def draw_mwebv(simulator, params: Dict[str, jnp.ndarray], *,
               key) -> Dict[str, jnp.ndarray]:
    """gen_MWEBV at the SN's position: its SIMLIB pointing, or under
    SN2GAL_RADEC its place on its host, where SNANA defers the lookup to
    (snlc_sim.c:12878-12880, load_SNHOST_GENLC_COORDS). Only the SFD98 paths
    read the position; OPT_MWEBV=1 reads the SIMLIB field value either way.
    Returns the map value, its error and the smeared truth."""
    sky = ((params['ra'], params['dec']) if simulator.host_settings.mskopt & 8
           else None)
    mwebv, mwebv_err, mwebv_smear = _gen_mwebv(
        simulator, params['sn_global_idx'], params['field_idx'],
        params['sim_redshift_cmb'], sky_coords=sky, key=key)
    return {'mwebv': mwebv, 'mwebv_err': mwebv_err, 'sim_mwebv': mwebv_smear}


def anisotropy_cos_sep(simulator, ra, dec):
    """Per-SN cos(angle to the CMB apex) at the SN's position (gen_dLmag
    takes the SN's galactic coords, snlc_sim.c:17626)."""
    if not simulator.cosmology.anisotropy_modelname:
        return 0.0
    return cos_sep_cmb_apex(ra, dec)


def _sample_wv07_av(host, keys):
    """WV07 host-AV: exponential(tau=0.4) + half-Gaussian(sigma^2=0.01) over
    GENRANGE_AV, exponential weight scaled by WV07_REWGT_EXPAV (snlc_sim.c:
    17953). Inverse-CDF draw (batch-invariant) reproduces SNANA's rejection PDF."""
    hd = host
    av_lo, av_hi = hd.av_range
    tau, sqsigma = 0.4, 0.01
    aexp, bexp = 1.0 / tau, 1.0 / np.sqrt(sqsigma * 2.0 * np.pi)
    if hd.wv07_rewgt_expav > -1.0e-9:
        aexp *= hd.wv07_rewgt_expav
    grid = np.linspace(av_lo, av_hi, 4001)
    pdf = aexp * np.exp(-grid / tau) + bexp * np.exp(-0.5 * grid * grid / sqsigma)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(grid))])
    cdf /= cdf[-1]
    u = jax.vmap(lambda k: jax.random.uniform(k))(keys)
    return jnp.interp(u, jnp.asarray(cdf), jnp.asarray(grid)).astype(jnp.float32)


def assign_simlib_field(simulator, global_indices, redshifts, *,
                        key):
    """Per-SN SIMLIB entry and the sky position it implies.

    SIMLIB_NREPEAT and SIMLIB_MAXRANSTART drive the cycle, which the
    observation scheduler repeats from the same formula. With CUTWIN_REDSHIFT
    windows the entry is instead drawn among those whose window holds the
    redshift, one per NREPEAT block (keep_SIMLIB_HEADER, snlc_sim.c:20144).
    """
    n_fields = simulator.field_data['ra'].shape[0]
    s = simulator.survey
    if simulator.simlib_windows is None:
        field_indices = simlib_field_index(
            key, global_indices, n_fields,
            s.simlib_nrepeat, s.simlib_maxranstart,
            simulator.field_start_index)
    else:
        blocks = global_indices // s.simlib_nrepeat
        field_indices = draw_simlib_entry(
            simulator.simlib_windows, redshifts,
            event_keys(key, 'SIMLIB_ENTRY', blocks))
    return (field_indices,
            *field_coords(simulator, {'field_idx': field_indices}))


def _entry_gengauss(field_data, name, field_indices):
    """Per-event entry (lo, hi, sigma), or None when no entry defines it."""
    table = field_data['gengauss'].get(name)
    if table is None:
        return None
    values = jnp.take(jnp.asarray(table), field_indices, axis=0)
    return tuple(values[:, i] for i in range(3))


def _simlib_redshifts(simulator, redshifts, field_indices, global_indices,
                      *, key):
    """The entry's say over zCMB. A GENRANGE_REDSHIFT window redraws it from
    the same dN/dz inside the window (genz_hubble, snlc_sim.c:22358). The
    entry's REDSHIFT under USE_SIMLIB_REDSHIFT, or its DISTANCE under
    USE_SIMLIB_DISTANCE through the distance modulus, is a zero-width window
    and sets zCMB to that value (regen_SIMLIB_GENRANGES,
    snlc_sim.c:22360-22370). The last of these keys in the entry header
    wins, and a window applies only when its lower edge lies above
    ``ZMIN_SIMLIB_REGEN``. Each rebinds the drawn value rather than adding
    to it.
    """
    windows = simulator.field_data['redshift_windows']
    if windows is None:
        return redshifts
    cdf_z, cdf_vals = simulator.redshift_cdf_z, simulator.redshift_cdf_vals
    window = jnp.take(jnp.asarray(windows), field_indices, axis=0)
    has = window[:, 0] > ZMIN_SIMLIB_REGEN
    lo = jnp.where(has, window[:, 0], 0.0)
    hi = jnp.where(has, window[:, 1], 0.0)
    u = jax.vmap(jax.random.uniform)(
        event_keys(key, 'SIMLIB_GENRANGE_REDSHIFT', global_indices))
    c_lo, c_hi = jnp.interp(lo, cdf_z, cdf_vals), jnp.interp(hi, cdf_z, cdf_vals)
    inside = jnp.where(lo == hi, lo,
                       jnp.interp(c_lo + u * (c_hi - c_lo), cdf_vals, cdf_z))
    return jnp.where(has, inside.astype(redshifts.dtype), redshifts)


def draw_peculiar_velocity(cosmology, batch_n_sne, redshifts, sn_ras,
                           sn_decs, vpec_keys, vpec_smear_keys, from_hostlib):
    """VPEC and the heliocentric redshift it produces.

    Returns ``(redshifts_helio, vpec_true, vpec_smear, vpec_err)``. VPEC_FORCE
    pins the velocity (snlc_sim.c:17204); otherwise a HOSTLIB VPEC
    (``from_hostlib``, USEVPEC) leaves it zero for the host to set, and
    GENSIGMA_VPEC draws it (gen_redshift_helio, :17226-17236). VPEC_ERR
    smears the reported value. A VEL_CMBAPEX of at least 1 adds galactic
    motion first, so zHEL is zCMB plus galactic motion plus VPEC.
    """
    cosmo_cfg = cosmology
    gensigma_vpec = cosmo_cfg.gensigma_vpec
    vpec_err_config = cosmo_cfg.vpec_err

    if cosmo_cfg.vel_cmbapex >= 1.0:
        redshifts_helio_galmotion = zcmb_to_zhelio(redshifts, sn_ras, sn_decs)
    else:
        redshifts_helio_galmotion = redshifts

    if cosmo_cfg.vpec_force < 1.0e8:  # snlc_sim.c:17204
        vpec_true = jnp.full(batch_n_sne, cosmo_cfg.vpec_force, dtype=jnp.float32)
        redshifts_helio = (1.0 + redshifts_helio_galmotion) * (1.0 + vpec_true / LIGHT_KM) - 1.0
        # the measured VPEC follows APPLY_VPEC_SMEAR = VPEC_ERR < GENSIGMA_VPEC
        # (gen_zsmear, snlc_sim.c:17441-17459); with the defaults it stays 0
        if vpec_err_config < gensigma_vpec:
            vpec_smear = vpec_true + vpec_err_config * jax.vmap(
                jax.random.normal)(vpec_smear_keys)
        else:
            vpec_smear = jnp.zeros(batch_n_sne)
    elif gensigma_vpec > 0.0 and not from_hostlib:
        vpec_true = gensigma_vpec * jax.vmap(jax.random.normal)(vpec_keys)
        redshifts_helio = (1.0 + redshifts_helio_galmotion) * (1.0 + vpec_true / LIGHT_KM) - 1.0
        if vpec_err_config < gensigma_vpec:
            vpec_smear = vpec_true + vpec_err_config * jax.vmap(
                jax.random.normal)(vpec_smear_keys)
        else:
            vpec_smear = jnp.zeros(batch_n_sne)
    else:
        vpec_true = jnp.zeros(batch_n_sne)
        vpec_smear = jnp.zeros(batch_n_sne)
        redshifts_helio = redshifts_helio_galmotion

    vpec_err = jnp.full(batch_n_sne, vpec_err_config, dtype=jnp.float32)
    return redshifts_helio, vpec_true, vpec_smear, vpec_err


def draw_input_checks(simulator):
    """The population-draw inputs SNANA aborts on, as ``(message, failed)``
    pairs, ``failed(redshifts, host_properties, weights)`` giving the failing
    events, ``weights`` the SUBPOP weights (:func:`_subpop_weights`) where a
    VARIATION shifts them.

    A host whose conditional GENPDF value lies outside the map aborts
    (:func:`outside_map`). A SUBPOP weight that its VARIATION takes
    negative aborts, as does an event for which every class's weight is
    zero, since no class can be drawn for it.
    """
    model = simulator.model_settings
    checks = []
    for param_name, column, grid in checked_genpdf_maps(model):
        checks.append((
            f"GENPDF for {param_name}: host '{column}' outside map range "
            f"[{grid[0]}, {grid[-1]}]. Set GENPDF_OPTMASK bit 1 (EXTRAP) to clamp.",
            lambda z, host, weights, column=column, grid=grid:
                outside_map(host[column], grid)))
    if _shifted_subpop_weights(model):
        checks.append((
            "A SUBPOP weight goes negative once its VARIATION is applied. A "
            "class cannot have negative weight; narrow the relation or the "
            "range of the quantity it depends on.",
            lambda z, host, weights: jnp.min(weights, axis=0) < 0.0))
        checks.append((
            "Every SUBPOP weight is zero for this event, so no class can be "
            "drawn for it. Give at least one class a weight that stays "
            "positive over the range its VARIATION depends on.",
            lambda z, host, weights: jnp.sum(weights, axis=0) <= 0.0))
    return checks


def _shifted_subpop_weights(model):
    return any(model.offsets_for('WGT', i) for i in range(len(model.subpops)))


def draw_input_failures(simulator, redshifts, host_properties):
    """Per event, which of :func:`draw_input_checks` its inputs fail first,
    counted from one; zero where none does."""
    weights = (_subpop_weights(simulator, redshifts, host_properties,
                               redshifts.shape[0])
               if _shifted_subpop_weights(simulator.model_settings) else None)
    code = jnp.zeros(redshifts.shape, dtype=jnp.int32)
    checks = draw_input_checks(simulator)
    for k in reversed(range(len(checks))):
        code = jnp.where(checks[k][1](redshifts, host_properties, weights),
                         k + 1, code)
    return code


def _sample_population_parameter(simulator, redshifts, host_properties,
                                 keys_for, param_name, subpop_index, drawn, settings=None):
    """Sample a named population parameter from its record.

    One component means every class draws it the same way, which is the
    ordinary case and the whole of a run with no classes.
    """
    pop = simulator.model_settings.populations.get(param_name)
    if pop is None:
        raise SNANAError(
            f"{param_name} not configured: no distribution or GENPDF found. "
            f"Check that the input file or GENPDF_FILE defines this parameter.")
    keys = keys_for(stream_name(param_name, dict(simulator.model.configured_population)))

    genpdf = pop.genpdf
    if genpdf is not None:
        if genpdf.cond_varnames:
            if host_properties is None:
                raise SNANAError(
                    f"GENPDF for {param_name} has conditioning variables "
                    f"{genpdf.cond_varnames} but no host_properties provided. "
                    f"Use host-first flow (conditional GENPDF requires HOSTLIB).")
            cond_arrays = []
            for cv_name in map(canonical_property_name, genpdf.cond_varnames):
                if cv_name not in host_properties:
                    raise SNANAError(
                        f"GENPDF for {param_name} requires conditioning "
                        f"variable '{cv_name}' not in host_properties. "
                        f"Available: {list(host_properties.keys())}")
                cond_arrays.append(host_properties[cv_name])
            return jax.vmap(
                lambda key, *values: sample_genpdf_conditional(key, genpdf, *values)
            )(keys, *cond_arrays)
        return jax.vmap(lambda key: sample_genpdf(key, genpdf))(keys)

    per_class = [_draw(param_name, dist, shifts, keys, redshifts,
                       host_properties, drawn, settings)
                 for dist, shifts in zip(pop.distributions, pop.shifts)]
    if len(per_class) == 1:
        return per_class[0]
    return jnp.take_along_axis(jnp.stack(per_class), subpop_index[None, :], axis=0)[0]


def _population_weight(simulator, param_name, value, host_properties,
                       subpop_index, settings=None):
    """prob(nominal)/prob(reweighted) at the drawn value, one where the
    parameter's distribution carries no GENPDF_EXPON_REWGT. A GENPDF map is
    read at the map's own coordinate; a class-dependent parameter reads the
    class's own distribution, at its unshifted parameters as SNANA reads
    INPUTS.GENGAUSS_SALT2x1 rather than the event's shifted copy."""
    pop = simulator.model_settings.populations[param_name]
    genpdf = pop.genpdf
    if genpdf is not None:
        if genpdf.expon_rewgt == 1.0:
            return 1.0
        cond = [host_properties[name] for name in genpdf.cond_varnames]
        at = jnp.log10(value) if genpdf.is_log else value
        prob = jax.vmap(lambda x, *c: pdf_genpdf(genpdf, x, *c))(at, *cond)
        return jnp.power(prob, 1.0 - genpdf.expon_rewgt)
    ratios = []
    for dist in pop.distributions:
        rewgt = dist.prob_expon_rewgt
        if rewgt == 1.0:
            ratios.append(jnp.ones_like(value))
            continue
        pdf = (pdf_exp_halfgauss if isinstance(dist, GenExpHalfGauss)
               else pdf_gengauss_asym)
        if settings:
            values = dict(settings)
            if isinstance(dist, GenGaussAsym):
                for field in ('sigma', 'sigma2', 'skew'):
                    if field in values:
                        array = jnp.asarray(values[field])
                        values[field] = (array[..., 0], array[..., 1])
            dist = replace(dist, **values)
        ratios.append(pdf(value, dist) / pdf(value, dist, rewgt))
    if len(ratios) == 1:
        return ratios[0]
    return jnp.take_along_axis(jnp.stack(ratios),
                               subpop_index[None, :], axis=0)[0]


def _draw(param_name, dist, shifts, keys, redshifts, host_properties, drawn, settings=None):
    """One component's draw, with the shifts on it applied."""
    if not shifts and not settings:
        sample = (sample_exp_halfgauss if isinstance(dist, GenExpHalfGauss)
                  else sample_gengauss_asym)
        return jax.vmap(lambda key: sample(key, dist))(keys)
    arrays, active = _shift_arrays(param_name, shifts, redshifts,
                                   host_properties, drawn)
    settings = dict(settings or {})
    bounds = settings.pop('range', None)
    if bounds is not None:
        bounds = jnp.broadcast_to(jnp.asarray(bounds), (keys.shape[0], 2))
    if settings:
        offsets, changed = distribution_shifts(dist, settings)
        arrays = tuple(value + offset for value, offset in zip(arrays, offsets))
        active = active | changed
    if isinstance(dist, GenExpHalfGauss):
        return jax.vmap(lambda key, limits, *values: sample_exp_halfgauss(
            key, dist, values, bounds=limits))(keys, bounds, *arrays)
    return jax.vmap(lambda key, limits, *values: sample_gengauss_asym(
        key, dist, values, active, bounds=limits))(keys, bounds, *arrays)


def _subpop_weights(simulator, redshifts, host_properties, n_sne):
    """Per-event relative weight of every class, as a (n_class, n_sne) array.
    A class's WGT may itself carry a VARIATION."""
    rows = []
    for index, sp in enumerate(simulator.model_settings.subpops):
        w = jnp.full(n_sne, sp.weight)
        for entry in simulator.model_settings.offsets_for('WGT', index):
            w = w + shift_array(entry, _quantity_values(
                entry, redshifts, host_properties))
        rows.append(w)
    return jnp.stack(rows)


def _draw_subpop(simulator, redshifts, host_properties, n_sne, keys):
    """The class index per event, drawn in proportion to the weights, or None
    when the run declares no classes."""
    if not simulator.model_settings.subpops:
        return None
    weights = _subpop_weights(simulator, redshifts, host_properties, n_sne)
    cumulative = jnp.cumsum(weights, axis=0)
    draw = jax.vmap(jax.random.uniform)(keys) * cumulative[-1]
    return jnp.sum(draw[None, :] > cumulative, axis=0)


def grey_offset(simulator, subpop_index, redshifts, host_properties, drawn,
                n_sne):
    """GENMAG_OFF_GLOBAL per SN, the drawn class's value where classes are
    declared, with the shifts a ZVARIATION or VARIATION puts on it
    (snlc_sim.c:12941). A class's own shift reaches only its own events.
    """
    model = simulator.model_settings
    if model.subpops:
        offsets = jnp.asarray([sp.genmag_off_global for sp in model.subpops])
        grey = offsets[subpop_index]
    else:
        grey = jnp.full(n_sne, model.genmag_off_global)

    per_class = model._offset_shifts.get('GENMAG_OFF_GLOBAL', ())
    for index, shifts in enumerate(per_class):
        for entry in shifts:
            shift = shift_array(entry, _quantity_values(
                entry, redshifts, host_properties, drawn))
            grey = grey + (shift if len(per_class) == 1
                           else jnp.where(subpop_index == index, shift, 0.0))
    return grey


def _draw_host_dust(simulator, batch_n_sne, sample_param, av_keys):
    """Host-galaxy AV and RV, and the pair SNANA reports.

    RV comes first because the EBV path takes AV = EBV * RV per SN. GENAV_WV07
    replaces the EBV/AV draw entirely. SNANA leaves SIM_AV/SIM_RV at NULLFLOAT
    unless gen_modelPar_dust runs (snlc_sim.c:9612-9613, gated at :14335); the
    branches here are its DOGEN_AV decision (prep_dustFlags, :7968-7975). The
    drawn values stay 0 and 3.1 for the extinction maths, only the reported
    pair differs.
    """
    if 'RV' in simulator.model_settings.populations:
        rv = sample_param('RV')
    else:
        rv = jnp.full(batch_n_sne, 3.1)

    if simulator.host_settings.genav_wv07:
        av = _sample_wv07_av(simulator.host_settings, av_keys)
        dogen_av = True
    elif 'EBV_HOST' in simulator.model_settings.populations:
        av = sample_param('EBV_HOST') * rv
        dogen_av = True
    elif 'AV' in simulator.model_settings.populations:
        av = sample_param('AV')
        dogen_av = True
    else:
        av = jnp.zeros(batch_n_sne)
        dogen_av = False

    sentinel = jnp.full(batch_n_sne, SNANA_NULLFLOAT)
    return av, rv, (av if dogen_av else sentinel), (rv if dogen_av else sentinel)


def distance_modulus_at(simulator, redshifts, redshifts_helio,
                        vpec, ra, dec):
    """The distance modulus per SN, with the dipole anisotropy at the SN's
    position (gen_dLmag, snlc_sim.c:17626)."""
    omega_m, _omega_k, omega_l, hubble_distance, w0, wa = cosmo_params(simulator.cosmology)
    _aniso = (ANISOTROPY_V04_PARS
              if simulator.cosmology.anisotropy_modelname else None)
    _cs = anisotropy_cos_sep(simulator, ra, dec)
    return jax.vmap(lambda z, zh, vp, cs: distance_modulus(
        z, zh, omega_m, omega_l, hubble_distance, w0, wa, vp,
        simulator.hz_map, anisotropy=_aniso, cos_sep=cs))(
            redshifts, redshifts_helio, vpec,
            jnp.broadcast_to(jnp.asarray(_cs), redshifts.shape))


def distance_mag(simulator, params, host_properties, *, key):
    """gen_distanceMag (snlc_sim.c:17468): the distance modulus, the
    weak-lensing offset and MUSHIFT at the SN's zCMB, zHEL, VPEC and
    position. Run for the placement, and again when the host moves the SN's
    redshift, then with the host's lensing column."""
    idx = params['sn_global_idx']
    redshifts = params['sim_redshift_cmb']
    mu = distance_modulus_at(
        simulator, redshifts, params['sim_redshift_helio'],
        params['vpec_true'], params['ra'], params['dec'])
    mushift = _draw_mushift(simulator.model_settings, event_keys(key, 'MUSHIFT', idx),
                            mu, simulator.murewgt)
    return {
        'mu': mu,
        'lensdmu': lensing_offset(simulator, redshifts,
                                  event_keys(key, 'LENSDMU', idx),
                                  host_properties),
        'mushift': jnp.zeros_like(redshifts) if mushift is None else mushift,
    }


def _draw_mushift(model, keys, mu, murewgt):
    """MUSHIFT: per-SN offset added to the reported distance modulus, flat
    over its range, or under a DNDZ reweight drawn from the reweight's CDF
    between mu plus either end of the range and returned as the drawn
    distance less mu. An event whose shifted range leaves the CDF's grid
    takes the flat draw (gen_MUSHIFT, snlc_sim.c:17542)."""
    mushift_lo, mushift_hi = model.mushift
    if not (mushift_lo or mushift_hi):
        return None
    u = jax.vmap(jax.random.uniform)(keys)
    flat = mushift_lo + (mushift_hi - mushift_lo) * u
    if murewgt is None:
        return flat
    mu_list, cdf = murewgt
    mu_lo, mu_hi = mu + mushift_lo, mu + mushift_hi
    inside = (mu_lo >= mu_list[0]) & (mu_hi <= mu_list[-1])
    cdf_lo = jnp.interp(mu_lo, mu_list, cdf)
    cdf_hi = jnp.interp(mu_hi, mu_list, cdf)
    mu_rewgt = jnp.interp(cdf_lo + (cdf_hi - cdf_lo) * u, cdf, mu_list)
    return jnp.where(inside, mu_rewgt - mu, flat)


def _simlib_header_override(simulator, sample_param, field_indices, keys_for):
    """USE_SIMLIB_SALT2: a population parameter the entry header names, under
    the same spelling as its GENGAUSS keys, is drawn from the entry's
    GENRANGE/GENSIGMA or pinned to its single value at draw time, so the model
    builds the SED from the overridden value (snlc_sim.c:22194, :22380-22387).
    """
    field_data = simulator.field_data

    def sample_with_override(name):
        val = sample_param(name)
        gengauss = _entry_gengauss(field_data, name, field_indices)
        if gengauss is not None:
            lo, hi, sigma = gengauss
            stream = stream_name(name, dict(simulator.model.configured_population))
            drawn = jax.vmap(sample_truncated_gaussian)(
                keys_for('SIMLIB_' + stream), 0.5 * (lo + hi), sigma, sigma, lo, hi)
            val = jnp.where(jnp.isfinite(lo), drawn.astype(val.dtype), val)
        pinned = jnp.take(jnp.asarray(field_data['headers'][name]),
                          field_indices)
        return jnp.where(~jnp.isnan(pinned), pinned.astype(val.dtype), val)

    return sample_with_override


def _draw_redshift_smear(rates, keys):
    """Observed-redshift smear normals (GENSIGMA_REDSHIFT / GENBIAS_REDSHIFT,
    snlc_sim.c:17405)."""
    if (rates.gensigma_redshift < 0.0 or rates.gensigma_redshift > 0.999
            or not (rates.gensigma_redshift > 0.0
                    or rates.genbias_redshift != 0.0)):
        return None
    return jax.vmap(jax.random.normal)(keys)


def _draw_peak_mjd(simulator, field_indices, peak_keys,
                   smear_keys, header_keys):
    """Peak MJD and the offset SNANA reports it with.

    Uniform over GENRANGE_PEAKMJD, then USE_SIMLIB_PEAKMJD pins it to the
    per-LIBID header value and MJD_EXPLODE pins every SN to the explosion time
    (SNANA sets GENRANGE_PEAKMJD=[MJD,MJD]; snlc_sim.c:2085). GENSIGMA_PEAKMJD
    is keyed on the global index so it stays batch-invariant, and is applied
    post-photometry by gen_peakmjd_smear.
    """
    mjd_min, mjd_max = simulator.rates.peakmjd_range
    # uniform with GENRANGE_PEAKMJD_EXCLUDE cut out (snlc_sim.c:17038-17039)
    exclude_min, exclude_max = simulator.rates.peakmjd_exclude
    gap_lo = max(exclude_min, mjd_min)
    gap = max(min(exclude_max, mjd_max) - gap_lo, 0.0) if exclude_max > 0.0 else 0.0
    peak_mjd = jax.vmap(
        lambda key: jax.random.uniform(key, minval=mjd_min, maxval=mjd_max - gap))(peak_keys)
    peak_mjd = jnp.where(peak_mjd >= gap_lo, peak_mjd + gap, peak_mjd)

    # entry GENRANGE_PEAKMJD, whatever USE_SIMLIB_PEAKMJD (snlc_sim.c:22166-22185, :22375)
    gengauss = _entry_gengauss(simulator.field_data, 'PEAKMJD', field_indices)
    if gengauss is not None:
        lo, hi, sigma = gengauss
        drawn = jax.vmap(sample_truncated_gaussian)(
            header_keys, 0.5 * (lo + hi), sigma, sigma, lo, hi)
        peak_mjd = jnp.where(jnp.isfinite(lo), drawn.astype(peak_mjd.dtype), peak_mjd)
    if simulator.survey.use_simlib_peakmjd:
        _pkov = jnp.take(
            jnp.asarray(simulator.field_data['headers']['PEAKMJD']),
            field_indices)
        peak_mjd = jnp.where(~jnp.isnan(_pkov),
                             _pkov.astype(peak_mjd.dtype), peak_mjd)
    if simulator.model_settings.mjd_explode is not None:
        peak_mjd = jnp.full_like(peak_mjd, simulator.model_settings.mjd_explode)

    gensigma_peakmjd = simulator.rates.gensigma_peakmjd
    peakmjd_smear = (
        jax.vmap(lambda k: jax.random.normal(k))(smear_keys)
        * gensigma_peakmjd) if gensigma_peakmjd > 1.0E-9 else None
    return peak_mjd, peakmjd_smear


def draw_placement(simulator, batch_n_sne: int,
                   global_start_idx: int,
                   template_budget: Optional[np.ndarray] = None,
                   *, key) -> Dict[str, jnp.ndarray]:
    """Where and when each event happens, SNANA's gen_event_driver up to the
    host (snlc_sim.c:12826-12887): the peak MJD, zCMB and the SIMLIB entry
    it picks, the SN's position, the peculiar velocity and so zHEL, the
    distance, and MWEBV unless the host will move the SN. The host and the
    population follow.

    Each SN's keys come from its global index, so the output does not depend
    on the batch split. Every event starts as passing the prefilter; the
    NEPOCH gate and the host draw reject later.
    """
    global_indices = global_start_idx + jnp.arange(batch_n_sne)
    keys_for = _keys_for(key, global_indices)

    redshifts = _generate_redshifts(simulator, batch_n_sne, global_start_idx,
                                    template_budget, key=key)
    field_indices, sn_ras, sn_decs = assign_simlib_field(
        simulator, global_indices, redshifts, key=key)
    redshifts = _simlib_redshifts(
        simulator, redshifts, field_indices, global_indices, key=key)
    peak_mjd, peakmjd_smear = _draw_peak_mjd(
        simulator, field_indices,
        keys_for('PEAKMJD'), keys_for('PEAKMJD_SMEAR'), keys_for('SIMLIB_PEAKMJD'))
    redshifts_helio, vpec_true, vpec_smear, vpec_err = draw_peculiar_velocity(
        simulator.cosmology, batch_n_sne, redshifts, sn_ras, sn_decs,
        keys_for('VPEC'), keys_for('VPEC_SMEAR'),
        from_hostlib=bool(simulator.host_settings.mskopt & 512))

    params = {
        'cid': global_indices + 1,
        'sn_global_idx': global_indices.astype(jnp.int32),
        'field_idx': field_indices,
        'ra': sn_ras,
        'dec': sn_decs,
        'sim_redshift_cmb': redshifts,
        'sim_redshift_helio': redshifts_helio,
        # the host-lookup redshift and match flag, which WRONGHOST_FILE
        # overrides (GENLC.REDSHIFT_HOST, CORRECT_HOSTMATCH; snlc_sim.c:12886)
        'sim_redshift_host': redshifts_helio,
        'CORRECT_HOSTMATCH': jnp.ones(batch_n_sne, dtype=jnp.int32),
        'vpec_true': vpec_true,
        'vpec': vpec_smear,
        'vpec_err': vpec_err,
        'peakmjd': peak_mjd,
        'prefilter_pass': jnp.ones(batch_n_sne, dtype=bool),
        'prefilter_reason_code': jnp.full(
            batch_n_sne, PREFILTER_REASON_CODES['PASS'], dtype=jnp.int16),
    }
    if peakmjd_smear is not None:
        params['peakmjd_smear'] = peakmjd_smear
    params.update(distance_mag(simulator, params, None, key=key))
    if not simulator.host_settings.mskopt & 8:
        params.update(draw_mwebv(simulator, params, key=key))
    return params


def _missing_population(name):
    """The error for a population parameter that is neither drawn nor
    supplied by its model."""
    return SNANAError(
        f"{name} has no distribution: give GENPEAK_{name}, GENSIGMA_{name} "
        f"and GENRANGE_{name}, or a GENPDF map.", function_name="draw_model")


def _population_dependencies(dependencies, params, host_properties, drawn):
    """Values available at this draw, under explicit physical quantity names."""
    values = []
    for name in dependencies:
        if name == 'REDSHIFT_CMB':
            value = params['sim_redshift_cmb']
        elif name == 'REDSHIFT_HELIO':
            value = params['sim_redshift_helio']
        elif name.startswith('HOST.'):
            value = host_properties[name.removeprefix('HOST.')]
        else:
            # Completed model draws retain their native precision in params.
            # drawn also carries this stage's draws and dust population names.
            value = params[name] if name in params else drawn[name]
        values.append(value)
    return values


def _parameter_sampler(simulator, params, host_properties, keys_for,
                       subpop_index, drawn, overrides=None, population_functions=()):
    """A draw of named population parameters that records each value in
    ``drawn``, so a later parameter's VARIATION may read it, and returns
    with it a callable giving SNANA's population weight so far,
    prob(nominal)/prob(reweighted) over every reweighted parameter
    (wgt_population_event, snlc_sim.c:31576)."""
    redshifts = params['sim_redshift_cmb']
    weight = jnp.ones_like(redshifts)

    overrides = overrides or {}
    custom = {name: (kind, function, deps) for name, kind, function, deps in population_functions}

    def sample(param_name):
        nonlocal weight
        given = overrides.get(param_name, {})
        if not isinstance(given, Mapping):
            value = given
        else:
            kind, function, deps = custom.get(param_name, (None, None, ()))
            inputs = _population_dependencies(deps, params, host_properties, drawn)
            keys = keys_for(stream_name(param_name, dict(simulator.model.configured_population)))
            if kind in ('sample', 'relation'):
                value = jax.vmap(lambda key, *args: function(key, *args, **given)
                                 if kind == 'sample' else function(*args, **given))(keys, *inputs)
                if value.shape != redshifts.shape:
                    raise SNANAError(f"Function for {param_name} must return one scalar per event",
                                     function_name="_parameter_sampler")
            else:
                if kind == 'settings':
                    pop = simulator.model_settings.populations[param_name]
                    def settings_for_event(key, *args):
                        settings = function(*args, **given)
                        validate_call_settings(param_name, pop, settings, conditional=True)
                        return {field: jnp.asarray(value) for field, value in settings.items()}
                    given = jax.vmap(settings_for_event)(keys, *inputs)
                value = _sample_population_parameter(
                    simulator, redshifts, host_properties, keys_for, param_name,
                    subpop_index, drawn, given)
                weight = weight * _population_weight(
                    simulator, param_name, value, host_properties, subpop_index, given)
        drawn[param_name] = value
        return value

    return sample, lambda: weight


def _population_values(simulator, params):
    """The drawn population values under their population names, which a
    VARIATION on a later draw may depend on."""
    return {**simulator.model.sn_variables(params),
            **{p.key: params[p.param] for p in HOST_DUST_PARAMETERS
               if p.param and p.param in params}}


def _subpop_index(simulator, params):
    return params['subpop'] if simulator.model_settings.subpops else None


def _keys_for(key, global_indices):
    """Per-SN keys of a named stream at the events' global indices."""
    return lambda name: event_keys(key, name, global_indices)


def draw_model(simulator, params,
               host_properties: Optional[Dict[str, jnp.ndarray]],
               template_budget: Optional[np.ndarray] = None,
               *, key, overrides=None, population_functions=()) -> Dict[str, jnp.ndarray]:
    """What each event is (gen_modelPar, snlc_sim.c:12892-12910): the SUBPOP
    class, the model's parameters with the amplitude from the distance, and
    the per-event smears. SNANA draws them after the host, or before it when
    the host's weight map reads them.

    Returns the new columns. Keys come from each SN's global index, as in
    :func:`draw_placement`.
    """
    batch_n_sne = params['sim_redshift_cmb'].shape[0]
    redshifts = params['sim_redshift_cmb']
    field_indices = params['field_idx']
    keys_for = _keys_for(key, params['sn_global_idx'])

    subpop_index = _draw_subpop(simulator, redshifts, host_properties,
                                batch_n_sne, keys_for('SUBPOP'))

    drawn = {}
    sample_param, wgt_population = _parameter_sampler(
        simulator, params, host_properties, keys_for, subpop_index, drawn,
        overrides=overrides, population_functions=population_functions)

    _phasecor_grid_arr = draw_phase_scatter(
        simulator.phasecor_cholesky_L,
        keys_for('PHASECOR'), simulator.model_settings.sigmaclip_magsmear)
    _magsmear_arr = draw_coherent_scatter(simulator.model_settings, keys_for('MAGSMEAR_COH'))
    _magsmear_filter_arr = draw_filter_scatter(
        simulator.model_settings, len(simulator.photometry.transmissions),
        keys_for('MAGSMEAR_FILTER'))
    _zsmear_arr = _draw_redshift_smear(simulator.rates, keys_for('ZSMEAR'))
    _gensmear_arr = draw_model_scatter(
        simulator.model_settings, simulator.gensmear, keys_for('GENSMEAR'))

    model_sample = sample_param
    if simulator.survey.use_simlib_salt2:
        model_sample = _simlib_header_override(
            simulator, sample_param, field_indices, keys_for)

    # the population draws in declaration order, which is the order a
    # VARIATION may read them in; then what the model makes of them
    sed_model = simulator.model
    declared = [name for name, _ in sed_model.configured_population]
    model_params = {
        name: model_sample(name) for name in
        sed_model.drawn_parameters(simulator.model_settings.populations)}
    try:
        model_params.update(sed_model.derived_parameters(
            {**params, **model_params}, keys_for,
            template_budget=template_budget))
    except KeyError as err:
        if err.args[0] in declared:
            raise _missing_population(err.args[0]) from None
        raise
    undrawn = [name for name in declared if name not in model_params]
    if undrawn:
        raise _missing_population(undrawn[0])

    _gensmear_arr = scale_model_scatter(simulator.model_settings, _gensmear_arr, model_params)

    model = {
        'wgt_population': wgt_population(),
        # the drawn class, carried as truth; -1 when none are declared
        'subpop': (subpop_index if subpop_index is not None
                   else jnp.full(batch_n_sne, -1, dtype=jnp.int32)),
        # coherent smear and MAGOFF (snlc_sim.c:8483-8491)
        'magsmear': jnp.zeros(batch_n_sne),
        'magoff': jnp.zeros(batch_n_sne),
        'template_idx': jnp.zeros(batch_n_sne, dtype=jnp.int32),
    }
    if _phasecor_grid_arr is not None:
        model['_phasecor_grid'] = _phasecor_grid_arr
    if _gensmear_arr is not None:
        model['_gensmear_randoms'] = _gensmear_arr
    if _zsmear_arr is not None:
        model['_zsmear_normal'] = _zsmear_arr
    model.update(model_params)
    if _magsmear_arr is not None:
        # compose with any model-level magsmear (NON1ASED per-template)
        model['magsmear'] = model['magsmear'] + _magsmear_arr
    if _magsmear_filter_arr is not None:
        model['_magsmear_filter_ran'] = _magsmear_filter_arr
    return model


def draw_dust(simulator, params,
              host_properties: Optional[Dict[str, jnp.ndarray]], *,
              key, overrides=None, population_functions=()) -> Dict[str, jnp.ndarray]:
    """Host-galaxy AV and RV (gen_modelPar_dust, snlc_sim.c:12912), at the
    event's final redshift. SNANA draws them after the host; a weight map
    that reads them needs them before it. Returns the dust columns and the
    population weight with the dust's reweighting folded in."""
    sample_param, dust_weight = _parameter_sampler(
        simulator, params, host_properties,
        _keys_for(key, params['sn_global_idx']),
        _subpop_index(simulator, params), _population_values(simulator, params),
        overrides=overrides, population_functions=population_functions)
    av, rv, sim_av, sim_rv = _draw_host_dust(
        simulator, params['sim_redshift_cmb'].shape[0], sample_param,
        event_keys(key, 'AV', params['sn_global_idx']))
    return {'av': av, 'rv': rv, 'sim_av': sim_av, 'sim_rv': sim_rv,
            'wgt_population': params['wgt_population'] * dust_weight()}


def draw_grey_offset(simulator, params,
                     host_properties: Optional[Dict[str, jnp.ndarray]], *,
                     overrides=None, population_functions=()):
    """GENMAG_OFF_GLOBAL per SN, with its redshift variation at the final
    zCMB, once the host's overrides are in (snlc_sim.c:12940)."""
    for name, kind, function, deps in population_functions:
        if name == 'GENMAG_OFF_GLOBAL':
            given = (overrides or {}).get(name, {})
            if not isinstance(given, Mapping):
                return given
            inputs = _population_dependencies(deps, params, host_properties,
                                              _population_values(simulator, params))
            # A dummy event axis also broadcasts relations without dependencies.
            return jax.vmap(lambda _, *args: function(*args, **given))(
                params['sim_redshift_cmb'], *inputs)
    return grey_offset(simulator, _subpop_index(simulator, params),
                       params['sim_redshift_cmb'], host_properties,
                       _population_values(simulator, params),
                       params['sim_redshift_cmb'].shape[0])
