"""Photometric noise: the noise-equivalent area, the zero-point variance
both programs add, the fit's flux error recomputed from the noise model
(FLUXERRCALC), the simulator's photo-electron variances
(gen_fluxNoise_calc), the FUDGE_FLUXCAL systematics, and the FLUXERRMODEL
maps: their reader, SNANA's choice of map and REDCOV entry, and their
evaluation.
"""

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from jax_snlc_core import logging as log
from jax_snlc_core.interpolation import interpolate_corners
from jax_snlc_core.logging import SNANAError
from jax_snlc_core.snana_files import (exact_values, gridmap, gridmap_segments, leading_values,
                                       resolve_fits_path, split_names)

logger = logging.getLogger(__name__)


def noise_equivalent_aperture(psf1, psf2, psf_ratio):
    """SNANA's NoiseEquivAperture (sntools.c:5342-5416): effective sky aperture
    in pixel^2 from the (possibly double-) Gaussian PSF parameters. The only
    branch is SNANA's own single-Gaussian check; lanes taking it use dummy
    denominators so the unselected double-Gaussian expression stays finite."""
    sqsig1 = psf1 * psf1
    sqsig2 = psf2 * psf2
    sqsum = sqsig1 + sqsig2
    single_gauss = (psf_ratio < 1.0e-5) | (psf2 < 0.0001)
    nea_single = 4.0 * np.pi * sqsig1

    safe_sqsig1 = np.where(single_gauss, 1.0, sqsig1)
    safe_psf1 = np.where(single_gauss, 1.0, psf1)
    safe_psf2 = np.where(single_gauss, 1.0, psf2)
    tmp = psf_ratio * sqsig2 / safe_sqsig1
    a1 = 1.0 / (1.0 + tmp)
    a2 = 1.0 - a1
    tmp2 = a1 * safe_psf2 / safe_psf1 + a2 * safe_psf1 / safe_psf2
    nea_double = 4.0 * np.pi * sqsum / (1.0 + tmp2 * tmp2)

    return np.where(single_gauss, nea_single, nea_double)


def zero_point_variance(flux, zpterr):
    """The variance a zero-point error ``zpterr`` (mag) puts on ``flux``,
    ``(flux (10^(0.4 zpterr) - 1))^2``, in the units of ``flux`` squared; the
    same term in SNANA's fit (FLUXERRCALC, snana.F90) and simulation
    (gen_fluxNoise_calc, snlc_sim.c:27900-27907). Written with arithmetic
    operators, so it takes numpy and JAX arrays alike."""
    err = flux * (10.0 ** (0.4 * zpterr) - 1.0)
    return err * err


def fluxcal_errcalc(fluxcal, fluxcal_err, zeropt, zpterr, skysig, skysig_t,
                    psf1, psf2, psf_ratio, gain, zp_fluxcal, sb_fluxcal=0.0,
                    pixsize=0.0):
    """SNANA's ERRCALC: the flux error recomputed from the noise model
    (sky + signal-Poisson + zeropoint jitter + host SB Poisson), in FLUXCAL
    units. Backs ERRTEST = ERRCALC/ERRTRUE (snana.F90:20670). Read noise is
    omitted: SNANA never populates SNLC_RDNOISE at fit time (only
    initialised to 0, snana.F90:15279), so RDNOISE_ADU=0 in its
    FLUXERRCALC. sb_fluxcal is the host surface brightness (FLUXCAL/arcsec^2,
    HOSTGAL_SB_FLUXCAL of the epoch's band); the host term (snana.F90:20640)
    needs it and pixsize."""
    zpscale = np.power(10.0, 0.4 * (zeropt - zp_fluxcal))  # FLUXADU/FLUXCAL
    fluxadu = zpscale * fluxcal
    fluxadu_err = zpscale * fluxcal_err

    tmprat = np.where(fluxadu_err != 0.0, fluxadu / np.where(fluxadu_err != 0.0,
                      fluxadu_err, 1.0), 0.0)
    sqerr_signal = np.where((tmprat > 0.5) & (gain > 0.0),
                            fluxadu / np.where(gain > 0.0, gain, 1.0), 0.0)

    area = noise_equivalent_aperture(psf1, psf2, psf_ratio)
    sky_sq = skysig ** 2 + np.maximum(0.0, skysig_t) ** 2
    sqerr_sky = area * sky_sq

    # host SB Poisson term, SNANA's AREA*SQ1*SBFLUXADU/SKY_AVG_ADU with
    # SKY_AVG_ADU = GAIN*SQ1, so it needs a sky noise as well as a gain
    # (snana.F90:20568-20569, :20641-20644)
    sbfluxadu = np.asarray(sb_fluxcal) * np.asarray(pixsize) ** 2 * zpscale
    sqerr_host = np.where((np.asarray(sb_fluxcal) > 0.001) & (gain > 0.0) & (sky_sq > 0.0),
                          area * sbfluxadu / np.where(gain > 0.0, gain, 1.0),
                          0.0)

    sqerr_zp = zero_point_variance(fluxadu, zpterr)

    return np.sqrt(sqerr_sky + sqerr_host + sqerr_signal + sqerr_zp) / zpscale


def noise_components_pe(zptavgs, gains, fluxcals, neas, skysigs, readnoises, zp_fluxcal):
    """Per-obs source flux, sky variance, and CCD-read variance in
    photoelectrons (gen_fluxNoise_calc, snlc_sim.c:27825-27857). Returns
    ``(flux_pe, sqerr_sky, sqerr_ccd, npe_over_fluxcal)``. The SIMLIB NEA is
    used directly, as SNANA does; it is validated positive at construction."""
    npe_over_fluxcal = jnp.power(10.0, 0.4 * (zptavgs - zp_fluxcal)) * gains
    flux_pe = fluxcals * npe_over_fluxcal
    sky_e = skysigs * gains
    sqerr_sky = neas * sky_e * sky_e
    sqerr_ccd = neas * jnp.square(readnoises)
    return flux_pe, sqerr_sky, sqerr_ccd, npe_over_fluxcal


def template_noise_variance_pe(neas, gains, zptavgs, template_skysigs,
                               template_readnoises, template_zpts):
    """Template sky+read variance in photoelectrons, scaled to the search
    zero-point (snlc_sim.c:27861-27885). SNANA's activation rule is
    ``template_skysig > 0`` with TEMPLATE_ZPT validated at construction
    (>= 10, snlc_sim.c:27866-27871)."""
    has_template = template_skysigs > 0.0
    template_sky_e = template_skysigs * gains
    template_var = (
        neas * template_sky_e * template_sky_e
        + neas * jnp.square(template_readnoises)
    )
    zpt_scale = jnp.power(10.0, 0.8 * (zptavgs - template_zpts))
    return jnp.where(has_template, template_var * zpt_scale, 0.0)


def noise_variances(fluxcal, zptavg, zpterr, gain, nea, skysig, readnoise,
                    template_skysig, template_readnoise, template_zpt,
                    fluxgal_pe, template_fluxcal, *, smearflag_zeropt, zp_fluxcal):
    """Each epoch's pre-fudge variances in photo-electrons and the S/N they
    give, elementwise over any shape of epochs (gen_fluxNoise_calc).

    The source term is the true FLUXCAL in photo-electrons; the sky and
    readout terms scale with the NEA; the template term applies where a
    template sky noise is given. SMEARFLAG_ZEROPT decides where the
    zero-point jitter goes: bit 1 adds it to the drawn error (``sqsig_SZ``),
    bit 2 to the reported one (``sqsig_calc_data``), and it scales the
    difference flux: ``fluxcal`` less ``template_fluxcal``, the SN's own
    FLUXCAL in the template image (zero without one). ``fluxgal_pe`` is the
    host's photo-electrons, zero unless SMEARFLAG_HOSTGAL adds them.

    Returns ``(flux_pe, npe_over_fluxcal, sqsig_S, sqsig_SZ, sqsig_T,
    sqsig_calc_data, snr_calc_szt, snr_calc_st)``: the source photo-electrons
    and photo-electrons per FLUXCAL; the source, host, sky and readout
    variance, that plus the drawn zero-point term, and the template variance;
    SQSIG_CALC_DATA; and the S/N of the reported error and of the
    source-plus-template error (snlc_sim.c:27911-27953).
    """
    flux_pe, sqerr_sky, sqerr_ccd, npe_over_fluxcal = noise_components_pe(
        zptavg, gain, fluxcal, nea, skysig, readnoise, zp_fluxcal=zp_fluxcal)
    template_var = template_noise_variance_pe(
        nea, gain, zptavg, template_skysig, template_readnoise, template_zpt)

    # Zero-point jitter on the difference flux, the SN less its own light in
    # the template (snlc_sim.c:27900-27907); SMEARFLAG_ZEROPT bit 1 puts it in
    # the TRUE (drawn) error, bit 2 in the DATA (reported) error (:27929-27939).
    sqerr_zp = zero_point_variance(flux_pe - template_fluxcal * npe_over_fluxcal, zpterr)
    sqerr_zp_true = jnp.where((smearflag_zeropt & 1) > 0, sqerr_zp, 0.0)
    sqerr_zp_data = jnp.where((smearflag_zeropt & 2) > 0, sqerr_zp, 0.0)

    # fluxgal_pe is the host-galaxy shot term (SMEARFLAG_HOSTGAL bit 1)
    sqsig_S = flux_pe + fluxgal_pe + sqerr_sky + sqerr_ccd
    sqsig_calc_data = sqsig_S + sqerr_zp_data + template_var
    return (flux_pe, npe_over_fluxcal, sqsig_S, sqsig_S + sqerr_zp_true,
            template_var, sqsig_calc_data, flux_pe / jnp.sqrt(sqsig_calc_data),
            flux_pe / jnp.sqrt(sqsig_S + template_var))


def apply_fluxcal_fudges(fluxcal, fluxcal_err, offset, err_scale, err_add,
                         err_pix, mag_err, nea):
    """SNANA EXEC_FUDGE_FLUXCAL (snana.F90:18792-19040): user systematic
    fudges on calibrated fluxes. Returns (flux, err1, add_var): the offset flux,
    the scaled-original error `err1 = err*err_scale`, and the additive fudge
    variance: additive (`err_add`, signed so a negative value subtracts in
    quadrature), per-pixel (`err_pix`, weighted by the noise-equivalent aperture
    `nea`, and contributing nothing unless it is positive), and fractional-mag
    (``mag_err * flux``). The caller sums err1^2 + add_var
    (plus any FLUXERRMODEL/HOSTNOISE contribution, also from err1) and sqrts."""
    flux = fluxcal + offset
    err1 = fluxcal_err * err_scale
    # negative ERRPIX adds nothing (snana.F90:18865)
    area = np.where(err_pix > 0.0, nea, 0.0)
    add_var = (np.sign(err_add) * (err_add * err_add)
               + (err_pix * err_pix) * area + (mag_err * flux) ** 2)
    return flux, err1, add_var


# Variable identifiers, matching SNANA's IPAR_FLUXERRMAP_*
SUPPORTED_VARNAMES: Tuple[str, ...] = (
    'MJD',
    'PSF',     # FWHM, arcsec
    'SKYSIG',  # ADU/pixel
    'ZP',      # ADU
    'LOGSNR',
    'SBMAG',   # mag/arcsec^2
    'GALMAG',  # galmag at NEA aperture, mag
    'SNSEP',   # SN-host separation, arcsec
)
VALUE_VARNAMES: Tuple[str, ...] = ('ERRSCALE', 'ERRADD')

# every band or field, and the default BAND:, FIELD: and REDCOV group
ALL = 'ALL'

# bit 1 SIM, bit 2 DATA; both by default (sntools_fluxErrModels.c:168)
MASK_APPLY_TRUE = 1
MASK_APPLY_DATA = 2
MASK_APPLY_DEFAULT = MASK_APPLY_TRUE | MASK_APPLY_DATA


@dataclass
class FluxErrMap:
    """One FLUXERRMODEL map (one MAPNAME/BAND/FIELD/VARNAMES block).

    The LAST varname is always 'ERRSCALE' or 'ERRADD' (the value column);
    the dimension varnames are `varnames[:-1]`.
    """
    name: str
    band: str
    field_group: str                       # DEFINE_FIELDGROUP name, or ALL
    varnames: List[str]                    # including the value column
    grid_axes: List[np.ndarray]            # 1-D axis per dim varname (NDIM = NVAR-1)
    grid_values: np.ndarray
    is_errscale: bool
    mask_apply: int = MASK_APPLY_DEFAULT
    scale_true: float = 1.0
    scale_data: float = 1.0

    @property
    def dim_varnames(self) -> List[str]:
        return self.varnames[:-1]


@dataclass
class RedCovEntry:
    """One REDCOV(...): bands_str entry."""
    field_group: str          # ALL for global, else a DEFINE_FIELDGROUP name
    bands: str
    redcov: float             # in [0, 1)


@dataclass
class FluxErrModelData:
    """Parsed contents of a FLUXERRMODEL_FILE."""
    maps: List[FluxErrMap] = field(default_factory=list)
    field_groups: Dict[str, List[str]] = field(default_factory=dict)
    redcov: List[RedCovEntry] = field(default_factory=list)


def _field_list(field_group: str, field_groups: Dict[str, List[str]]) -> str:
    """The FIELDLIST string a map or REDCOV entry matches against: a FIELD
    naming a DEFINE_FIELDGROUP expands to the group's field list, any other
    stands as a literal SIMLIB field list (set_FIELDLIST_FLUXERRMODEL,
    sntools_fluxErrModels.c:846-874)."""
    if field_group == ALL:
        return ALL
    if field_group in field_groups:
        return '+'.join(field_groups[field_group])
    return field_group


def fluxerr_maps_serving(maps: List[FluxErrMap], field_groups: Dict[str, List[str]],
                         band_char: str, field_name: str) -> List[int]:
    """Map indices matching SNANA's BANDLIST and expanded FIELDLIST rules
    (INDEX_MAP_FLUXERRMODEL, sntools_fluxErrModels.c:877-923).

    The caller rejects multiple matches when an observation uses the pair.
    """
    matches = []
    for i, m in enumerate(maps):
        fieldlist = _field_list(m.field_group, field_groups)
        if ((m.band == ALL or band_char in m.band)
                and (fieldlist == ALL or field_name in fieldlist)):
            matches.append(i)
    return matches


def redcov_entries_serving(redcov: List[RedCovEntry], field_groups: Dict[str, List[str]],
                           band_char: str, field_name: str) -> List[int]:
    """Indices of the REDCOV entries that match (band, field).

    SNANA's INDEX_REDCOV_FLUXERRMODEL with OPT_FIELD=2, as the simulation
    calls it (sntools_fluxErrModels.c:928-968, snlc_sim.c:28148): the band
    a substring of the entry's band list, and the field a substring of its
    group-expanded field list or the entry for ALL fields. SNANA aborts when
    an observation finds more than one, so the caller does too.
    """
    return [k for k, entry in enumerate(redcov)
            if band_char in entry.bands
            and (entry.field_group == ALL
                 or field_name in _field_list(entry.field_group, field_groups))]


def check_redcov_entries(redcov: List[RedCovEntry]) -> None:
    """Abort on a REDCOV entry that defines a band its field group already
    has, as SNANA does while loading the entries (load_REDCOV_FLUXERRMODEL,
    sntools_fluxErrModels.c:556-566): each band of an entry is matched
    against the entries before it by INDEX_REDCOV_FLUXERRMODEL with
    OPT_FIELD=1, so an earlier entry for ALL fields matches it, as does one
    whose group name contains the entry's group name.

    Only the simulation reads REDCOV, so it checks the entries it uses: the
    file's, or those of a FLUXERRMODEL_REDCOV override that replaces them.
    An overlap only in the groups' field lists passes here; the simulation
    aborts on it if an epoch is observed in a shared field and band.
    """
    for k, entry in enumerate(redcov):
        for band in entry.bands:
            if any(band in earlier.bands
                   and (earlier.field_group == ALL
                        or entry.field_group in earlier.field_group)
                   for earlier in redcov[:k]):
                raise SNANAError(
                    f"Cannot define {band}-{entry.field_group} band more than once; "
                    f"each band and field group can be defined only once in REDCOV keys.",
                    function_name="check_redcov_entries")


def _build_fluxerr_map(name: str, band: str, field_group: str,
                       varnames: List[str], rows: List[List[float]],
                       scale_true: float, scale_data: float) -> FluxErrMap:
    """Build a FluxErrMap from parsed VARNAMES + ROW data."""
    if varnames is None:
        raise SNANAError(f"FLUXERRMODEL map {name!r} has no VARNAMES",
                         function_name="read_fluxerr_model_file")
    grid_axes, values = gridmap(rows, varnames[:-1], f"FLUXERRMODEL map {name!r}")
    return FluxErrMap(
        name=name,
        band=band,
        field_group=field_group,
        varnames=varnames,
        grid_axes=grid_axes,
        grid_values=values[..., 0].astype(np.float32),
        is_errscale=varnames[-1] == 'ERRSCALE',
        mask_apply=MASK_APPLY_DEFAULT,
        scale_true=scale_true,
        scale_data=scale_data,
    )


def parse_redcov(key: str, value: str) -> List[RedCovEntry]:
    """The REDCOV entries of one key and its value, as
    parse_REDCOV_FLUXERRMODEL reads them (sntools_fluxErrModels.c:402-482).

    The key, such as ``REDCOV``, ``REDCOV(DEEP)`` or ``FLUXERRMODEL_REDCOV``,
    names an optional field group in parentheses; the value is a
    comma-separated list of ``bands:correlation``, such as ``g:0.6,ri:0.4``.
    """
    field_group = key[key.index('(') + 1:key.rindex(')')] if '(' in key else ALL
    entries = []
    for item in value.split(','):
        bands, colon, correlation = item.partition(':')
        if not bands or not colon:
            raise SNANAError(f"REDCOV item {item!r} is not bands:correlation",
                             function_name="parse_redcov")
        rho = float(correlation)
        if not 0.0 <= rho < 1.0:
            raise SNANAError(
                f"REDCOV value must be in [0, 1), got {rho} for bands {bands!r}",
                function_name="parse_redcov")
        entries.append(RedCovEntry(field_group=field_group, bands=bands, redcov=rho))
    return entries


# keys that belong to the map MAPNAME: opens and ENDMAP: closes
_FLUXERRMODEL_MAP_KEYS = ('BAND', 'FILTER', 'FIELD', 'SCALE_FLUXERR_TRUE',
                          'SCALE_FLUXERR_DATA', 'VARNAMES', 'ROW')


def read_fluxerr_model_file(filepath: str) -> FluxErrModelData:
    """Parse a FLUXERRMODEL_FILE into its maps, field groups and REDCOV rows
    (INIT_FLUXERRMODEL, sntools_fluxErrModels.c:130-246).

    Each map is keyed by some of MJD, PSF, SKYSIG, ZP, LOGSNR, SBMAG, GALMAG
    and SNSEP and supplies an ERRSCALE (multiplicative) or ERRADD (added in
    quadrature) correction to the flux error; maps may differ in variables
    and grid. A REDCOV key, spelt with REDCOV or REDCOR, gives the
    between-band correlation, within a field group, of the variance the maps
    add. Environment variables in ``filepath`` are expanded. The format,
    after any DOCUMENTATION block::

        OPT_EXTRAP: 1                  # inert: SNANA always clips
                                       # (sntools_fluxErrModels.c:114)
        DEFINE_FIELDGROUP: SHALLOW  S1+S2+C1+C2
        DEFINE_FIELDGROUP: DEEP     X3+C3
        REDCOV(SHALLOW):  g:0.6,r:0.6,i:0.6,z:0.6

        MAPNAME: FLUXERR_SCALE
        BAND:    g
        FIELD:   SHALLOW
        SCALE_FLUXERR_TRUE: 1.0
        SCALE_FLUXERR_DATA: 1.0
        VARNAMES: SBMAG ERRSCALE
        ROW: 20.5  2.967  # comment
        ROW: 21.5  2.967
        ENDMAP:
    """
    path = resolve_fits_path(Path(os.path.expandvars(filepath)), required=True)
    data = FluxErrModelData()
    block = None   # the open map's settings, until its rows end

    # INIT_FLUXERRMODEL, sntools_fluxErrModels.c:130-243
    for segment in gridmap_segments(path, 'ROW:'):
        for key, values in segment:
            if block is not None and key in ('ENDMAP', 'MAPNAME'):
                data.maps.append(_build_fluxerr_map(**block))
                block = None
            if key == 'ENDMAP':
                continue
            if block is not None and block['rows'] and key != 'ROW':
                log.warn(logger, "FLUXERRMODEL", f"{path}: {key}: among map "
                         f"{block['name']!r}'s rows is not read")
                continue
            if key in _FLUXERRMODEL_MAP_KEYS and block is None:
                raise SNANAError(f"{path}: {key}: comes outside a MAPNAME: ... ENDMAP: block",
                                 function_name="read_fluxerr_model_file")
            if key == 'OPT_EXTRAP':
                # inert: SNANA always holds a query to the grid edges
                leading_values(path, key, values, 1)
            elif key == 'NVAR':
                # obsolete: SNANA counts VARNAMES
                leading_values(path, key, values, 1)
            elif key == 'DEFINE_FIELDGROUP':
                name, fields = leading_values(path, key, values, 2)
                data.field_groups[name] = [f for f in re.split('[+,]', fields) if f]
            elif 'REDCOV' in key or 'REDCOR' in key:
                data.redcov.extend(parse_redcov(key, *leading_values(path, key, values, 1)))
            elif key == 'MAPNAME':
                name = leading_values(path, key, values, 1)[0]
                block = dict(name=name, band=ALL, field_group=ALL, varnames=None,
                             rows=[], scale_true=1.0, scale_data=1.0)
            elif key in ('BAND', 'FILTER'):
                block['band'] = leading_values(path, key, values, 1)[0]
            elif key == 'FIELD':
                block['field_group'] = leading_values(path, key, values, 1)[0]
            elif key == 'SCALE_FLUXERR_TRUE':
                block['scale_true'] = float(leading_values(path, key, values, 1)[0])
            elif key == 'SCALE_FLUXERR_DATA':
                block['scale_data'] = float(leading_values(path, key, values, 1)[0])
            elif key == 'VARNAMES':
                names = [name.upper() for name in split_names(' '.join(values))]
                if (len(names) < 2 or names[-1] not in VALUE_VARNAMES
                        or not set(names[:-1]) <= set(SUPPORTED_VARNAMES)):
                    raise SNANAError(
                        f"{path}: map {block['name']!r} VARNAMES {names} must be some of "
                        f"{SUPPORTED_VARNAMES} followed by one of {VALUE_VARNAMES}",
                        function_name="read_fluxerr_model_file")
                block['varnames'] = names
            elif key == 'ROW':
                if block['varnames'] is None:
                    raise SNANAError(f"{path}: map {block['name']!r} has a ROW before VARNAMES",
                                     function_name="read_fluxerr_model_file")
                block['rows'].append([float(v) for v in exact_values(
                    path, key, values, len(block['varnames']))])
            else:
                log.warn(logger, "FLUXERRMODEL", f"{path}: {key}: is not a key here and is "
                         f"not read")
        if block is not None and block['rows']:
            data.maps.append(_build_fluxerr_map(**block))
            block = None
    if block is not None:
        data.maps.append(_build_fluxerr_map(**block))
    return data


def fluxerr_map_index(data: FluxErrModelData, band_char: str, field_name: str) -> int:
    """The row of :func:`stacked_fluxerr_maps` serving (band, field): the map
    SNANA's INDEX_MAP_FLUXERRMODEL chooses, or the last, inert row where no
    map serves it."""
    matches = fluxerr_maps_serving(data.maps, data.field_groups, band_char, field_name)
    if len(matches) > 1:
        raise SNANAError(
            f"Invalid NMATCH={len(matches)} FLUXERRMODEL maps for BAND={band_char!r} "
            f"and FIELD={field_name!r}.", function_name='fluxerr_map_index')
    return matches[0] if matches else len(data.maps)


def fluxerr_variables(data: FluxErrModelData) -> Tuple[str, ...]:
    """Every variable any map uses, in the order the maps first name them."""
    return tuple(dict.fromkeys(v for m in data.maps for v in m.dim_varnames))


def stacked_fluxerr_maps(data: FluxErrModelData) -> Dict[str, np.ndarray]:
    """The maps over :func:`fluxerr_variables`, as arrays with one row per map
    and a last, inert row (an ERRSCALE of 1).

    ``grids`` is (n_maps + 1, *shape). A map is constant along a variable it
    does not use, an axis of one node; each map's axes keep their own
    ``origins``, ``steps`` and ``sizes`` (n_maps + 1, n_variables), and are
    padded to the longest. Each row also carries its SCALE_FLUXERR_TRUE and
    DATA, whether it applies to the TRUE and the DATA error, and whether it
    is an ERRSCALE map.
    """
    variables = fluxerr_variables(data)
    n_maps = len(data.maps)
    sizes = np.ones((n_maps + 1, len(variables)), dtype=np.int32)
    origins = np.zeros((n_maps + 1, len(variables)))
    steps = np.ones((n_maps + 1, len(variables)))
    for i, m in enumerate(data.maps):
        for name, axis in zip(m.dim_varnames, m.grid_axes):
            d = variables.index(name)
            sizes[i, d] = len(axis)
            origins[i, d] = axis[0]
            if len(axis) > 1:
                steps[i, d] = axis[1] - axis[0]
    shape = sizes.max(axis=0)
    grids = np.ones((n_maps + 1, *shape), dtype=np.float32)
    for i, m in enumerate(data.maps):
        order = np.argsort([variables.index(name) for name in m.dim_varnames])
        values = np.transpose(m.grid_values, order).reshape(sizes[i])
        grids[i] = np.pad(values, [(0, s - n) for s, n in zip(shape, sizes[i])], mode='edge')
    return {
        'grids': grids,
        'origins': origins,
        'steps': steps,
        'sizes': sizes,
        'is_errscale': np.array([m.is_errscale for m in data.maps] + [True]),
        'scale_true': np.array([m.scale_true for m in data.maps] + [1.0]),
        'scale_data': np.array([m.scale_data for m in data.maps] + [1.0]),
        'applies_true': np.array([bool(m.mask_apply & MASK_APPLY_TRUE) for m in data.maps]
                                 + [False]),
        'applies_data': np.array([bool(m.mask_apply & MASK_APPLY_DATA) for m in data.maps]
                                 + [False]),
    }


def interpolate_fluxerr_maps(maps, map_index, query):
    """Each query's value on its map, by multilinear interpolation on that
    map's uniform grid, the query held to the grid's edges as SNANA always
    holds it (interp_GRIDMAP).

    ``maps`` is :func:`stacked_fluxerr_maps`, ``map_index`` the map row of
    each query, any shape, and ``query`` that shape plus a last axis of the
    values of :func:`fluxerr_variables`.
    """
    lower, frac = [], []
    for d in range(query.shape[-1]):
        origin, step, top = (jnp.asarray(maps[name])[map_index, d].astype(query.dtype)
                             for name in ('origins', 'steps', 'sizes'))
        x = jnp.clip((query[..., d] - origin) / step, 0.0, top - 1)
        i = jnp.floor(x).astype(jnp.int32)
        lower.append(i)
        frac.append(x - i.astype(x.dtype))
    return interpolate_corners(maps['grids'], lower, frac, map_index)


def apply_fluxerr_model(err, value, is_errscale):
    """The flux error a map value gives (apply_FLUXERRMODEL,
    sntools_fluxErrModels.c:972-1002): an ERRSCALE map multiplies the error,
    an ERRADD map adds its value in quadrature, in FLUXCAL units."""
    return jnp.where(is_errscale, err * value, jnp.sqrt(err * err + value * value))


def fluxerr_model_sqscales(maps, rows, err_model_val, npe_over_fluxcal, sqsig_calc_data):
    """The simulation's FLUXERRMODEL variance scales (SQSCALE) for the TRUE
    and DATA errors (snlc_sim.c:28110-28135): get_FLUXERRMODEL's TRUE and
    DATA errors over the input error, SIG_CALC/Npe_over_FLUXCAL before any
    fudge, from each epoch's row of :func:`stacked_fluxerr_maps` and its
    value there. A side the map does not apply to keeps the input error, and
    a TRUE scale of exactly 1 becomes 1.001.
    """
    err_in = jnp.sqrt(sqsig_calc_data) / npe_over_fluxcal
    modelled = apply_fluxerr_model(err_in, err_model_val, maps['is_errscale'][rows])
    err_true = jnp.where(maps['applies_true'][rows], modelled * maps['scale_true'][rows], err_in)
    err_data = jnp.where(maps['applies_data'][rows], modelled * maps['scale_data'][rows], err_in)
    ratio_true = err_true / err_in
    ratio_true = jnp.where(ratio_true == 1.0, 1.001, ratio_true)
    ratio_data = err_data / err_in
    return ratio_true * ratio_true, ratio_data * ratio_data


def evaluate_fluxerrmodel_data(model, band_chars, fields, err, conditions):
    """The fit's FLUXERRMODEL (get_FLUXERRMODEL, DATA error): each epoch's
    error from the map SNANA's INDEX_MAP_FLUXERRMODEL chooses for its band
    and field, then SCALE_FLUXERR_DATA. ``conditions`` maps each of the
    maps' VARNAMES to a per-epoch array; an epoch no map serves keeps its
    error, and a file without maps changes no error. The evaluation is in
    double precision whatever frame the run is in, as the fit's other error
    terms are.
    """
    err = np.asarray(err, dtype=np.float64)
    if not model.maps:
        return err
    maps = stacked_fluxerr_maps(model)
    pairs = sorted(set(zip(np.asarray(band_chars).tolist(), np.asarray(fields).tolist())))
    row_of = {pair: fluxerr_map_index(model, *pair) for pair in pairs}
    rows = np.array([row_of[pair] for pair in zip(np.asarray(band_chars).tolist(),
                                                  np.asarray(fields).tolist())], dtype=np.int32)
    with jax.enable_x64(True):
        query = jnp.stack([jnp.asarray(conditions[v], dtype=jnp.float64)
                           for v in fluxerr_variables(model)], axis=-1)
        value = interpolate_fluxerr_maps(maps, rows, query)
        modelled = np.asarray(apply_fluxerr_model(err, value, maps['is_errscale'][rows]))
    return np.where(maps['applies_data'][rows], modelled * maps['scale_data'][rows], err)
