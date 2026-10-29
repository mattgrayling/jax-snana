"""Photometric noise: the noise-equivalent area, the zero-point variance
both programs add, the fit's flux error recomputed from the noise model
(FLUXERRCALC), the simulator's photo-electron variances
(gen_fluxNoise_calc), the FUDGE_FLUXCAL systematics, and the FLUXERRMODEL
maps: their reader, SNANA's choice of map and REDCOV entry, and their
evaluation.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from jax_snlc_core.interpolation import interpolate_corners
from jax_snlc_core.logging import SNANAError
from jax_snlc_core.snana_files import open_text, resolve_fits_path, skip_docana


def noise_equivalent_aperture(psf1, psf2, psf_ratio):
    """SNANA's NoiseEquivAperture (sntools.c:5345-5419): effective sky aperture
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
    (gen_fluxNoise_calc, snlc_sim.c:27842-27849). Written with arithmetic
    operators, so it takes numpy and JAX arrays alike."""
    err = flux * (10.0 ** (0.4 * zpterr) - 1.0)
    return err * err


def fluxcal_errcalc(fluxcal, fluxcal_err, zeropt, zpterr, skysig, skysig_t,
                    psf1, psf2, psf_ratio, gain, zp_fluxcal, sb_fluxcal=0.0,
                    pixsize=0.0):
    """SNANA's ERRCALC: the flux error recomputed from the noise model
    (sky + signal-Poisson + zeropoint jitter + host SB Poisson), in FLUXCAL
    units. Backs ERRTEST = ERRCALC/ERRTRUE (snana.F90:20578). Read noise is
    omitted: SNANA never populates SNLC_RDNOISE at fit time (only
    initialised to 0, snana.F90:15190), so RDNOISE_ADU=0 in its
    FLUXERRCALC. sb_fluxcal is the host surface brightness (FLUXCAL/arcsec^2,
    HOSTGAL_SB_FLUXCAL of the epoch's band); the host term (snana.F90:20548)
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
    # (snana.F90:20476-20477, :20549-20552)
    sbfluxadu = np.asarray(sb_fluxcal) * np.asarray(pixsize) ** 2 * zpscale
    sqerr_host = np.where((np.asarray(sb_fluxcal) > 0.001) & (gain > 0.0) & (sky_sq > 0.0),
                          area * sbfluxadu / np.where(gain > 0.0, gain, 1.0),
                          0.0)

    sqerr_zp = zero_point_variance(fluxadu, zpterr)

    return np.sqrt(sqerr_sky + sqerr_host + sqerr_signal + sqerr_zp) / zpscale


def noise_components_pe(zptavgs, gains, fluxcals, neas, skysigs, readnoises, zp_fluxcal):
    """Per-obs source flux, sky variance, and CCD-read variance in
    photoelectrons (gen_fluxNoise_calc, snlc_sim.c:27767-27799). Returns
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
    zero-point (snlc_sim.c:27803-27827). SNANA's activation rule is
    ``template_skysig > 0`` with TEMPLATE_ZPT validated at construction
    (>= 10, snlc_sim.c:27808-27813)."""
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
    source-plus-template error (snlc_sim.c:27853-27895).
    """
    flux_pe, sqerr_sky, sqerr_ccd, npe_over_fluxcal = noise_components_pe(
        zptavg, gain, fluxcal, nea, skysig, readnoise, zp_fluxcal=zp_fluxcal)
    template_var = template_noise_variance_pe(
        nea, gain, zptavg, template_skysig, template_readnoise, template_zpt)

    # Zero-point jitter on the difference flux, the SN less its own light in
    # the template (snlc_sim.c:27842-27849); SMEARFLAG_ZEROPT bit 1 puts it in
    # the TRUE (drawn) error, bit 2 in the DATA (reported) error (:27871-27881).
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
    """SNANA EXEC_FUDGE_FLUXCAL (snana.F90:18700-18948): user systematic
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

# Default field-group name when a map's FIELD: is "ALL" or unset
ALL_GROUP = 'ALL'

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
    field_group: str                       # DEFINE_FIELDGROUP name, or ALL_GROUP
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
    field_group: str          # ALL_GROUP for global, else a DEFINE_FIELDGROUP name
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
    if field_group == ALL_GROUP:
        return ALL_GROUP
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
        if ((m.band == ALL_GROUP or band_char in m.band)
                and (fieldlist == ALL_GROUP or field_name in fieldlist)):
            matches.append(i)
    return matches


def redcov_entries_serving(redcov: List[RedCovEntry], field_groups: Dict[str, List[str]],
                           band_char: str, field_name: str) -> List[int]:
    """Indices of the REDCOV entries that match (band, field).

    SNANA's INDEX_REDCOV_FLUXERRMODEL with OPT_FIELD=2, as the simulation
    calls it (sntools_fluxErrModels.c:928-968, snlc_sim.c:28090): the band
    a substring of the entry's band list, and the field a substring of its
    group-expanded field list or the entry for ALL fields. SNANA aborts when
    an observation finds more than one, so the caller does too.
    """
    return [k for k, entry in enumerate(redcov)
            if band_char in entry.bands
            and (entry.field_group == ALL_GROUP
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
                   and (earlier.field_group == ALL_GROUP
                        or entry.field_group in earlier.field_group)
                   for earlier in redcov[:k]):
                raise SNANAError(
                    f"Cannot define {band}-{entry.field_group} band more than once; "
                    f"each band and field group can be defined only once in REDCOV keys.",
                    function_name="check_redcov_entries")


def _build_fluxerr_map(name: str, band: str, field_group: str,
                       varnames: List[str], rows: List[List[float]],
                       scale_true: float, scale_data: float) -> FluxErrMap:
    """Build a FluxErrMap from parsed VARNAMES + ROW data.

    SNANA enforces uniform binning per axis (`sntools_gridmap.c:31` aborts on
    non-uniform), and so does this. Vectorised, with no nested per-row
    loops.
    """
    n_dim = len(varnames) - 1
    is_errscale = varnames[-1].upper() == 'ERRSCALE'

    arr = np.asarray(rows, dtype=np.float64)
    dim_data = arr[:, :n_dim]      # (n_rows, n_dim)
    val_data = arr[:, n_dim]       # (n_rows,)

    grid_axes: List[np.ndarray] = [np.unique(dim_data[:, d]) for d in range(n_dim)]

    for d, ax in enumerate(grid_axes):
        if len(ax) < 2:
            continue
        diffs = np.diff(ax)
        if not np.allclose(diffs, diffs[0], rtol=1e-5, atol=1e-12):
            raise SNANAError(
                f"FLUXERRMODEL map {name!r}, dim {varnames[d]!r}: "
                f"non-uniform grid spacing not supported (diffs={diffs[:5]}...)",
                function_name="_build_fluxerr_map")

    grid_shape = tuple(len(a) for a in grid_axes)
    expected = int(np.prod(grid_shape))
    if expected != len(val_data):
        raise SNANAError(
            f"FLUXERRMODEL map {name!r}: expected {expected} rows "
            f"(from axes {[len(a) for a in grid_axes]}), got {len(val_data)}",
            function_name="_build_fluxerr_map")

    indices = tuple(np.searchsorted(axis, dim_data[:, d])
                    for d, axis in enumerate(grid_axes))
    positions = np.ravel_multi_index(indices, grid_shape)
    if len(np.unique(positions)) != expected:
        raise SNANAError(
            f"FLUXERRMODEL map {name!r}: each grid position must occur exactly once",
            function_name="_build_fluxerr_map")
    grid_values = np.empty(grid_shape, dtype=np.float32)
    grid_values[indices] = val_data

    return FluxErrMap(
        name=name,
        band=band,
        field_group=field_group,
        varnames=varnames,
        grid_axes=grid_axes,
        grid_values=grid_values,
        is_errscale=is_errscale,
        mask_apply=MASK_APPLY_DEFAULT,
        scale_true=scale_true,
        scale_data=scale_data,
    )


def parse_redcov_string(s: str) -> List[RedCovEntry]:
    """Parse a REDCOV string in the SNANA format.

    Forms:
        "REDCOV(GROUP): g:0.6,r:0.6"          (file-level keyword)
        "REDCOV: g:0.6"                        (file-level keyword, global group)
        "FLUXERRMODEL_REDCOV(GROUP) g:0.6,r:0.6"  (config override)
        "FLUXERRMODEL_REDCOV g:0.6"               (config override, global)
        "g:0.6,r:0.6"                          (raw bands string, global group)

    Returns one entry per band token.

    Multiple bands sharing one REDCOV value are returned as a single entry
    with `bands` as the concatenated string (e.g., "griz").
    """
    s = s.strip()

    field_group = ALL_GROUP

    upper = s.upper()
    if upper.startswith('FLUXERRMODEL_REDCOV') or upper.startswith('REDCOV'):
        i_paren = s.find('(')
        i_colon = s.find(':')
        if i_paren >= 0 and (i_colon < 0 or i_paren < i_colon):
            j = s.find(')', i_paren + 1)
            if j < 0:
                raise SNANAError(f"Unclosed REDCOV group in {s!r}",
                                 function_name="parse_redcov_string")
            field_group = s[i_paren + 1:j].strip()
            tail = s[j + 1:].strip()
            if tail.startswith(':'):
                tail = tail[1:].strip()
        else:
            tail = s.split(':', 1)[1].strip() if ':' in s else s.split(None, 1)[1]
        s = tail

    entries: List[RedCovEntry] = []
    for token in s.split(','):
        token = token.strip()
        if not token:
            continue
        if ':' not in token:
            raise SNANAError(f"REDCOV token missing colon: {token!r}",
                             function_name="parse_redcov_string")
        bands_str, val_str = token.split(':', 1)
        bands_str = bands_str.strip()
        val = float(val_str.strip())
        if not (0.0 <= val < 1.0):
            raise SNANAError(
                f"REDCOV value must be in [0, 1), got {val} for bands {bands_str!r}",
                function_name="parse_redcov_string")
        entries.append(RedCovEntry(field_group=field_group, bands=bands_str, redcov=val))

    return entries


def read_fluxerr_model_file(filepath: str) -> FluxErrModelData:
    """Parse a FLUXERRMODEL_FILE into its maps, field groups and REDCOV rows
    (sntools_fluxErrModels.c).

    Each map is keyed by some of MJD, PSF, SKYSIG, ZP, LOGSNR, SBMAG, GALMAG
    and SNSEP and supplies an ERRSCALE (multiplicative) or ERRADD (added in
    quadrature) correction to the flux error. A REDCOV row gives the
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
    expanded = Path(os.path.expandvars(filepath))
    with open_text(resolve_fits_path(expanded, required=True)) as f:
        lines = f.readlines()

    data = FluxErrModelData()

    # Per-map state, accumulated until ENDMAP:
    cur_name: Optional[str] = None
    cur_band: str = 'ALL'
    cur_field: str = ALL_GROUP
    cur_scale_true: float = 1.0
    cur_scale_data: float = 1.0
    cur_varnames: Optional[List[str]] = None
    cur_rows: List[List[float]] = []

    def _finalise_map():
        nonlocal cur_name, cur_band, cur_field, cur_scale_true, cur_scale_data
        nonlocal cur_varnames, cur_rows
        if cur_name is None:
            return
        if cur_varnames is None:
            raise SNANAError(
                f"FLUXERRMODEL map {cur_name!r} missing VARNAMES",
                function_name="_finalise_map")
        if not cur_rows:
            raise SNANAError(
                f"FLUXERRMODEL map {cur_name!r} has no ROW entries",
                function_name="_finalise_map")
        m = _build_fluxerr_map(cur_name, cur_band, cur_field, cur_varnames, cur_rows,
                               cur_scale_true, cur_scale_data)
        data.maps.append(m)
        cur_name = None
        cur_band = 'ALL'
        cur_field = ALL_GROUP
        cur_scale_true = 1.0
        cur_scale_data = 1.0
        cur_varnames = None
        cur_rows = []

    for raw in skip_docana(lines):
        stripped = raw.strip()
        if not stripped or stripped.startswith('#'):
            continue

        upper = stripped.upper()

        # inert: SNANA always clips (sntools_fluxErrModels.c:114)
        if upper.startswith('OPT_EXTRAP:'):
            continue

        if upper.startswith('DEFINE_FIELDGROUP:'):
            tail = stripped.split(':', 1)[1].strip().split()
            if len(tail) < 2:
                raise SNANAError(
                    f"DEFINE_FIELDGROUP needs name + field list: {stripped!r}",
                    function_name="read_fluxerr_model_file")
            group_name = tail[0]
            field_list_str = ' '.join(tail[1:])
            field_names = [f.strip() for f in field_list_str.replace(',', '+').split('+')
                           if f.strip()]
            data.field_groups[group_name] = field_names
            continue

        if upper.startswith('REDCOV(') or upper.startswith('REDCOV:'):
            data.redcov.extend(parse_redcov_string(stripped))
            continue

        if upper.startswith('MAPNAME:'):
            _finalise_map()
            cur_name = stripped.split(':', 1)[1].strip().split()[0]
            cur_band = 'ALL'
            cur_field = ALL_GROUP
            cur_scale_true = 1.0
            cur_scale_data = 1.0
            cur_varnames = None
            cur_rows = []
            continue

        # BAND:/FILTER: and FIELD: may share a line
        if (upper.startswith('BAND:') or upper.startswith('FILTER:')
                or upper.startswith('FIELD:')):
            tokens = stripped.split()
            i = 0
            while i < len(tokens):
                tok = tokens[i].upper().rstrip(':')
                if tok in ('BAND', 'FILTER') and i + 1 < len(tokens):
                    cur_band = tokens[i + 1]
                    i += 2
                elif tok == 'FIELD' and i + 1 < len(tokens):
                    cur_field = tokens[i + 1]
                    i += 2
                else:
                    i += 1
            continue

        if upper.startswith('SCALE_FLUXERR_TRUE:'):
            cur_scale_true = float(stripped.split(':', 1)[1].strip().split()[0])
            continue
        if upper.startswith('SCALE_FLUXERR_DATA:'):
            cur_scale_data = float(stripped.split(':', 1)[1].strip().split()[0])
            continue

        if upper.startswith('VARNAMES:'):
            parts = stripped.split(':', 1)[1].strip().split()
            if len(parts) < 2:
                raise SNANAError(
                    f"VARNAMES needs at least one dim + value column: {stripped!r}",
                    function_name="read_fluxerr_model_file")
            value_col = parts[-1].upper()
            if value_col not in VALUE_VARNAMES:
                raise SNANAError(
                    f"VARNAMES last column must be one of {VALUE_VARNAMES}, "
                    f"got {value_col!r} in map {cur_name!r}",
                    function_name="read_fluxerr_model_file")
            for v in parts[:-1]:
                vu = v.upper()
                if vu not in SUPPORTED_VARNAMES:
                    raise SNANAError(
                        f"FLUXERRMODEL: unsupported variable {v!r} in map "
                        f"{cur_name!r}. Supported: {SUPPORTED_VARNAMES}",
                        function_name="read_fluxerr_model_file")
            cur_varnames = [v.upper() for v in parts]
            continue

        # NVAR is informational; skip
        if upper.startswith('NVAR:'):
            continue

        if upper.startswith('ROW:'):
            row_str = stripped.split(':', 1)[1].strip()
            if '#' in row_str:
                row_str = row_str.split('#', 1)[0].strip()
            vals = [float(x) for x in row_str.split()]
            if cur_varnames is None:
                raise SNANAError(
                    f"ROW before VARNAMES in map {cur_name!r}",
                    function_name="read_fluxerr_model_file")
            if len(vals) != len(cur_varnames):
                raise SNANAError(
                    f"ROW has {len(vals)} columns but VARNAMES has "
                    f"{len(cur_varnames)} in map {cur_name!r}",
                    function_name="read_fluxerr_model_file")
            cur_rows.append(vals)
            continue

        if upper.startswith('ENDMAP'):
            _finalise_map()
            continue

    _finalise_map()

    # one VARNAMES and grid across maps (SNANA allows mixed; jax does not)
    if data.maps:
        varname_set = {tuple(m.varnames) for m in data.maps}
        if len(varname_set) != 1:
            raise SNANAError(
                f"FLUXERRMODEL_FILE {expanded!r}: maps use mixed VARNAMES, "
                f"which is not supported. Maps use: {sorted(varname_set)}",
                function_name="read_fluxerr_model_file")

        ax_signatures = {
            tuple(tuple(round(float(v), 9) for v in a) for a in m.grid_axes)
            for m in data.maps
        }
        if len(ax_signatures) != 1:
            raise SNANAError(
                f"FLUXERRMODEL_FILE {expanded!r}: maps use different grid "
                f"axes, which is not supported. All maps must share the same "
                f"grid for each dimension.", function_name="read_fluxerr_model_file")

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


def stacked_fluxerr_maps(data: FluxErrModelData):
    """The maps' values stacked as one (n_maps + 1, *grid) array, the last row
    inert (1 for ERRSCALE, 0 for ERRADD), with the shared axes' origins and
    steps and each row's SCALE_FLUXERR_TRUE/DATA and whether it applies to
    the TRUE and the DATA error. The reader has checked that the maps share
    VARNAMES and axes."""
    maps = data.maps
    axes = maps[0].grid_axes
    inert = 1.0 if maps[0].is_errscale else 0.0
    grids = np.concatenate([np.stack([m.grid_values for m in maps]),
                            np.full((1,) + maps[0].grid_values.shape, inert)]).astype(np.float32)
    return {
        'grids': grids,
        'origins': tuple(float(a[0]) for a in axes),
        'steps': tuple(float(a[1] - a[0]) if a.shape[0] > 1 else 1.0 for a in axes),
        'scale_true': np.array([m.scale_true for m in maps] + [1.0]),
        'scale_data': np.array([m.scale_data for m in maps] + [1.0]),
        'applies_true': np.array([bool(m.mask_apply & MASK_APPLY_TRUE) for m in maps] + [False]),
        'applies_data': np.array([bool(m.mask_apply & MASK_APPLY_DATA) for m in maps] + [False]),
    }


def interpolate_fluxerr_maps(grids, map_index, query, origins, steps):
    """Each query's value on its map, by multilinear interpolation on the
    maps' uniform grid, the query held to the grid's edges as SNANA always
    holds it (interp_GRIDMAP).

    ``grids`` is (n_maps, n_0, ..., n_d-1) from :func:`stacked_fluxerr_maps`,
    ``map_index`` the map row of each query, any shape, and ``query`` that
    shape plus a last axis of the d coordinates.
    """
    shape = grids.shape[1:]
    lower, frac = [], []
    for d, (origin, step, n) in enumerate(zip(origins, steps, shape)):
        x = jnp.clip((query[..., d] - origin) / step, 0.0, float(n - 1))
        i = jnp.floor(x).astype(jnp.int32)
        lower.append(i)
        frac.append(x - i.astype(x.dtype))
    return interpolate_corners(grids, lower, frac, map_index)


def apply_fluxerr_model(err, value, is_errscale):
    """The flux error a map value gives (apply_FLUXERRMODEL,
    sntools_fluxErrModels.c:972-1002): an ERRSCALE map multiplies the error,
    an ERRADD map adds its value in quadrature, in FLUXCAL units. Written
    with arithmetic operators, so it takes numpy and JAX arrays alike."""
    return err * value if is_errscale else (err * err + value * value) ** 0.5


def fluxerr_model_sqscales(err_model_val, scale_true, scale_data, applies_true,
                           applies_data, npe_over_fluxcal, sqsig_calc_data,
                           is_errscale):
    """The simulation's FLUXERRMODEL variance scales (SQSCALE) for the TRUE
    and DATA errors (snlc_sim.c:28044-28062): get_FLUXERRMODEL's TRUE and
    DATA errors over the input error, SIG_CALC/Npe_over_FLUXCAL before any
    fudge. A side the map does not apply to keeps the input error, and a
    TRUE scale of exactly 1 becomes 1.001.
    """
    err_in = jnp.sqrt(sqsig_calc_data) / npe_over_fluxcal
    modelled = apply_fluxerr_model(err_in, err_model_val, is_errscale)
    err_true = jnp.where(applies_true, modelled * scale_true, err_in)
    err_data = jnp.where(applies_data, modelled * scale_data, err_in)
    ratio_true = err_true / err_in
    ratio_true = jnp.where(ratio_true == 1.0, 1.001, ratio_true)
    ratio_data = err_data / err_in
    return ratio_true * ratio_true, ratio_data * ratio_data


def evaluate_fluxerrmodel_data(model, band_chars, fields, err, conditions):
    """The fit's FLUXERRMODEL (get_FLUXERRMODEL, DATA error): each epoch's
    error from the map SNANA's INDEX_MAP_FLUXERRMODEL chooses for its band
    and field, then SCALE_FLUXERR_DATA. ``conditions`` maps each of the
    maps' VARNAMES to a per-epoch array; an epoch no map serves keeps its
    error, and a file without maps changes no error. The interpolation is in
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
                           for v in model.maps[0].dim_varnames], axis=-1)
        value = np.asarray(interpolate_fluxerr_maps(jnp.asarray(maps['grids']), rows, query,
                                                    maps['origins'], maps['steps']))
    modelled = apply_fluxerr_model(err, value, model.maps[0].is_errscale)
    return np.where(maps['applies_data'][rows], modelled * maps['scale_data'][rows], err)
