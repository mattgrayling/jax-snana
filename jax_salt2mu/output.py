"""BBC output writers: .M0DIF, .FITRES, .COV, .YAML (SNANA outFile_driver)."""

from typing import List

import dataclasses

import jax.numpy as jnp
import numpy as np

from jax_snlc_core.utils.error_handling import SNANAError

from jax_snlc_core.io.snana_table import (program_comment, snana_version_comment,
                                          write_varnames_table)

from .results import BBCOutput, MUDIFERR_EMPTY
from .zbins import ZBinGrid
from .config import BLINDMASK_FIXPAR, BLINDMASK_MUz, BLIND_COSINE_M0
from .likelihood import make_loss
from jax_snlc_core.io.fits_path import open_text
from .tablevar import MISSING_TOKEN, output_varnames
from .distances import mu_at


# wide enough to reproduce any input column (SNANA copies them verbatim, :23348)
_PASSTHROUGH_FLOAT_FMT = "%.9g"

# the written MUPULL is capped here (SALT2mu.c:25296)
MUPULL_MAX = 99.999

# rows cat_only copies; any other leading key is skipped (SALT2mu.c:24436)
_ROW_KEYS = frozenset(("SN:", "ROW:", "GAL:"))


def _fmt_for(arr):
    arr = np.asarray(arr)
    if arr.dtype.kind in "iub":
        return "%d"
    if arr.dtype.kind in "fc":
        return _PASSTHROUGH_FLOAT_FMT
    return "%s"


def _param_label(name, config):
    """u13=2 renames the CC-prior scale in every output (SALT2mu.c:22112)."""
    if name == "scalePCC" and config._u_flags.get("u13", 0) == 2:
        return "scalePIa"
    return name


def _fit_summary(result, out: BBCOutput, zbins: ZBinGrid, config):
    """The ``# SALT2mu fit results:`` block (write_fitres_misc,
    SALT2mu.c:24935-24998): NSNFIT / contamination under a CC prior /
    -2lnL / chi2(Ia)/dof / sigint / each floated nuisance +- err /
    per-bin m0 +- err / M0avg."""
    lines = ["SALT2mu fit results: "]
    if out.biascor_summary:
        ndim, nsim, note, sigint_bcor = out.biascor_summary
        lines += [f" NSIM({ndim}D-BIASCOR)   = {nsim}   {note}",
                  f" sigint({ndim}D-BIASCOR) = {sigint_bcor:.3f} "]
    lines += [f" NSNFIT        = {result.nsnfit} "]
    if out.probcc is not None:
        with np.errstate(invalid="ignore", divide="ignore"):
            note = "photo-classified subset"
            lines += [f" SUM_PROBIa    = {out.sum_probia:9.2f}    # {note} ",
                      f" SUM_PROBCC    = {out.sum_probcc:9.2f}    # {note} ",
                      f" CONTAM_DATA   = {out.contam_data:.4f}    "
                      f"# SUM_PROB ratio for {note}"]
            if out.nsnfit_truecc is not None:
                contam_true = float(np.divide(
                    out.nsnfit_truecc, result.nsnfit - out.nsnspec_ia))
                lines += [f" CONTAM_TRUE   = {contam_true:.4f}    "
                          f"# true NCC/(NIa+NCC) for {note} (sim-data only)"]
    chi2_ia, ndof_ia = result.chi2sum_ia, result.ndof_ia
    lines += [f" -2log(L)     = {result.chi2:.2f} ",
              f" dchi2red/dsigint = {result.dchi2red_dsigint:.3f}",
              f" chi2(Ia)/dof = {chi2_ia:.2f}/{ndof_ia:.0f} = "
              f"{chi2_ia / ndof_ia:.3f}  "
              if ndof_ia > 0 else f" chi2(Ia)     = {chi2_ia:.2f} ",
              f" sigint   = {result.sigint:.4f}  ({result.niter} iterations)"]
    for name, err in result.nuis_err.items():
        lines.append(f" {_param_label(name, config):<14s} = "
                     f"{result.nuisance[name]:10.5f} +- {err:8.5f}   ")
    for iz in range(zbins.nbin):
        # an unfloated bin prints "too few events->IGNORED" in SNANA (:23208);
        # jax leaves the line out, so the test is the bin's error, not its count
        if out.m0diferr[iz] >= MUDIFERR_EMPTY:
            continue
        lines.append(
            f" m0_{iz:02d}-<M0avg>  = {out.m0dif[iz]:11.5f} +- {out.m0diferr[iz]:9.5f} "
            f" ({zbins.lo[iz]:.3f} < z < {zbins.hi[iz]:.3f}, N={out.nfit[iz]}) ")
    lines.append(f" M0avg          = {out.m0avg:14.5f} ")
    lines.append(f" sum_N over m0_[nn] bins: {int(out.nfit.sum())} "
                 f" (compare to NSNFIT={result.nsnfit})")
    return lines


BBC_VERSION = 4   # SALT2mu.c output-format version stamp


def _provenance_header(out: BBCOutput) -> list:
    """The block SNANA opens both .M0DIF and .FITRES with (write_version_info).

    SNANA_VERSION carries jax's own build string: the field name is part of the
    format that downstream readers key on, so the name is kept and the value says
    which program actually wrote the file.
    """

    lines = [f"{k + ':':<30s}{v} " for k, v in (out.version_photometry or {}).items()]
    # after SNANA's own lines: its create_covariance reads only the first ten
    cosmology = [f"{f'SIM_{k}(BIASCOR):':<30s}{v} "
                 for k, v in out.biascor_cosmology.items()]
    return lines + [f"ISDATA_REAL:   {int(out.isdata_real)} ",
                    f"{snana_version_comment()} ",
                    f"{program_comment('salt2mu.py')} ",
                    f"BBC_VERSION:   {BBC_VERSION} "] + cosmology


def _cosmo_header(config):
    """Under BLINDMASK_FIXPAR SNANA prints the unblinded value and the offset
    formula rather than the blinded number (prep_blindVal_strings, :23716)."""

    blinded = config.blindflag & BLINDMASK_FIXPAR
    out = []
    for name, attr, slot in (("Omega_L", "omega_l", 9), ("w0", "w0", 11)):
        amp, phase = config.blindpar.get(slot, (0.0, 0.0))
        if blinded and amp != 0.0:
            val = f"{config._unblind.get(attr, getattr(config, attr)):8.4f} + {amp:6.4f}*cos({phase:f})"
        else:
            val = f"{getattr(config, attr):.3f}"
        out.append(f"  REF {name:<12s}:  {val} ")
    return out


def _blind_message(config) -> list:
    """write_blindFlag_message (SALT2mu.c:24880)."""

    flag = config.blindflag
    if not flag:
        return []
    lines = ["", " *** WARNING: RESULTS ARE BLINDED *** "]
    if flag & BLINDMASK_MUz:
        lines.append(f" m0_## values have blind offset = cos( z * {BLIND_COSINE_M0:.3f} ) ")
    else:
        for name, attr, slot in (("Omega_L", "omega_l", 9), ("w0", "w0", 11)):
            amp, phase = config.blindpar.get(slot, (0.0, 0.0))
            if amp != 0.0:
                base = config._unblind.get(attr, getattr(config, attr))
                lines.append(f"    {name:<13s}= {base:8.4f} + {amp:6.4f}*cos({phase:f}) ")
    return lines + [""]


def output_path(config, suffix: str) -> str:
    """Path for an output file. SNANA leaves PREFIX unset ("NONE", SALT2mu.c:6492)
    and has no default filename, so writing without prefix= is an error."""
    if config.prefix == "NONE":
        raise SNANAError("prefix= is required to write output files")
    return f"{config.prefix}{suffix}"


def m0dif_columns(zbins: ZBinGrid, out: BBCOutput) -> dict:
    """The M0DIF columns: ROW zHDMIN zHDMAX zHD MUDIF MUDIFERR MUREF NFIT.

    What :func:`write_m0dif` writes, and what a cosmology fit downstream reads
    back, so a caller holding the output needs no file.
    """
    return {
        "ROW": np.arange(1, zbins.nbin + 1),
        "zHDMIN": zbins.lo, "zHDMAX": zbins.hi, "zHD": out.zbin_z,
        "MUDIF": out.m0dif, "MUDIFERR": out.m0diferr, "MUREF": out.muref,
        "NFIT": out.nfit,
    }


def write_m0dif(path, zbins: ZBinGrid, out: BBCOutput, result, config, warnings=()):
    """One row per z bin: ROW zHDMIN zHDMAX zHD MUDIF MUDIFERR MUREF NFIT."""
    header = _provenance_header(out) + \
        [None, "Reference cosmology params for MUDIF: "] + _cosmo_header(config)
    for name in ("alpha0", "beta0"):
        header.append(f"  FIT {name:<11s} :  "
                      f"{result.nuisance.get(name, 0.0):.4f} "
                      f"+- {result.nuis_err.get(name, 0.0):.4f} ")
    header += [""] + _blind_message(config) + list(warnings) + [
        None, "For cosmology fit: zHD, MU=MUREF+MUDIF, MUERR=MUDIFERR", None]
    cols = m0dif_columns(zbins, out)
    fmt = {"ROW": "%d", "zHDMIN": "%.5f", "zHDMAX": "%.5f", "zHD": "%.5f",
           "MUDIF": "%.5f", "MUDIFERR": "%.5f", "MUREF": "%.5f", "NFIT": "%d"}
    write_varnames_table(path, list(cols), cols, fmt, rowkey="ROW:",
                         header_comments=header, blank_separator=False)


# write_fitres_line_append column order and formats (SALT2mu.c:25281)
_BBC_FMT = {"CUTMASK": "%d",
            "MU": "%8.5f", "MUMODEL": "%8.5f", "MUERR": "%8.5f",
            "MUERR_RENORM": "%7.4f", "MUERR_RAW": "%7.4f", "MUERR_VPEC": "%7.4f",
            "MURES": "%7.4f", "MUPULL": "%6.3f",
            "M0DIF": "%7.4f", "M0DIFERR": "%7.4f",
            "CHI2": "%.2f", "CHI2_BEAMS": "%.2f", "PROBCC_BEAMS": "%.8e",
            "biasCor_nevt": "%4d", "biasCor_mu": "%6.4f", "biasCorErr_mu": "%6.3f",
            "biasCor_mB": "%6.4f", "biasCor_x1": "%6.3f", "biasCor_c": "%6.4f",
            "biasCor_muCOVSCALE": "%6.3f", "biasCor_muCOVADD": "%6.4f",
            "IDSAMPLE": "%d", "IZBIN": "%2d"}


def _cutmask_write_warning(config) -> List[str]:
    """write_NWARN(FLAG=1) (SALT2mu.c:24752): the FITRES, and only the FITRES,
    says when the written row set is not the fitted sample."""
    if config.cutmask_write == 0:
        return []
    return [f"WARNING(MINOR): cutmask_write={config.cutmask_write}"]


def write_fitres(path, out: BBCOutput, zbins: ZBinGrid, result, config,
                 passthrough, warnings=()):
    """One row per (accepted) SN: passthrough columns then the BBC set.

    ``passthrough`` is an ordered ``{name: array}`` (already masked to the fitted
    events, aligned with the output arrays) and must start with ``CID``.
    """
    cols, fmt = {}, {}
    for name, arr in passthrough.items():
        cols[name] = np.asarray(arr)
        fmt[name] = _fmt_for(arr)

    izbin = out.izbin
    beams = out.probcc is not None
    # 0 for every written row: only cutmask_write=0 (the fit sample) is supported
    bbc = {"CUTMASK": np.zeros(len(izbin), dtype=np.int64),
           "MU": out.mu, "MUMODEL": out.mumodel, "MUERR": out.muerr}
    if out.muerr_renorm is not None:
        bbc["MUERR_RENORM"] = out.muerr_renorm
    bbc.update({"MUERR_RAW": out.muerr_raw, "MUERR_VPEC": out.muerr_vpec,
                "MURES": out.mures,
                # the written pull is capped, one-sided, so the column keeps its
                # width (SALT2mu.c:25296); the CHI2 column is not capped
                "MUPULL": np.minimum(out.mupull, MUPULL_MAX),
                "M0DIF": out.m0dif[izbin], "M0DIFERR": out.m0diferr[izbin]})
    bbc["CHI2_BEAMS" if beams else "CHI2"] = out.chi2
    if beams:
        bbc["PROBCC_BEAMS"] = out.probcc
    if out.biascor:
        bbc.update(out.biascor)
        bbc["IDSAMPLE"] = out.idsample
    # Rebinning also needs the redshift bin when bias corrections are off.
    bbc["IZBIN"] = izbin
    for name, arr in bbc.items():
        cols[name] = np.asarray(arr)
        fmt[name] = _BBC_FMT.get(name, "%.5g")

    header = _provenance_header(out) + [
        None,
        "HESSIAN",
        f"CPU: {out.yaml.cpu_minutes:.2f} minutes",
    ] + _blind_message(config) + [
        "MU-RESIDUAL NOTE: MURES = MU-MUMODEL ",
        "For cosmology fit: use zHD, MU, MUERR_RENORM",
        "  [MUERR_RENORM is Eq 10 of arxiv:2306.05819, MUERR is sig_{mu,i}]",
        None] + _cutmask_write_warning(config) + list(warnings)
    post = [None] * 3 + _fit_summary(result, out, zbins, config) + [None]
    if config.cutmask_write == -9:
        # no SN rows, so no VARNAMES line either: the file is the header and the
        # fit-results block (write_fitres_driver, SALT2mu.c:24254, :24399)
        with open(path, "w") as f:
            for line in header + post:
                f.write("\n" if line is None else f"# {line}\n")
        return
    write_varnames_table(path, list(cols), cols, fmt, rowkey="SN:",
                         header_comments=header + [None],
                         blank_separator=False, post_varnames_comments=post)


def write_cov(path, cov_m0):
    """M0-block statistical covariance (write_M0_cov): nzbin, then row-major
    ``%le`` values; SNANA's 1e5 diagonal marks unfloated bins."""
    cov = np.asarray(cov_m0)
    nbin = cov.shape[0]
    with open(path, "w") as f:
        f.write(f"{nbin}\n")
        for iz in range(nbin):
            f.write(f"# ---------- begin row {iz} ------------ \n")
            for val in cov[iz]:
                f.write(f"{val:.6e}\n")


def write_yaml(path, summary, result, config, timings=None):
    """SNANA's .YAML summary for the pipeline aggregator (write_yaml_info,
    SALT2mu.c:23571): per-event-type counts, reject tallies, the by-sample
    breakdown, and the fitted parameters; then, jax's own, the seconds each
    section of the cell took, so a slow run says where.

    NFIT_EXEC is always 1: jax does not re-fit (see crazy_m0_errors).
    """
    rows = [(f"NEVT_{k}:", f"{summary.nevt.get(k, 0)} ")
            for k in ("DATA", "BIASCOR", "CCPRIOR")]
    rows += [("NEVT_REJECT_CUTWIN:", summary.nreject_cutwin),
             ("NEVT_REJECT_BIASCOR:", summary.nreject_biascor),
             ("ABORT_IF_ZERO:", summary.nevt.get("DATA", 0)),
             ("NWARN_CRAZYERR:", summary.ncrazy),
             ("NFIT_EXEC:", 1)]

    with open(path, "w") as f:
        for key, val in rows:
            f.write(f"{key:<22.22s} {val}\n")
        f.write(f"{'CPU_MINUTES:':<22.22s} {summary.cpu_minutes:.2f}\n")
        f.write(f"{'ISDATA_REAL:':<22.22s} {int(summary.isdata_real)}\n")

        if summary.sample_names:
            f.write("\n")
            _yaml_list(f, "SAMPLE_LIST:", summary.sample_names, lead_first=False)
            _yaml_list(f, "IDSAMPLE_LIST:", summary.sample_ids)
            for evtype in ("DATA", "BIASCOR", "CCPRIOR"):
                counts = summary.nevt_by_sample.get(evtype)
                if counts:
                    _yaml_list(f, f"NEVT_{evtype}_bySAMPLE:", counts)
            f.write("\n")
            _yaml_list(f, "NREJECT_CUTWIN_bySAMPLE:",
                       summary.nreject_cutwin_by_sample)
            _yaml_list(f, "NREJECT_BIASCOR_bySAMPLE:",
                       summary.nreject_biascor_by_sample)

        f.write("\nBBCFIT_RESULTS:\n")
        nsnfit = result.nsnfit
        f.write(f"  - NSNFIT:            {nsnfit:5d}     "
                f"{int(np.sqrt(nsnfit) + 0.5)}\n")
        f.write(f"  - SIGINT:            {result.sigint:.5f}    0\n")
        f.write(f"  - CHI2RED_IA:        "
                f"{result.chi2sum_ia / (result.ndof_ia + 1.0e-8):.3f}    0\n")
        for name, err in result.nuis_err.items():
            f.write(f"  - {_param_label(name, config) + ':':<18s}"
                    f"{result.nuisance[name]:11.5f} {err:8.5f} \n")
        if summary.contam_data is not None:
            f.write(f"  - {'CONTAM_DATA:':<20s}  "
                    f"{summary.contam_data:.5f}  {0.0:.5f}   "
                    f"# photo-classified subset \n")
        if summary.contam_true is not None:
            f.write(f"  - {'CONTAM_TRUE:':<20s}  "
                    f"{summary.contam_true:.5f}  "
                    f"{summary.contam_true_err:.5f}    "
                    f"# photo-classified subset (sim data only) \n")
        f.write("\nFIT_DIAGNOSTICS:\n")
        f.write(f"  max_gradient: {result.max_gradient:.8e}\n")
        f.write(f"  max_projected_gradient: {result.max_projected_gradient:.8e}\n")
        if timings:
            f.write("\nTIMING_SEC:\n")
            for name, value in timings.items():
                f.write(f"  {name + ':':<18s} {value:.3f}\n")


def _yaml_list(f, key, values, lead_first=True):
    """catVarList_with_comma joins with "," after prefixing each entry with a
    space, so the body reads " a, b". SAMPLE_LIST is the exception: its first
    entry ("ALL") is written without one."""
    parts = [f" {v}" for v in values]
    if not lead_first and parts:
        parts[0] = parts[0].lstrip()
    f.write(f"{key:<25.25s} {','.join(parts)}\n")


def write_passcuts_list(path, which, tv, idsample):
    """SNANA update_fitres_passcuts: the CIDs of a biasCor/CC-prior sample that pass
    the BBC cuts, as a diagnostic list. No effect on the fit.

    ``idsample`` is the per-event sample assignment, which is only known once the
    samples are partitioned, hence after the maps are built.
    """
    has_field = tv.field is not None
    with open(path, "w") as f:
        f.write(f"# list of {which} events passing BBC cuts \n\n")
        f.write("VARNAMES:  CID  IDSAMPLE  IDSURVEY" + ("  FIELD\n" if has_field else "\n"))
        for i in np.where(tv.cutmask)[0]:
            row = (f"SN: {str(tv.name[i]):>12s}  {int(idsample[i]):2d}  "
                   f"{int(tv.idsurvey[i]):3d}  ")
            if has_field:
                row += f"{tv.field[i]} "
            f.write(row + "\n")


def write_m0_csv(path, zbins, out, config):
    """Binned Hubble diagram for CosmoMC (write_M0_csv, SALT2mu.c:23962). MU is the
    reference distance at the reported z plus M0DIF, not the weighted MUREF column."""
    zhd = out.zbin_z
    muref = mu_at(zhd, config)
    with open(path, "w") as f:
        f.write("# name   zcmb zhel zerr   mu muerr \n")
        for iz in range(zbins.nbin):
            f.write(f"BIN{iz:02d}   {zhd[iz]:.5f} {zhd[iz]:.5f} {0.0:.1f}   "
                    f"{muref[iz] + out.m0dif[iz]:.4f} {out.m0diferr[iz]:.4f} \n")


def write_chi2grid(path, spec, inputs, result, nbin):
    """chi2 scan over the model's ``chi2grid_axes`` (write_chi2grid,
    SALT2mu.c:24085): each row varies one parameter over its fixed range with
    the others held at their fitted values.

    SNANA scans ``fcn`` itself, so under a CC prior this is the BEAMS sum and not the
    Ia chi2. It runs from outFile_driver (:23511), after the fit has restored
    muerrsq_last to its own errors, hence the swapped-in muerrsq_final.
    Joint mode scans its normalized likelihood at the fitted intrinsic scatter.
    """
    post_fit = inputs._replace(muerrsq_last=result.muerrsq_final)
    spec = dataclasses.replace(spec, floated=(),
                               with_log=result.fitflag_sigmb == 2)
    axes = spec.model.chi2grid_axes
    # a scanned parameter the model has not got stays at SNANA's parval 0
    fitted = {name: result.nuisance.get(name, 0.0) for _, name, _ in axes}
    steps = {name: (hi - lo) / nbin for _, name, (lo, hi) in axes}
    m0 = jnp.asarray(result.m0)

    def chi2(values):
        fixed = dict(result.nuisance, sigint=result.sigint, **values)
        loss = make_loss(dataclasses.replace(spec, defaults=tuple(sorted(fixed.items()))),
                         post_fit)
        if spec.mode == "joint":
            return float(loss(jnp.append(m0, result.sigint)))
        return float(loss(m0, result.sigint, post_fit.muerrsq_last))

    with open(path, "w") as f:
        labels = [label for label, _, _ in axes]
        f.write("VARNAMES:  ROW  " + "  ".join(labels)
                + " " + " ".join(f"chi2_{label}" for label in labels) + " \n")
        for i in range(nbin):
            point = {name: lo + steps[name] * i for _, name, (lo, _) in axes}
            grid = " ".join(f"{point[name]:.4f}" for _, name, _ in axes)
            scans = "  ".join(f"{chi2({**fitted, name: point[name]}):.6e}"
                              for _, name, _ in axes)
            f.write(f"ROW: {i+1:5d}  {grid}   {scans} \n")


_DOCANA = "DOCUMENTATION:"
_DOCANA_END = "DOCUMENTATION_END:"
_PAD = "    "


def _cat_docana_notes(paths, config, droplist) -> List[str]:
    """The CAT_FILE_NOTES block embedded before DOCUMENTATION_END (:23890)."""

    body = [snana_version_comment()]
    if config.cat_prescale <= 1:
        body.append(f"COLUMN_APPEND:  {config.append_varname_missing} ")
        body.append(f"COLUMN_DROP:    {droplist} ")
    else:
        body.append(f"PRESCALE: {config.cat_prescale}")
    body.append("INPUT_CAT_FILES: ")
    body += [f" - {p} " for p in paths]
    return [f"{_PAD}CAT_FILE_NOTES:"] + [f"{_PAD}{_PAD}{n}" for n in body]


def _cat_comment_notes(tables, paths, config, droplist) -> List[str]:
    """The same provenance as a comment block when there is no DOCANA (:23923)."""

    lines = []
    # SNTABLE_VERSION_PHOTOMETRY accumulates one entry per file read, comma-joined.
    photometry = ",".join(t.meta["VERSION_PHOTOMETRY"] for t in tables
                          if t.meta.get("VERSION_PHOTOMETRY"))
    if photometry:
        lines.append(f"# VERSION_PHOTOMETRY: {photometry} ")
    lines.append(f"# {snana_version_comment()} ")
    lines.append("# Catenated data files: ")
    lines += [f"#   + {p} " for p in paths]
    if config.cat_prescale <= 1:
        lines.append(f"#\n# Appended columns: {config.append_varname_missing} ")
        lines.append(f"#\n# Dropped columns: {droplist} ")
    else:
        lines.append(f"# PRESCALE = {config.cat_prescale}")
    lines.append("#")
    return lines


def _cat_header(tables, paths, config, droplist) -> str:
    """write_cat_info (SALT2mu.c:25005).

    With a DOCUMENTATION block in the first input, copy it and embed the cat notes
    just before its end key; otherwise write the same provenance as comments.
    """
    docana = _read_docana(paths[0])
    if docana is not None:
        notes = _cat_docana_notes(paths, config, droplist)
        return "\n".join(docana + [""] + notes + [_DOCANA_END, "", ""]) + "\n"
    return "\n".join(_cat_comment_notes(tables, paths, config, droplist)) + "\n"


def _read_docana(path):
    """Lines of the leading DOCUMENTATION block, excluding its end key; None if absent."""

    out = []
    with open_text(path) as f:
        for line in f:
            stripped = line.rstrip("\n")
            key = stripped.split()[0] if stripped.split() else ""
            if not out and key != _DOCANA:
                return None
            if key == _DOCANA_END:
                return out
            out.append(stripped)
    return None


def _cat_droplist(varnames_by_file, kept) -> str:
    """Input columns present in some file but not carried into the output.

    catVarList_with_comma (SALT2mu.c:9784) joins them with commas.
    """
    seen = dict.fromkeys(n for names in varnames_by_file for n in names)
    return ",".join(n for n in seen if n not in set(kept))


def write_cat_fitres(path, tables, paths, config) -> int:
    """Concatenate the input rows and return how many were written.

    write_fitres_driver under cat_only (SALT2mu.c:24172): no cuts, no fit, and no
    appended BBC block (define_varnames_append returns early, :24688). Values are
    re-emitted as their original tokens with "-9.0" where a file lacks the column,
    matching write_fitres_line (:24501), so the read must be ``as_text``.
    """

    varnames_by_file = [t.varnames_by_file[0] for t in tables]
    varnames = output_varnames(varnames_by_file, config)
    droplist = _cat_droplist(varnames_by_file, varnames)
    prescale = config.cat_prescale
    nline = nwrite = 0

    with open(path, "w") as f:
        f.write(_cat_header(tables, paths, config, droplist))
        f.write("VARNAMES: " + "".join(f"{n} " for n in varnames) + "\n\n")
        for table in tables:
            columns = [table.get(n) for n in varnames]
            for irow in range(table.size):
                if table.rowkey[irow] not in _ROW_KEYS:
                    continue
                # the prescale counter runs across all files, and only over rows
                # that got this far (:23283 rejects other keys before :23290)
                nline += 1
                if nline % prescale != 0:
                    continue
                words = "".join(
                    f"{MISSING_TOKEN if col is None else col[irow]} "
                    for col in columns)
                f.write(f"{table.rowkey[irow]} {words}\n")
                nwrite += 1
    return nwrite
