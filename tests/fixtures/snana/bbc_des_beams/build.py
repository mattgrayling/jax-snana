#!/usr/bin/env python
"""Assemble the DES BEAMS fixture set from its three SNANA sims.

Derived fixtures, so not registered in tests/fixtures/regenerate.py (which runs
SNANA from an input file). Run this after regenerating any parent, or these go
stale silently:

    python build.py [--check]

Parents, all DES griz on the same SIMLIB, kcor and reference cosmology, and all
IDSURVEY=10 -- SALT2mu partitions by survey, so a sample from a different survey
would land in an IDSAMPLE with no biasCor and abort in print_eventStats:

    ../bbc_des_data/FITRES.TEXT      Ia data
    ../bbc_des_biascor/FITRES.TEXT   Ia biasCor for the 5D map
    ../bbc_ccprior_sim/FITRES.TEXT   CC sim for the BEAMS prior

The four BIASCOR_* variants are generated in temporary directories by pytest
and refs.py; the command above writes only the other derived inputs.

Outputs, each the parent reduced to the columns SALT2mu.exe reads, values copied
as the parent's own tokens so no precision is lost:

    DATA.FITRES         the clean Ia sample
    BIASCOR.FITRES      the biasCor sample
    CCPRIOR.FITRES      the CC prior
    CONTAM_DATA.FITRES  DATA plus N_CONTAM CC events at CIDs 100000+, so
                        SALT2mu.exe sees one table with contamination in it
    CONTAM_MIXED.FITRES CONTAM_DATA plus TYPE: Ia are 1, contaminants alternate
                        101/102, for sntype_select and force_probcc0
    CONTAM_SIMTYPE.FITRES  CONTAM_DATA plus sim truth and TYPE (every 4th Ia
                        spec-typed 1), for CONTAM_TRUE and the spec subset
    DATA_MIXTMPL.FITRES DATA with every other SIM_TEMPLATE_INDEX set to 101, so
                        the CC and Ia prescales each act on half the sample
    DATA_DUP.FITRES     DATA with its first 20 rows repeated, for iflag_duplicate
    DATA_SPLIT_A/B      DATA halved into two datafiles with different columns
                        (A adds PROB_TEST, B drops SNRSUM), for append_varname
    DATA_MASS/BIASCOR_MASS  a uniform HOST_LOGMASS in [8,12] injected at a fixed
                        seed, for the gamma0 and 4D-logmass tests; the DES sim
                        carries no host masses (every event is -9999)
    DATA_SDSS/BIASCOR_SDSS  every other row relabelled IDSURVEY=1, giving two
                        IDSAMPLEs for the sigint_fix per-sample tests
    DATA_FIELD/BIASCOR_FIELD  FIELD cycling C3, X3, C3+X3, E2, for the field-group
                        and exact field-list tests
    catA/catB/stray     small hand-shaped tables for the cat_only tests: distinct
                        PROBA/PROBB and ONLY_A/ONLY_B columns to exercise the
                        column union, and a stray non-row key in stray.FITRES
    DATA_PHOTOZ.FITRES  DATA plus OPT_PHOTOZ/zPRIOR/zPRIORERR, half the rows
    BIASCOR_PHOTOZ      flagged photo-z, for the zSPEC/zPHOT IDSAMPLE split
    DATA_REAL.FITRES    DATA without the SIM_* columns, so SALT2mu reads it as
                        real data (blinding, and the CC prescale with nothing
                        to act on) without the contaminants CONTAM_* carry
    cid_reject.list     first 100 CIDs      cid_select.list  first 1700 CIDs

PROBIA is the one fabricated column: a truth-based stand-in for a classifier
(Ia ~0.95, CC ~0.05), assigned deterministically from the row index so the file
is reproducible. It separates the classes perfectly, which is fine for a parity
test -- both programs read the same numbers -- but means these fixtures cannot
catch an error confined to genuinely ambiguous events.

Why a dedicated DES set: the CC prior is the per-z-bin spread of mu - mumodel over
the CC events, so it needs x1 and c constrained. The LSST fixtures' CC sample had
77% of its events sitting on SALT's initial guess (x1=0.001, c=0.010, x1ERR=0.30,
cERR=0.03) because the fits never converged on a ~50-epoch cadence, which collapsed
the top z bin's spread to 0.004. A near-delta prior like that flips an ordinary
event's classification under a 1e-3 parameter change. DES's single-season cadence
(~92 epochs) gives median x1ERR = 0.69 and no placeholder rows.
"""


import argparse
import gzip
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SNANA = HERE.parent

BIASCOR_VARIANTS = ("BIASCOR_MASS.FITRES", "BIASCOR_SDSS.FITRES",
                   "BIASCOR_FIELD.FITRES", "BIASCOR_PHOTOZ.FITRES")

N_CONTAM = 120              # ~7% contamination of the 1775-event data sample
CID_CONTAM_START = 100000
X1MAX, CMAX = 3.0, 0.3      # BBC's own window; outside it an event is cut anyway

PROBIA_IA = ("0.999 0.986 0.972 0.964 0.951 0.947 0.938 0.929 0.918 0.907 "
             "0.896 0.884 0.947 0.962 0.933 0.909 0.975 0.881 0.955 0.921").split()
PROBIA_CC = ("0.001 0.014 0.028 0.036 0.049 0.053 0.062 0.071 0.082 0.093 "
             "0.104 0.116 0.053 0.038 0.067 0.091 0.025 0.119 0.045 0.079").split()

DATA_COLUMNS = ("CID IDSURVEY FIELD zHD zHDERR zHEL VPEC VPECERR MWEBV "
                "HOST_LOGMASS x1 x1ERR c cERR x0 x0ERR mB mBERR "
                "COV_x1_c COV_x1_x0 COV_c_x0 SNRMAX1 SNRSUM").split()

# DATA keeps its SIM truth so SALT2mu reads it as a simulation. CONTAM_DATA
# deliberately does not: without SIM_TEMPLATE_INDEX, SALT2mu treats the table as real
# data, which is what the BEAMS tests need -- the contaminants must be classified by
# the fit rather than looked up. The side effect is that blinding defaults on for a
# real-data table, so anything running on CONTAM_DATA must set blindflag explicitly.
SIM_COLUMNS = ("SIM_TEMPLATE_INDEX SIM_ZCMB SIM_DLMAG SIM_WGT_POP SIM_x1 SIM_c "
               "SIM_alpha SIM_beta SIM_gammaDM SIM_x0 SIM_mB").split()

BIASCOR_COLUMNS = ("CID IDSURVEY FIELD zHD zHDERR HOST_LOGMASS SNRMAX1 "
                   "x1 x1ERR c cERR x0 x0ERR COV_x1_c COV_x1_x0 COV_c_x0 "
                   "SIM_ZCMB SIM_DLMAG SIM_WGT_POP SIM_x1 SIM_c SIM_alpha "
                   "SIM_beta SIM_gammaDM SIM_x0 SIM_mB SIM_TEMPLATE_INDEX").split()

CCPRIOR_COLUMNS = ("CID IDSURVEY FIELD zHD zHDERR HOST_LOGMASS SNRMAX1 "
                   "x1 x1ERR c cERR x0 x0ERR COV_x1_c COV_x1_x0 COV_c_x0 "
                   "SIM_ZCMB SIM_DLMAG SIM_WGT_POP SIM_gammaDM SIM_x0 "
                   "SIM_TEMPLATE_INDEX").split()


def read_rows(path: Path):
    varnames, rows = None, []
    if not path.is_file():
        path = path.with_name(path.name + ".gz")
    if not path.is_file():
        sys.exit(f"missing parent {path}; run "
                 f"`python tests/fixtures/regenerate.py --only {path.parent.name}`")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        lines = stream.read().splitlines()
    for line in lines:
        tok = line.split()
        if not tok:
            continue
        if tok[0] == "VARNAMES:":
            varnames = tok[1:]
        elif tok[0] == "SN:" and varnames is not None:
            rows.append(dict(zip(varnames, tok[1:])))
    return rows


def table(columns, rows, header):
    lines = [f"# {h}" for h in header] + ["#", "VARNAMES: " + " ".join(columns)]
    lines += ["SN: " + " ".join(r[c] for c in columns) for r in rows]
    return "\n".join(lines) + "\n"


def in_box(row) -> bool:
    return abs(float(row["x1"])) < X1MAX and abs(float(row["c"])) < CMAX


def build():
    data = read_rows(SNANA / "bbc_des_data" / "FITRES.TEXT")
    biascor = read_rows(SNANA / "bbc_des_biascor" / "FITRES.TEXT.gz")
    cc = read_rows(SNANA / "bbc_ccprior_sim" / "FITRES.TEXT.gz")

    pool = [r for r in cc if in_box(r)]
    if len(pool) < N_CONTAM:
        sys.exit(f"only {len(pool)} CC events inside the x1/c box, need {N_CONTAM}")
    # evenly spaced through the pool so the contaminants span its redshift range
    step = len(pool) / N_CONTAM
    picked = [pool[int(i * step)] for i in range(N_CONTAM)]

    contam = [{**r, "PROBIA": PROBIA_IA[i % len(PROBIA_IA)]}
              for i, r in enumerate(data)]
    contam += [{**r, "CID": str(CID_CONTAM_START + i),
                "PROBIA": PROBIA_CC[i % len(PROBIA_CC)]}
               for i, r in enumerate(picked)]

    mixed = [{**r, "TYPE": "1" if int(r["CID"]) < CID_CONTAM_START
              else str(101 + int(r["CID"]) % 2)} for r in contam]

    # contaminated sample with truth and TYPE: every 4th Ia is spec-typed 1,
    # so typeIa_ccprior=1 gives a nonzero spec-confirmed subset
    contam_simtype = [
        {**r, "TYPE": (("1" if i % 4 == 0 else "0")
                       if int(r["CID"]) < CID_CONTAM_START
                       else str(101 + int(r["CID"]) % 2))}
        for i, r in enumerate(contam)]

    mixtmpl = [{**r, "SIM_TEMPLATE_INDEX": "101" if i % 2 else r["SIM_TEMPLATE_INDEX"]}
               for i, r in enumerate(data)]

    split_a = [{**r, "PROB_TEST": f"{0.5 + 0.001 * i:.4f}"}
               for i, r in enumerate(data[:len(data) // 2])]
    split_b = data[len(data) // 2:]

    cids = [r["CID"] for r in data]

    # OPT_PHOTOZ is written by no SNANA program; SALT2mu reads it as an optional
    # column defaulting to 0 (SALT2mu.c:8538), so a survey supplies it externally.
    # Both tables must carry it: biasCor has no zHELERR, so without zPRIOR every
    # biasCor event classifies as spec-z and the zPHOT sample empties.
    def photoz(rows):
        out = []
        for i, r in enumerate(rows):
            opt = 2 if i % 2 else 0
            zerr = (0.02 if (i // 2) % 2 == 0 else 0.10) if opt else 0.001
            out.append({**r, "OPT_PHOTOZ": str(opt), "zPRIOR": r["zHD"],
                        "zPRIORERR": f"{zerr:.4f}"})
        return out

    data_pz, biascor_pz = photoz(data), photoz(biascor)

    # cat_only concatenates datafiles and exits -- no fit -- so these stay small.
    # The columns are what matters: PROBA/PROBB match the default append pattern,
    # ONLY_A/ONLY_B appear in one file each and must be dropped from the union.
    cat_cols = ("CID IDSURVEY TYPE FIELD CUTFLAG_SNANA ERRFLAG_FIT "
                "zHEL zHELERR zHD zHDERR").split()
    def cat(rows, prob, only, start):
        return [{**r, "TYPE": "1", "CUTFLAG_SNANA": "3", "ERRFLAG_FIT": "0",
                 prob: f"{0.11 + 0.01 * i:.2f}", only: f"{7.0 + i:.1f}"}
                for i, r in enumerate(rows[start:start + (20 if prob == "PROBA" else 15)])]
    catA, catB = cat(data, "PROBA", "ONLY_A", 0), cat(data, "PROBB", "ONLY_B", 20)

    # Second survey is SDSS (IDSURVEY=1), deliberately *lower* than DES (10), so
    # surveygroup order disagrees with numeric order.
    # The sim writes HOST_LOGMASS = -9999 for every event, so the mass-step tests
    # need one injected. Uniform [8,12] about the logmass_cen=10 they use, from a
    # fixed seed so the tables are reproducible.
    rng = __import__("random").Random(4131)
    def with_mass(rows):
        return [{**r, "HOST_LOGMASS": f"{rng.uniform(8.0, 12.0):.4f}"} for r in rows]
    data_mass, biascor_mass = with_mass(data), with_mass(biascor)

    def sdss_split(rows):
        return [{**r, "IDSURVEY": "1"} if i % 2 else dict(r)
                for i, r in enumerate(rows)]
    data_sdss, biascor_sdss = sdss_split(data), sdss_split(biascor)

    fields = ("C3", "X3", "C3+X3", "E2")
    def field_split(rows):
        return [{**r, "FIELD": fields[i % 4]} for i, r in enumerate(rows)]
    data_field, biascor_field = field_split(data), field_split(biascor)

    return {
        "DATA.FITRES": table(
            DATA_COLUMNS + SIM_COLUMNS, data,
            ["Clean Ia data sample, from ../bbc_des_data/FITRES.TEXT.",
             f"{len(data)} events, with SIM truth so SALT2mu reads it as a",
             "simulation (and so does not blind it). Built by build.py."]),
        "BIASCOR.FITRES": table(
            BIASCOR_COLUMNS, biascor,
            ["Ia biasCor sample, from ../bbc_des_biascor/FITRES.TEXT.",
             f"{len(biascor)} events at a single sim (alpha, beta) = (0.14, 3.1),",
             "so the 5D map is a 1x1 abg grid. Built by build.py."]),
        "CCPRIOR.FITRES": table(
            CCPRIOR_COLUMNS, cc,
            ["CC prior for the BEAMS contamination model, from",
             f"../bbc_ccprior_sim/FITRES.TEXT. {len(cc)} events, of which",
             f"{len(pool)} fall inside BBC's |x1|<3, |c|<0.3 box.",
             "Built by build.py."]),
        "catA.FITRES": table(cat_cols + ["PROBA", "ONLY_A"], catA,
                             ["20 DES rows for the cat_only tests. Built by build.py."]),
        "catB.FITRES": table(cat_cols + ["PROBB", "ONLY_B"], catB,
                             ["15 DES rows with different columns from catA.",
                              "Built by build.py."]),
        "DATA_MASS.FITRES": table(
            DATA_COLUMNS + SIM_COLUMNS, data_mass,
            ["DATA with a uniform HOST_LOGMASS in [8,12] (seed 4131), for the",
             "mass-step tests. Built by build.py."]),
        "BIASCOR_MASS.FITRES": table(
            BIASCOR_COLUMNS, biascor_mass,
            ["BIASCOR with the same injected HOST_LOGMASS. Built by build.py."]),
        "DATA_SDSS.FITRES": table(
            DATA_COLUMNS + SIM_COLUMNS, data_sdss,
            ["DATA with every other row relabelled IDSURVEY=1 (SDSS), for the",
             "per-IDSAMPLE sigint_fix tests. Built by build.py."]),
        "BIASCOR_SDSS.FITRES": table(
            BIASCOR_COLUMNS, biascor_sdss,
            ["BIASCOR split the same way; both are needed or print_eventStats",
             "aborts on a data IDSAMPLE with no biasCor. Built by build.py."]),
        "DATA_FIELD.FITRES": table(
            DATA_COLUMNS + SIM_COLUMNS, data_field,
            ["DATA with FIELD cycling C3, X3, C3+X3, E2, for the field-group and",
             "exact field-list tests. Built by build.py."]),
        "BIASCOR_FIELD.FITRES": table(
            BIASCOR_COLUMNS, biascor_field,
            ["BIASCOR with the same FIELD cycle. Built by build.py."]),
        "DATA_PHOTOZ.FITRES": table(
            DATA_COLUMNS + SIM_COLUMNS
            + ["zHELERR", "OPT_PHOTOZ", "zPRIOR", "zPRIORERR"], data_pz,
            ["DATA with the photo-z columns: every other event OPT_PHOTOZ=2, with",
             "zPRIORERR alternating 0.02/0.10 so zspec_errmax_idsample splits them.",
             "Built by build.py."]),
        "BIASCOR_PHOTOZ.FITRES": table(
            BIASCOR_COLUMNS + ["OPT_PHOTOZ", "zPRIOR", "zPRIORERR"], biascor_pz,
            ["BIASCOR with the same photo-z columns. Doctoring only the data aborts",
             "in print_eventStats once a biasCor is used: a data IDSAMPLE with no",
             "biasCor events is fatal. Built by build.py."]),
        "DATA_REAL.FITRES": table(
            DATA_COLUMNS, data,
            ["The clean Ia sample with no SIM_* columns, so SALT2mu reads it as real",
             "data. Same events as DATA.FITRES -- no contaminants, so the fit stays",
             "well away from the beta bound. Built by build.py."]),
        "CONTAM_MIXED.FITRES": table(
            DATA_COLUMNS + ["PROBIA", "TYPE"], mixed,
            [f"CONTAM_DATA with TYPE: {len(data)} Ia at 1, {N_CONTAM} contaminants",
             "alternating 101/102. Built by build.py."]),
        "CONTAM_SIMTYPE.FITRES": table(
            DATA_COLUMNS + ["SIM_TEMPLATE_INDEX", "SIM_ZCMB", "SIM_DLMAG",
                            "SIM_WGT_POP", "SIM_gammaDM", "SIM_x0",
                            "PROBIA", "TYPE"], contam_simtype,
            ["CONTAM_DATA with sim truth and TYPE: SIM_TEMPLATE_INDEX marks the",
             f"{N_CONTAM} true contaminants, every 4th Ia is spec-typed TYPE=1,",
             "for CONTAM_TRUE and the spec-confirmed subtraction. Only the SIM",
             "columns both parents carry (the CC sim has no SALT truth).",
             "Built by build.py."]),
        "DATA_MIXTMPL.FITRES": table(
            DATA_COLUMNS + SIM_COLUMNS, mixtmpl,
            ["DATA with every other SIM_TEMPLATE_INDEX set to 101, so the CC and Ia",
             "prescales each act on half a sample of real Ia photometry.",
             "Built by build.py."]),
        "DATA_DUP.FITRES": table(
            DATA_COLUMNS + SIM_COLUMNS, data + data[:20],
            ["DATA with its first 20 rows repeated, for iflag_duplicate.",
             "Built by build.py."]),
        "DATA_SPLIT_A.FITRES": table(
            DATA_COLUMNS + SIM_COLUMNS + ["PROB_TEST"], split_a,
            ["First half of DATA plus a PROB_TEST column, for append_varname.",
             "Built by build.py."]),
        "DATA_SPLIT_B.FITRES": table(
            [c for c in DATA_COLUMNS + SIM_COLUMNS if c != "SNRSUM"], split_b,
            ["Second half of DATA without SNRSUM, for append_varname.",
             "Built by build.py."]),
        "CONTAM_DATA.FITRES": table(
            DATA_COLUMNS + ["PROBIA"], contam,
            ["Contaminated data for the per-event-pIa BEAMS tests.",
             f"{len(data)} Ia from ../bbc_des_data + {N_CONTAM} CC events from",
             f"../bbc_ccprior_sim (CID {CID_CONTAM_START}+), drawn from the",
             f"{len(pool)} inside BBC's |x1|<3, |c|<0.3 box. No SIM_TEMPLATE_INDEX",
             "column, so SALT2mu.exe reads the table as real data.",
             "Built by build.py."]),
        "cid_reject.list": "\n".join(cids[:100]) + "\n",
        "cid_select.list": "\n".join(cids[:1700]) + "\n",
    }, len(pool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report what would change without writing")
    args = ap.parse_args()

    tables, n_pool = build()
    print(f"CC pool inside the x1/c box: {n_pool}, taking {N_CONTAM} as contaminants")
    changed = False
    for name, text in tables.items():
        if name in BIASCOR_VARIANTS:
            continue
        path = HERE / name
        stored = path if path.exists() else path.with_name(path.name + ".gz")
        same = False
        if stored.exists():
            opener = gzip.open if stored.suffix == ".gz" else open
            with opener(stored, "rt") as stream:
                same = stream.read() == text
        print(f"  {name}: {'unchanged' if same else 'WOULD CHANGE'}")
        changed |= not same
        if not args.check:
            if len(text.encode()) >= 100_000:
                path.with_name(path.name + ".gz").write_bytes(
                    gzip.compress(text.encode(), compresslevel=6, mtime=0))
                path.unlink(missing_ok=True)
            else:
                path.write_text(text)
                path.with_name(path.name + ".gz").unlink(missing_ok=True)
    if args.check and changed:
        sys.exit(1)
    if not args.check:
        print("written")


if __name__ == "__main__":
    main()
