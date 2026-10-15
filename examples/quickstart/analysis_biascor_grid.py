"""Generate, fit and apply a multidimensional SALT bias-correction sample.

Run from the repository root with SNDATA_ROOT set as in Getting started.
The coarse cells are for demonstrating the workflow, not survey analysis.
"""

import argparse

from jax_pipeline import BBC, Fit, Sim, run

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--outdir", default="run_biascor_grid")
args = parser.parse_args()
inputs = "examples/quickstart"

survey = Sim("sim", f"{inputs}/sim.input",
             overrides={"NGENTOT_LC": 10000, "RANSEED": 1,
                        "GENMAG_SMEAR": 0.10})
biascor = Sim("biascor", f"{inputs}/sim_biascor_grid.input")
fit = Fit("fit", f"{inputs}/fit.nml", datasets=[survey, biascor],
          overrides={"CUTWIN_REDSHIFT": "0.05 1.3"})
distances = BBC("distances", data=fit.version(survey), biascor=fit.version(biascor),
                input_file=f"{inputs}/bbc_biascor_grid.input")
results = run(distances, outdir=args.outdir)
print(f"BBC outputs: {args.outdir}/distances")
