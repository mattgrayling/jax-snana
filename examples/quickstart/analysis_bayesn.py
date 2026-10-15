"""Simulate and fit a small BayeSN sample using the bundled inputs.

Run from the repository root with SNDATA_ROOT set as described under
"Preparing the examples" in the Installation guide.
"""

import argparse

from jax_pipeline import Fit, Sim, run

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--outdir", default="run_bayesn")
args = parser.parse_args()

survey = Sim("sim", "examples/quickstart/sim_bayesn.input")
fitted = Fit("fit", "examples/quickstart/fit_bayesn.nml", datasets=[survey])
results = run(fitted, outdir=args.outdir)
print(results["fit"]["sim"]["nominal"].df[
    ["CID", "zHD", "THETA", "AV", "MU_LCFIT", "MUERR_LCFIT"]
])
