"""Run a small bias-corrected analysis using the bundled survey inputs.

Run from the repository root with SNDATA_ROOT set as in Getting started.
The redshift-only correction and fixed standardisation/scatter settings keep
this example small; they are described in the cosmology-analysis guide.
"""

import argparse

from jax_pipeline import BBC, Cosmology, Covariance, Fit, Sim, run

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--outdir", default="run_biascor")
args = parser.parse_args()
inputs = "examples/quickstart"

survey = Sim("sim", f"{inputs}/sim.input",
             overrides={"NGENTOT_LC": 600, "RANSEED": 1,
                        "GENMAG_SMEAR": 0.10})
biascor = Sim("biascor", f"{inputs}/sim.input",
              overrides={"NGENTOT_LC": 6000, "RANSEED": 2,
                         "GENMAG_SMEAR": 0.10})
fit = Fit("fit", f"{inputs}/fit.nml", datasets=[survey, biascor])
distances = BBC("distances", data=fit.version(survey), biascor=fit.version(biascor),
                input_file=f"{inputs}/bbc_biascor.input")
covariance = Covariance("covariance", distances=distances)
cosmology = Cosmology("cosmology", covariance=covariance,
                      overrides={"model": "lcdm", "method": "grid"})
results = run(cosmology, outdir=args.outdir)
print(results["cosmology"]["ALL"]["point_estimate"])
