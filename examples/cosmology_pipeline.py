"""A cosmology analysis, from simulated light curves to a fitted cosmology.

Nothing here is specific to this repository. A pip-installed copy of the
package gives you `jax_pipeline`, and a script like this one belongs in your
own directory beside your own SNANA input files. The stages are the pieces of
the analysis; the order below is how you have chosen to connect them, and
changing the science means changing this file, not the package.

Every stage writes its usual SNANA-format outputs under `outdir`, and hands
the next stage what those files hold. Running the script again loads every
stage whose inputs have not changed rather than rerunning it, so a change to
the distance fit does not repeat the simulations.

The classifier is SuperNNova, which you install yourself, with the PyTorch
build for your machine, as the documentation's classification page describes.

The eight input files it expects under `inputs`:

    sim_data.input      the survey's Type Ia population
    sim_cc.input        the survey's core-collapse contaminants
    sim_train_ia.input  the Type Ia population the classifier learns from
    sim_train_cc.input  the core-collapse population it learns from
    sim_biascor.input   a large, same-survey Ia simulation for the bias correction
    fit.nml             the light-curve fit, for every simulation
    bbc.input           the distance fit (SALT2mu), naming the classifier's column
    wfit.yml            the cosmology fit
"""

from jax_pipeline import (BBC, Classify, Cosmology, Covariance, Fit,
                          Sim, Train, configure_precision, run)
from jax_pipeline.supernnova import SuperNNova

configure_precision(True)
inputs = "inputs"
outdir = "run"
events = 50_000

# The survey: Type Ia and core-collapse populations written into one data
# version, as a Pippin SIM group is. Then the larger Ia simulation whose
# fitted parameters the distance fit uses to correct its biases, that one
# again under a different intrinsic-scatter model for the systematic on the
# correction, and a core-collapse simulation for the contamination model. How
# many events and which seed are the input file's own keys; batch_size is
# what a large run is paced by, and a GPU wants tens of thousands of events
# at a time.
data = Sim("data_sim", [f"{inputs}/sim_data.input", f"{inputs}/sim_cc.input"],
           overrides=[{"NGENTOT_LC": events, "RANSEED": 1},
                      {"NGENTOT_LC": events, "RANSEED": 4}],
           batch_size=20_000)
biascor = Sim("biascor_sim", f"{inputs}/sim_biascor.input",
              overrides={"NGENTOT_LC": events * 5, "RANSEED": 2}, batch_size=20_000)
biascor_c11 = Sim("biascor_sim_c11", f"{inputs}/sim_biascor.input",
                  overrides={"NGENTOT_LC": events * 5, "RANSEED": 3,
                             "GENMAG_SMEAR_MODELNAME": "C11"}, batch_size=20_000)
ccprior = Sim("ccprior_sim", f"{inputs}/sim_cc.input",
              overrides={"NGENTOT_LC": events * 2, "RANSEED": 5}, batch_size=20_000)

# The calibration systematics: each a label and the fit keys it changes. The
# nominal fit is always run and is never listed.
calibration = {
    "zpt_g": {"MAGOBS_SHIFT_ZP": "g 0.01"},
    "zpt_r": {"MAGOBS_SHIFT_ZP": "r 0.01"},
    "filter_g": {"FILTER_LAMSHIFT": "g 10.0"},
}

# The same fit settings for every simulation. Only the data carries the
# calibration systematics: a bias correction is measured once, as in SNANA.
# The simulations share one fit, which fits them all at one set of compiled
# shapes.
data_fit = Fit("data_fit", f"{inputs}/fit.nml", datasets=[data],
               fitopts=calibration)
sim_fit = Fit("sim_fit", f"{inputs}/fit.nml",
              datasets=[biascor, biascor_c11, ccprior])
biascor_fit = sim_fit.version(biascor)
biascor_c11_fit = sim_fit.version(biascor_c11)
ccprior_fit = sim_fit.version(ccprior)

# Classification: SuperNNova learns from its own simulation, written from
# models other than the ones the bias correction uses, so that it is not
# tested on what it learned from (the DES 5-year analysis trained on such a
# sample). Every event of the data then gets a Type Ia probability from the
# light curves, and the fitted tables carry it on to the distance fit.
supernnova = SuperNNova(filters=("DES-g", "DES-i", "DES-r", "DES-z"))
training = Sim("train_sim", [f"{inputs}/sim_train_ia.input", f"{inputs}/sim_train_cc.input"],
               overrides=[{"NGENTOT_LC": events, "RANSEED": 6},
                          {"NGENTOT_LC": events, "RANSEED": 7}],
               batch_size=20_000)
trained = Train("trained", supernnova, photometry=training)
classified = Classify("classified", supernnova, fit=data_fit, model=trained,
                      photometry=data)

# The distance fit, once per (fit systematic x distance systematic), with the
# contamination handled by the classifier's column and the core-collapse
# prior. A distance systematic changes SALT2mu keys; giving simfile_biascor a
# stage instead of a file makes it consume that stage, as a MUOPT naming
# another simulation does in Pippin.
distances = BBC(
    "distances", data=classified, biascor=biascor_fit, ccprior=ccprior_fit,
    input_file=f"{inputs}/bbc.input",
    muopts={"tight_chi2": {"chi2max": 16},
            "fixed_alpha_beta": {"u1": 0, "u2": 0},
            "c11_scatter": {"simfile_biascor": biascor_c11_fit}},
)

# Further covariance matrices beyond ALL, which is always built first from
# every systematic, in create_covariance's own form: "[label] [fitopt
# filter,muopt filter]", where "" takes everything, "=X" is exact, "+X"
# contains and "-X" excludes. DEFAULT is the nominal, so an exact match on it
# leaves the statistical matrix alone.
covariance = Covariance(
    "covariance", distances=distances,
    covopts=["[NOSYS] [=DEFAULT,=DEFAULT]", "[CALIB] [+zpt,]"],
)

# One cosmology fit over every covariance matrix, in one call.
cosmology = Cosmology("cosmology", covariance=covariance,
                      input_file=f"{inputs}/wfit.yml",
                      covopts=("ALL", "NOSYS", "CALIB"))

results = run(cosmology, outdir=outdir)

# The results are what the stages wrote, so the answer is read off the
# cosmology fit's own summary.
for covopt, fit in results["cosmology"].items():
    for name, value in fit["point_estimate"].items():
        spread = fit["parameters"][name]["std"]
        print(f"{covopt:>6s} {name:>8s} = {value:8.4f} +- {spread:.4f}")
