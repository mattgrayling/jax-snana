from jax_pipeline import BBC, Cosmology, Covariance, Fit, Sim, run

inputs = "examples/quickstart"

survey = Sim("sim", f"{inputs}/sim.input",
             overrides={"NGENTOT_LC": 600, "RANSEED": 1})

light_curves = Fit("fit", f"{inputs}/fit.nml", datasets=[survey],
                   fitopts={"zpt_g": {"MAGOBS_SHIFT_ZP": "g 0.01"}})

distances = BBC("distances", data=light_curves,
                overrides={"nzbin": 5, "zmin": 0.05, "zmax": 1.2,
                           "u1": 1, "u2": 1})

covariance = Covariance("covariance", distances=distances,
                        overrides={"COSMOLOGY": {"OMEGA_MATTER": 0.315,
                                                 "OMEGA_LAMBDA": 0.685}})

cosmology = Cosmology("cosmology", covariance=covariance,
                      overrides={"model": "lcdm", "ompri": 0.315, "dompri": 0.007})

results = run(cosmology, outdir="run")

for covopt, fitted in results["cosmology"].items():
    for name, value in fitted["point_estimate"].items():
        spread = fitted["parameters"][name]["std"]
        print(f"{covopt:>6s} {name:>8s} = {value:8.4f} +- {spread:.4f}")
