# Cosmology settings

This page lists the settings for cosmology fitting with `wfit`.
{doc}`cosmology` gives worked examples.

YAML and Python use the names in the tables below. Command-line options
replace underscores with hyphens, such as `--zmin 0.02`. `--config` selects
a YAML file; explicit command-line values override it. SNANA's single-dash
flags, such as `-lcdm` and `-mucovsys_file`, are also supported.

```{eval-rst}
.. configtable:: jax_wfit.config.WFITConfig
   :omit-sections: Fixed
```

For explicitly rejected SNANA keys, see {ref}`unsupported-cosmology-keys`.

(cosmology-prior-settings)=
## External constraints

The default matter-density prior is centred on 0.315 with width 99, so it is
uninformative over the fitted range. The CMB shift-parameter measurement is
1.710 ± 0.019 from [Komatsu et al. (2009)](https://lambda.gsfc.nasa.gov/product/wmap/dr3/pub_papers/fiveyear/cosmology/wmap_5yr_cosmo.pdf),
evaluated at `cmb_redshift=1089`. YAML `cmb: true` leaves
the matter-density prior as configured. The SNANA command-line flag `-cmb`
also enables that prior and sets its width to 0.9. A BAO constraint replaces
the matter-density prior, whether selected through YAML or the command line.

BAO predictions use the fixed Hubble constant and sound horizon set by
`bao_h0` and `bao_rd`; their uncertainties enter the BAO covariance through
`bao_h0_sigma` and `bao_rd_sigma`. The selected sample's files under
`$SNDATA_ROOT/models/BAO` supply the distance measurements and covariance:
[DESI Collaboration (2024)](https://arxiv.org/abs/2404.03002) for `DESI_2024`
and [Alam et al. (2021)](https://arxiv.org/abs/2007.08991) for `SDSS4_2020`.

`perturb_priors` adds random Gaussian shifts to an informative matter-density
prior and to a simulation-centred CMB constraint (`cmb_sim`), with the seed
set by `prior_perturbation_seed`. SNANA's
`-ranseed_Rcmb N` also enables perturbation when `N` is positive; setting
`--prior-perturbation-seed N` alone does not enable it.

(cosmology-covariance-settings)=
## Covariance cuts and adjustments

Covariance inputs match the full Hubble diagram before redshift cuts.
`total_inverse` does not support redshift cuts or uncertainty adjustments.
`factorized` supports redshift and event-count cuts and additional scatter
through `snrms`, but not replacement measurement errors or interpolation
between diagrams with `hdibc`.

(cosmology-blinding-settings)=
## Blinding metadata

With `blind_mode: auto` (`-blind_auto`), `ISDATA_REAL > 0` enables blinding;
values at or below zero, including -1, are treated as simulations. A missing
`ISDATA_REAL` header stops the fit. `blind_seed` sets the reproducible
parameter shifts and is omitted from blinded output, together with the
recovered-bias summary.
