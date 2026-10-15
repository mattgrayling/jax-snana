# Bundled data

jax-SNANA includes a small set of models and filter transmissions so that
users can fit light curves in Python without first obtaining these files
separately. The repository also includes supporting data for the worked
examples and tests. For survey analyses, users should supply their own
survey inputs and calibration files, either from `SNDATA_ROOT` or another
directory.

## Models

The following models are included in the installed package:

| Model | Name for Python fits | Source |
|---|---|---|
| BayeSN G26 | `BAYESN.G26` | BayeSN model distribution |
| BayeSN W22 | `BAYESN.W22` | SNANA supporting-data distribution |
| SALT3.Dovekie | `SALT3.DOVEKIE` | SNANA supporting-data distribution |
| SALT3.K21 | `SALT3.K21` | SNANA supporting-data distribution |

For example, `LightCurveFitter("BAYESN.G26")` loads the included G26 model.
{doc}`models` describes the models and how to select another model by
supplying its directory. The relevant publications are listed in
{doc}`citing`.

(bundled-filters)=
## Filters and calibration

SNANA makes a deliberate choice not to build any filter transmissions into
the code. This is because filter transmissions are often remeasured and
iterate over time, so it is important in any cosmology analysis that the
filter files being used are explicitly kept track of. This is a different
approach to Python packages such as
[sncosmo](https://sncosmo.readthedocs.io/en/stable/bandpass-list.html), which
downloads filter responses. This latter approach is convenient as it allows
users to fit light curves without providing their own filter response files,
though it may lead to issues if filter transmissions are not updated centrally.
jax-snana makes a compromise; it does include built-in filters but will only
support these through the Python interfaces in scripts, intended for quick
applications that do not need the same level of precision as a full cosmology
analysis. When using the command-line interface for larger-scale jobs,
built-in filters cannot be used and the user must provide a KCOR file.

The filter set follows BayeSN's main branch, with transmission curves,
names and calibration information taken from its calibration branch.
The Dovekie cross-calibration corrections are applied by default.

The catalogue includes Pan-STARRS 1, DES, LSST, ZTF, ATLAS, CSP, HST/WFC3,
Swift/UVOT and other optical and near-infrared filters. Names follow the
BayeSN convention, for example `g_PS1`, `g_DES` and `p48g`. To list the
available names for a fitter, use
`fitter.calibration.get_filter_names()`.
{ref}`fit-filters` explains how to associate these names with the bands in
your light curves and how to supply your own calibration.

## Example and test data

The example input files and survey data used in {doc}`getting_started`
and {doc}`cosmology_analysis` are included in the repository, rather than
installed by pip. To obtain them, clone the repository as described in
{ref}`installation-examples`.

The examples use a small SIMLIB representative of DES, included
for testing, and a DES *griz* calibration file copied from the SNANA
DES-SN5YR supporting data. The repository also supplies a minimal
`SNDATA_ROOT`, including `SURVEY.DEF` and small synthetic dust maps, so the
examples can run without the full supporting-data distribution.

Further test data are included under `tests/fixtures/data`. For survey
work, the full SNANA supporting-data distribution is available on
[Zenodo](https://zenodo.org/records/12655677/latest).
