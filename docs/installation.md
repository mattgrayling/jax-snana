# Installation

jax-SNANA can be installed via pip or directly from the source code.
For GPU use, first install a GPU build of JAX as described under
{ref}`installation-gpu`.

## Installing with pip

You can install the latest release with pip:

```bash
pip install jax-snana
```

jax-snana requires Python 3.12 or newer and JAX 0.10.1 or newer.

## Installing from source

If you want the latest development version or want to contribute to the code you
can instead install from a clone of the repository:

```bash
git clone https://github.com/mattgrayling/jax-snana.git
cd jax-snana
pip install -e .
```

## Optional extras

- `test`: pytest and its plugins, plus the documentation dependencies, since
  the test suite includes a documentation build.
- `docs`: Sphinx, MyST-Parser and the Read the Docs theme, for building these
  pages on their own.
- `redback`: `redback-jax`, needed only for the `redback-jax` models described
  in {doc}`models`.

You can install any combination of these, with pip or from source:

```bash
pip install 'jax-snana[redback]'
pip install -e '.[test,redback]'
```

(installation-gpu)=
## GPU support

If you want to use jax-SNANA on a GPU, note that the version installed with
pip is by default CPU-only. You must therefore install a GPU build of JAX
prior to installing jax-SNANA. For this, please follow the [JAX
installation guide](https://docs.jax.dev/en/latest/installation.html). You
can verify whether your GPU is recognised by JAX with the following command:

```python
import jax
print(jax.devices())
```

## Supporting data

SNANA is built around the `SNDATA_ROOT` directory (accessible on
[Zenodo](https://zenodo.org/records/12655677/latest)), which contains a variety of supporting input files including
survey definitions, filter transmission and models, and does not build
these into the package itself. For the most part, jax-SNANA follows this
approach. However, jax-SNANA does build-in some supporting data. For
example, a number of small files are included to support the testing suite
and allow it to run even without access to `SNDATA_ROOT`. A number of small
files are also included to support the examples in {doc}`getting_started`.
Finally, jax-SNANA does build in a small variety of model files and filter
transmissions to support light curve fitting within Python scripts at a
minimal start-up cost to the user. {doc}`bundled_data` describes the included
files and their provenance. The built-in filters are intended for exploratory
fits in Python. Command-line analyses require a calibration file.

(installation-examples)=
## Preparing the examples

The examples in {doc}`getting_started` and {doc}`cosmology_analysis` use
input files and supporting data included in the repository. If you installed
with pip, download the examples by cloning the repository:

```bash
git clone https://github.com/mattgrayling/jax-snana.git
cd jax-snana
```

The example inputs give paths to the model, a {ref}`SIMLIB <format-simlib>`
containing the observing times and conditions, and a filter calibration file
({ref}`kcor <format-kcor>`). These paths are relative to the repository root.
Run the examples from there. The repository also includes a minimal SNANA
data tree containing `SURVEY.DEF`, small synthetic dust maps for the examples
and the BAO prior files:

```bash
export SNDATA_ROOT=$PWD/tests/fixtures/data/sndata_root
```

For survey analyses, set `SNDATA_ROOT` to your supporting-data directory.
The full data tree is available from the Zenodo link above:

```bash
export SNDATA_ROOT=/path/to/SNDATA_ROOT
```
