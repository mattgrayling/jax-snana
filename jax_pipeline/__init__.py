"""Analyses as graphs of stages, run in one process.

A stage runs one of this project's packages: a simulation, a light-curve fit,
a distance fit, a covariance or a cosmology fit. Each writes its usual
SNANA-format outputs under its own directory and returns what it reads back
from them, so you can inspect any step or load it back without rerunning it::

    from jax_pipeline import run, Sim, Fit

    sim = Sim("sim", "sim.input", overrides={"NGENTOT_LC": 5000})
    fit = Fit("fit", "fit.nml", datasets=[sim])
    results = run(fit, outdir="run")
    table = results["fit"]["sim"]["nominal"]

A stage with systematics returns a dict keyed by what was applied:
``"nominal"``, a systematic's label, or the ``(fitopt, muopt)`` pair where
both were. On disk each cell keeps SNANA's numbered directory name. Real data
enter through :class:`Data`, and fitted tables made elsewhere through
:class:`Fit` given ``fitres``. A stage of your own is a dataclass with a
name, ``inputs()`` and ``run()``, and is used like the built-in ones. Running the same graph
again loads every stage whose inputs have not changed.

Tables returned by Fit, Classify and BBC are
:class:`~jax_snlc_core.io.fitres_reader.FitresTable` objects, with the rows in ``df``
and the ``# KEY: value`` header as strings in ``meta``. Covariance returns its
Hubble diagram as a pandas DataFrame and its covariance entries as dictionaries
containing either factors or a dense inverse matrix. The distance fit's
alpha, for example, is ``float(m0dif.meta["FIT alpha0"].split()[0])``.

Systematics are handled here, not in the packages, as SNANA keeps FITOPTs out
of its executables. A fit's FITOPTs and a distance fit's MUOPTs are each a
dict from label to the input keys it overrides, and each stage runs once per
cell of the grid they form. Precision is set for the whole process: a script
that wants double calls
:func:`~jax_snlc_core.utils.precision.configure_precision` before
:func:`run`.

The stages report through Python's ``logging`` at INFO. The command-line
entry points configure logging, but a script must do so itself. To see the
banners, the CUTSTAT block and each stage's ``loaded`` or ``running`` line,
call ``logging.basicConfig(level=logging.INFO)`` before :func:`run`. Without
it only warnings are printed, and ``MANIFEST.yml`` in the output directory
records what was loaded and what was rerun.
"""

from jax_snlc_core.utils.precision import configure_precision

from .classify import Classify, Train
from .run import run
from .report import cosmology_summary
from .stages import BBC, Cosmology, Covariance, Data, Fit, Sim, Stage

__all__ = [
    "run", "cosmology_summary", "configure_precision", "Stage",
    "Data", "Sim", "Fit", "Train", "Classify", "BBC", "Covariance",
    "Cosmology",
]
