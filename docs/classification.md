# Classification

jax-SNANA provides support for photometric classification tools; it is possible
to use existing trained models and apply them directly to simulations or real
data, or alternatively to directly train a photometric classification
algorithm with jax-SNANA simulations. Support is currently restricted to
SuperNNova ([Möller & de Boissière 2020](https://arxiv.org/abs/1901.06384)),
although this code is modular and can be extended to support further
classification tools.

Install SuperNNova following its [installation
instructions](https://github.com/supernnova/SuperNNova#installation) to use
the examples below.

If JAX and SuperNNova will share a GPU, set
`XLA_PYTHON_CLIENT_PREALLOCATE=false` before starting Python so that JAX
leaves memory for PyTorch.

## Applying a trained model

The following Pippin example assumes a simulation or data task named
`DATA` and a corresponding light curve fitting task.
Add the trained classifier through `CLASSIFICATION`:

```yaml
CLASSIFICATION:
  SNN:
    CLASSIFIER: SuperNNovaClassifier
    MODE: predict
    MASK: DATA
    OPTS:
      MODEL: trained_model/model.pt
      PROB_COLUMN_NAME: PROB_SNN
      LIST_FILTERS: [DES-g, DES-r, DES-i, DES-z]
```

Run the configuration with `jax_pipeline analysis.yml`, as described in
{doc}`snana_and_pippin`. {doc}`pippin_settings` lists the classification
options.

`LIST_FILTERS` (`filters` in Python) lists the passband labels in the
photometry tables. These examples use the DES names `DES-g`, `DES-r`,
`DES-i` and `DES-z`.

Prediction requires the settings and normalisation files saved alongside
`model.pt`, so keep the trained-model directory intact. Its `cli_args.json`
records the training filter labels under `list_filters`; the supplied
photometry must use those labels.

In Python, `data` is a `Sim` or `Data` stage and `data_fit` its corresponding
`Fit` stage, constructed as shown in {doc}`building_pipelines`:

```python
from jax_pipeline import Classify, run
from jax_pipeline.supernnova import SuperNNova

supernnova = SuperNNova(filters=("DES-g", "DES-r", "DES-i", "DES-z"))
classified = Classify("classified", supernnova, fit=data_fit, photometry=data,
                      model="trained_model/model.pt")
results = run(classified, outdir="classification")
classified_fits = results["classified"]["nominal"].df
```

In Python, pass a `band_map` to `SuperNNova` if the photometry's filter
labels need renaming to match the trained model. For example, for a model
trained with single-letter band names:

```python
supernnova = SuperNNova(
    filters=("DES-g", "DES-r", "DES-i", "DES-z"),
    band_map={"DES-g": "g", "DES-r": "r", "DES-i": "i", "DES-z": "z"},
)
```

`PROB_SNN` contains the estimated type Ia probability. The stage adds it
to each fitted-results table, including every FITOPT, and writes the
probabilities to `predictions.csv`. For their use in standardisation,
see {ref}`bbc-contamination`.

## Training a model

You may want to train a photometric classification algorithm as part of an
analysis pipeline, for example for Ia/non-Ia classification in a cosmological
analysis. The following provides an example of multiple simulation inputs
being combined to train a classifier:

```yaml
SIM:
  TRAIN:
    IA:
      BASE: train_ia.input
    CC:
      BASE: train_cc.input
CLASSIFICATION:
  SNN_TRAIN:
    CLASSIFIER: SuperNNovaClassifier
    MODE: train
    MASK: TRAIN
    OPTS:
      LIST_FILTERS: [DES-g, DES-r, DES-i, DES-z]
```

Set `MODEL: SNN_TRAIN` on the prediction task to use the trained model.
In Python, the following trains a classifier and applies it to `data`:

```python
from jax_pipeline import Classify, Sim, Train, run
from jax_pipeline.supernnova import SuperNNova

supernnova = SuperNNova(filters=("DES-g", "DES-r", "DES-i", "DES-z"))
training = Sim("train_sim", ["train_ia.input", "train_cc.input"])
trained = Train("trained", supernnova, photometry=training)
classified = Classify("classified", supernnova, fit=data_fit, model=trained,
                      photometry=data)
results = run(classified, outdir="classification")
```

Saved classifiers are reused when the training configuration and inputs
are unchanged. See {ref}`pipeline-reruns` for rerunning the training stage.

The simulation's {ref}`README <simulation-readme>` contains
`GENTYPE_TO_NAME`, mapping simulated type codes to supernova types.
This supplies the training labels; use `sntypes` to provide a different
mapping.

Training and preprocessing options are passed as keyword arguments to
`SuperNNova`. See SuperNNova's [training
walkthrough](https://supernnova.readthedocs.io/latest/training/index.html)
and [settings
documentation](https://supernnova.readthedocs.io/latest/configuration/index.html)
for their meaning. The {py:class}`SuperNNova API
<jax_pipeline.supernnova.SuperNNova>` lists the arguments
accepted by jax-SNANA and their defaults.

## Importing existing probabilities

If you already have classification probabilities, supply a CSV with event
IDs in the first column, named `CID` or `SNID`, followed by the probability
columns, for example `PROB_IA`:

```python
from jax_pipeline import Classify

classified = Classify("classified", fit=data_fit, predictions="predictions.csv")
```

The probability columns, including any reported uncertainties, are joined
to every FITOPT table by event ID. An event with no prediction gets a
probability of zero.

To incorporate another classification tool, see {ref}`custom-classifier`.
