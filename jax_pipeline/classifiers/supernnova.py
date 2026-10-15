"""SuperNNova (Möller & de Boissière 2020), a recurrent neural network that
classifies light curves, run as Pippin's SuperNNovaClassifier runs it."""

import json
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from astropy.io import fits

from jax_snlc_core.io.snana_table import write_varnames_table
from jax_snlc_core.utils.error_handling import SNANAError

# SNANA's photometric-only type is 100 above the spectroscopic one
_OFFSET_TYPE_PHOT = 100

# the HEAD's host redshifts SuperNNova can classify with
_HOST_REDSHIFTS = ("HOSTGAL_PHOTOZ", "HOSTGAL_PHOTOZ_ERR",
                   "HOSTGAL_SPECZ", "HOSTGAL_SPECZ_ERR")


@dataclass(frozen=True)
class SuperNNova:
    """Train or apply a SuperNNova light-curve classifier.

    ``filters`` lists the bands as named in the PHOT files. ``sntypes``
    maps type codes to names, using ``Ia`` for type Ia supernovae; if omitted,
    training reads this mapping from the simulation's README.

    ``variant`` selects the network type. ``redshift`` selects its redshift
    input, and ``norm`` its flux normalisation. ``num_layers`` and
    ``hidden_dim`` set the network dimensions; ``cyclic`` selects cyclic
    training. ``seed``, ``batch_size`` and ``nb_epoch`` control training,
    with SuperNNova's epoch default used when ``nb_epoch`` is None.

    ``phot_reject_bits`` identifies flagged epochs to exclude, and
    ``redshift_label`` names the redshift column. ``band_map`` renames bands
    for prediction, for example ``{'DES-g': 'g'}``, to match those used in
    training. The trained model requires its accompanying ``cli_args.json``
    and normalisation files.

    If supplied, the fitted table identifies the successfully fitted events
    for SuperNNova's training-sample split. See :doc:`classification` for
    installation, examples and links to SuperNNova's settings.
    """

    filters: tuple[str, ...]
    column: str = "PROB_SNN"
    variant: str = "vanilla"
    redshift: str = "zspe"
    norm: str = "cosmo"
    cyclic: bool = True
    seed: int = 0
    batch_size: int = 128
    num_layers: int = 2
    hidden_dim: int = 32
    nb_epoch: int | None = None
    sntypes: Mapping[int, str] | None = None
    use_cuda: bool = False
    phot_reject_bits: tuple[int, ...] = (8, 16, 32, 64, 128, 256, 512)
    redshift_label: str = "REDSHIFT_FINAL"
    band_map: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.variant not in ("vanilla", "variational", "bayesian"):
            raise SNANAError(f"SuperNNova variant must be vanilla, variational "
                             f"or bayesian, not {self.variant}",
                             function_name="SuperNNova")

    def train(self, fit, photometry, outdir) -> Path:
        """Builds SuperNNova's dataset from ``photometry`` and trains on it;
        returns the model file, beside which SuperNNova keeps the settings
        and normalisation a prediction reads back."""
        sntypes = self._sntypes(photometry)
        dump = Path(outdir) / "dump"
        self._snn("make_data", self._data_args(fit, photometry, dump, sntypes),
                  Path(outdir))
        self._snn("train_rnn", self._model_args(dump, sntypes), Path(outdir))
        models = sorted((dump / "models").glob("*/*.pt"))
        if len(models) != 1:
            raise SNANAError(f"SuperNNova left {len(models)} model files under "
                             f"{dump / 'models'}, not one", function_name="SuperNNova")
        return models[0]

    def predict(self, fit, photometry, model) -> pd.Series | pd.DataFrame:
        """The type Ia probability of every event in ``photometry``, from the
        model a training wrote: SuperNNova's ``all_class0``, or for a
        Bayesian variant its median over the model's samples with their
        standard deviation as ``<column>_ERR``. Probabilities are returned
        under the configured ``column`` name, indexed by event identifier."""
        if photometry is None or model is None:
            raise SNANAError("SuperNNova classifies a version's light curves with "
                             "a trained model; give the stage both",
                             function_name="SuperNNova")
        light_curves = self._light_curves(photometry)
        # SuperNNova gives every event NaN, with no error, when the bands
        # differ from those the model was trained on
        trained_on = json.loads((Path(model).parent / "cli_args.json").read_text())["list_filters"]
        bands = sorted(set(light_curves["FLT"]))
        if not set(trained_on) <= set(bands):
            raise SNANAError(f"the model {model} was trained on bands {trained_on}, but "
                             f"the light curves of {photometry} have {bands}; give "
                             f"band_map to rename them", function_name="SuperNNova")
        # an optional dependency, imported only by a run that classifies with it
        from supernnova.validation.validate_onthefly import classify_lcs

        snids, preds = classify_lcs(light_curves, str(model),
                                    "cuda" if self.use_cuda else "cpu")
        unclassified = int(np.isnan(preds).any(axis=(1, 2)).sum())
        if unclassified:
            raise SNANAError(f"SuperNNova gave {unclassified} of {len(snids)} events in "
                             f"{photometry} no probability", function_name="SuperNNova")
        index = pd.Index(snids, dtype=str)
        samples = preds[:, :, 0]
        if self.variant == "vanilla":
            return pd.Series(samples[:, 0], index=index, name=self.column)
        # SuperNNova's aggregation: the median and pandas' sample std per event
        return pd.DataFrame({self.column: np.median(samples, axis=1),
                             f"{self.column}_ERR": np.std(samples, axis=1, ddof=1)},
                            index=index)

    def _light_curves(self, photometry) -> pd.DataFrame:
        """Every event's epochs in ``filters`` with its host redshifts, as
        SuperNNova's dataset step prepares them (process_single_FITS): the
        epochs with any of ``phot_reject_bits`` dropped, and
        ``redshift_label`` in place of the spectroscopic host redshift. The
        bands are then renamed by ``band_map``."""
        reject = sum(self.phot_reject_bits)
        heads = sorted(Path(photometry).glob("**/*HEAD.FITS*"))
        if not heads:
            raise SNANAError(f"{photometry} holds no HEAD file", function_name="SuperNNova")
        tables = []
        for head_path in heads:
            head = fits.getdata(head_path, 1)
            phot = fits.getdata(head_path.with_name(head_path.name.replace("HEAD", "PHOT")), 1)
            first, last = head["PTROBS_MIN"] - 1, head["PTROBS_MAX"]
            rows = np.concatenate([np.arange(a, b) for a, b in zip(first, last)])
            event = np.repeat(np.arange(len(head)), last - first)
            # SuperNNova keeps those the HEAD has
            hosts = {name: np.asarray(head[name], dtype=np.float64)
                     for name in _HOST_REDSHIFTS if name in head.columns.names}
            if self.redshift_label != "none":
                hosts["HOSTGAL_SPECZ"] = np.asarray(head[self.redshift_label], dtype=np.float64)
                hosts["HOSTGAL_SPECZ_ERR"] = np.asarray(head[f"{self.redshift_label}_ERR"],
                                                        dtype=np.float64)
            table = pd.DataFrame({
                "SNID": np.char.strip(np.asarray(head["SNID"], dtype=str))[event],
                "MJD": np.asarray(phot["MJD"], dtype=np.float64)[rows],
                "FLUXCAL": np.asarray(phot["FLUXCAL"], dtype=np.float64)[rows],
                "FLUXCALERR": np.asarray(phot["FLUXCALERR"], dtype=np.float64)[rows],
                "FLT": np.char.strip(np.asarray(phot["BAND"], dtype=str))[rows],
                "PEAKMJD": np.asarray(head["PEAKMJD"], dtype=np.float64)[event],
                **{name: values[event] for name, values in hosts.items()}})
            keep = (((np.asarray(phot["PHOTFLAG"])[rows] & reject) == 0)
                    & table["FLT"].isin(self.filters).to_numpy())
            tables.append(table[keep])
        light_curves = pd.concat(tables, ignore_index=True)
        light_curves["FLT"] = light_curves["FLT"].replace(dict(self.band_map))
        return light_curves

    def _data_args(self, fit, photometry, dump: Path, sntypes) -> list[str]:
        args = ["--dump_dir", str(dump), "--raw_dir", str(photometry),
                "--list_filters", *self.filters,
                "--phot_reject", "PHOTFLAG",
                "--phot_reject_list", *map(str, self.phot_reject_bits),
                "--redshift_label", self.redshift_label,
                "--redshift", self.redshift, "--norm", self.norm,
                "--seed", str(self.seed), "--sntypes", json.dumps(sntypes)]
        if fit is not None:
            # SuperNNova reads FITOPT000.FITRES from --fits_dir
            fits_dir = dump.parent / "fits"
            fits_dir.mkdir(parents=True, exist_ok=True)
            write_varnames_table(fits_dir / "FITOPT000.FITRES", fit.varnames,
                                 {n: fit.df[n].to_numpy() for n in fit.varnames}, {})
            args += ["--fits_dir", str(fits_dir)]
        return args + self._device()

    def _model_args(self, dump: Path, sntypes) -> list[str]:
        args = ["--dump_dir", str(dump), "--list_filters", *self.filters,
                "--batch_size", str(self.batch_size),
                "--num_layers", str(self.num_layers),
                "--hidden_dim", str(self.hidden_dim),
                "--model", self.variant,
                "--redshift", self.redshift, "--norm", self.norm,
                "--seed", str(self.seed), "--sntypes", json.dumps(sntypes)]
        if self.variant == "bayesian":
            args += ["--num_inference_samples", "20"]
        if self.cyclic and self.variant != "bayesian":
            args.append("--cyclic")
        if self.nb_epoch is not None:
            args += ["--nb_epoch", str(self.nb_epoch)]
        return args + self._device()

    def _sntypes(self, photometry) -> dict[str, str]:
        """``sntypes``, or the version README's types as SuperNNova reads
        them: ``Ia`` for a Type Ia line, the type name for any other."""
        if self.sntypes is not None:
            return {str(k): v for k, v in self.sntypes.items()}
        readme = Path(photometry) / f"{Path(photometry).name}.README"
        if not readme.is_file():
            raise SNANAError(f"{photometry} has no {readme.name} to name its GENTYPEs, "
                             f"so SuperNNova cannot tell the Type Ia events to train "
                             f"on; give sntypes", function_name="SuperNNova")
        lines = yaml.safe_load(readme.read_text())["DOCUMENTATION"]["GENTYPE_TO_NAME"]
        sntypes = {}
        for gentype, line in lines.items():
            kind, name = line.split()[:2]
            for key in (gentype, gentype + _OFFSET_TYPE_PHOT):
                sntypes[str(key)] = "Ia" if kind == "Ia" else name
        return sntypes

    def _device(self) -> list[str]:
        return ["--use_cuda"] if self.use_cuda else []

    def _snn(self, action: str, args: list[str], logdir: Path) -> None:
        """One ``snn`` command, its output kept in ``snn_<action>.log``."""
        log = logdir / f"snn_{action}.log"
        with log.open("w") as stream:
            done = subprocess.run([sys.executable, "-m", "cli.run", action, *args],
                                  stdout=stream, stderr=subprocess.STDOUT)
        if done.returncode != 0:
            tail = "\n".join(log.read_text().splitlines()[-20:])
            raise SNANAError(f"snn {action} failed:\n{tail}", function_name="SuperNNova")

