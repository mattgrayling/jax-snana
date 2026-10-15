"""
Fit result data structures.

Provides dataclasses for storing and manipulating fit results,
including parameter values, uncertainties, and fit quality metrics.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import jax.numpy as jnp
import pandas as pd
from astropy.table import MaskedColumn, Table

from jax_snlc_core.utils.error_handling import SNANAError
from .batch import FIT_REASON_DESCRIPTIONS


def prob_ndof(ndof, prior_chi2):
    """The NDOF a fit probability is taken at: the fit's ``ndof`` plus one
    where the prior's chi2 exceeds 0.01, as the chi2 it is taken of includes
    the prior (NDOF_STORE(1), snana.F90:28092-28108)."""
    return ndof + (prior_chi2 > 0.01)


@dataclass
class BatchFitResult:
    """Fit results for a batch of supernovae; every array is indexed by
    supernova (first dimension n_sne)."""
    snids: np.ndarray

    params: Dict[str, jnp.ndarray]           # each [n_sne]
    param_errors: Dict[str, jnp.ndarray]     # each [n_sne]
    param_covariances: jnp.ndarray           # [n_sne, n_params, n_params]
    cov_output_params: List[str]             # names covered by param_covariances

    chi_squared: jnp.ndarray  # the minimised chi2, prior included (SNANA's FITCHI2)
    # epochs fitted, less the parameters the data constrain (not the held
    # ones, nor those a prior constrains; zPHOT counts): SNANA's NDOF
    ndof: jnp.ndarray
    prior_chi2: jnp.ndarray   # [n_sne]; included in chi_squared

    observation_mask: jnp.ndarray  # epochs the fit used [n_sne, max_obs]
    hessian_cond: jnp.ndarray      # [n_sne]
    hessian_positive: jnp.ndarray  # [n_sne], before covariance repair
    edm: jnp.ndarray               # estimated distance to minimum [n_sne]

    # zPRIOR and zPRIORERR (snlc_fit.F90:8561); None for spec-z
    z_prior: Optional[Tuple[jnp.ndarray, jnp.ndarray]]

    # {FITWIN_ suffix: [n_sne] ok mask}
    iter1_verdicts: Dict[str, jnp.ndarray]
    outcomes: Optional[pd.DataFrame] = None
    context: Optional[list] = None
    model_columns: Optional[Dict[str, np.ndarray]] = None

    # Positive lower/upper profile distances and per-side status, [n_sne, 2].
    minos_errors: Optional[Dict[str, jnp.ndarray]] = None
    minos_status: Optional[Dict[str, jnp.ndarray]] = None

    # SNANA error sources, including step-size replacements (code 6).
    error_flags: Optional[Dict[str, jnp.ndarray]] = None

    @property
    def n_sne(self):
        return len(self.snids)

    @property
    def fitprob_ndof(self):
        """The NDOF FITPROB is taken at (:func:`prob_ndof`)."""
        return prob_ndof(self.ndof, self.prior_chi2)

    @property
    def param_names(self):
        """List of fitted parameter names."""
        return list(self.params.keys())

    def take(self, indices) -> 'BatchFitResult':
        """Return the objects selected by ``indices``, in that order."""
        indices = np.asarray(indices)
        if indices.ndim != 1 or (indices.size and not np.issubdtype(indices.dtype, np.integer)):
            raise SNANAError('indices must be a one-dimensional sequence of integers')
        indices = np.arange(self.n_sne)[indices.astype(np.intp)]
        outcomes = None
        if self.outcomes is not None:
            rows = self.outcomes.set_index('fit_index')
            outcomes = rows.loc[indices].reset_index()
            outcomes['fit_index'] = np.arange(len(indices))
        return BatchFitResult(
            snids=self.snids[indices],
            params={k: v[indices] for k, v in self.params.items()},
            param_errors={k: v[indices] for k, v in self.param_errors.items()},
            minos_errors=(None if self.minos_errors is None else
                          {k: v[indices] for k, v in self.minos_errors.items()}),
            minos_status=(None if self.minos_status is None else
                          {k: v[indices] for k, v in self.minos_status.items()}),
            error_flags=(None if self.error_flags is None else
                         {k: v[indices] for k, v in self.error_flags.items()}),
            param_covariances=self.param_covariances[indices],
            cov_output_params=self.cov_output_params,
            chi_squared=self.chi_squared[indices],
            ndof=self.ndof[indices],
            prior_chi2=self.prior_chi2[indices],
            observation_mask=self.observation_mask[indices],
            hessian_cond=self.hessian_cond[indices],
            edm=self.edm[indices],
            hessian_positive=self.hessian_positive[indices],
            z_prior=(None if self.z_prior is None
                     else tuple(a[indices] for a in self.z_prior)),
            iter1_verdicts={k: v[indices]
                            for k, v in self.iter1_verdicts.items()},
            outcomes=outcomes,
            context=None if self.context is None else [self.context[i] for i in indices],
            model_columns=(None if self.model_columns is None else
                           {name: values[indices] for name, values in self.model_columns.items()}),
        )

    def filter_by_mask(self, mask) -> 'BatchFitResult':
        """Keep only the SNe where ``mask`` is True."""
        mask = np.asarray(mask)
        if mask.shape != (self.n_sne,):
            raise SNANAError(
                f"mask of {mask.shape} does not match n_sne={self.n_sne}; "
                f"a shorter mask would silently drop the tail of the sample",
                function_name="filter_by_mask")
        return self.take(np.flatnonzero(mask))

    @classmethod
    def concatenate(cls, results: list) -> 'BatchFitResult':
        """One result holding every SN in ``results``, in list order."""
        def cat(name):
            return jnp.concatenate([getattr(r, name) for r in results])

        if not results:
            raise SNANAError('Cannot concatenate an empty result list')
        if any(r.param_names != results[0].param_names for r in results):
            raise SNANAError('Cannot concatenate results with different parameter vectors')
        for field in ('outcomes', 'model_columns', 'minos_errors', 'minos_status',
                      'error_flags'):
            present = [getattr(result, field) is not None for result in results]
            if any(present) and not all(present):
                raise SNANAError(f'Cannot mix results with and without {field}')
        fitted = [r for r in results if r.n_sne]
        verdict_names = set(fitted[0].iter1_verdicts) if fitted else set()
        if any(set(r.iter1_verdicts) != verdict_names for r in fitted):
            raise SNANAError('Cannot concatenate fits with different first-round verdicts')
        contexts = [entry for r in results for entry in (r.context or [])]
        if contexts:
            first = contexts[0][0]
            if any(c['calibration'] != first['calibration'] or c['config'].model != first['config'].model
                   for c, _ in contexts):
                raise SNANAError('Cannot concatenate results with different models or calibration')
            if any(r.context is None and r.n_sne for r in results):
                raise SNANAError('Cannot mix results with and without prediction context')
        outcomes, offset, input_offset = [], 0, 0
        for r in results:
            if r.outcomes is not None:
                frame = r.outcomes.copy()
                frame.loc[frame.fit_index >= 0, 'fit_index'] += offset
                frame['input_index'] += input_offset
                outcomes.append(frame)
                input_offset += len(frame)
            offset += r.n_sne
        width = max(r.observation_mask.shape[1] for r in results)
        masks = [jnp.pad(r.observation_mask, ((0, 0), (0, width-r.observation_mask.shape[1]))) for r in results]
        z_prior = results[0].z_prior
        return cls(
            snids=np.concatenate([r.snids for r in results]),
            params={name: jnp.concatenate([r.params[name] for r in results])
                    for name in results[0].param_names},
            param_errors={
                name: jnp.concatenate([r.param_errors[name] for r in results])
                for name in results[0].param_names},
            minos_errors=(None if results[0].minos_errors is None else
                {name: jnp.concatenate([r.minos_errors[name] for r in results])
                 for name in results[0].param_names}),
            minos_status=(None if results[0].minos_status is None else
                {name: jnp.concatenate([r.minos_status[name] for r in results])
                 for name in results[0].param_names}),
            error_flags=(None if results[0].error_flags is None else
                {name: jnp.concatenate([r.error_flags[name] for r in results])
                 for name in results[0].param_names}),
            param_covariances=cat('param_covariances'),
            cov_output_params=results[0].cov_output_params,
            chi_squared=cat('chi_squared'),
            ndof=cat('ndof'),
            prior_chi2=cat('prior_chi2'),
            observation_mask=jnp.concatenate(masks),
            hessian_cond=cat('hessian_cond'),
            edm=cat('edm'),
            hessian_positive=cat('hessian_positive'),
            z_prior=(None if z_prior is None else tuple(
                map(jnp.concatenate, zip(*(r.z_prior for r in results))))),
            iter1_verdicts={
                name: jnp.concatenate([r.iter1_verdicts[name]
                                       for r in fitted])
                for name in sorted(verdict_names)},
            outcomes=pd.concat(outcomes, ignore_index=True) if outcomes else None,
            context=contexts or None,
            model_columns=(None if results[0].model_columns is None else {
                name: np.concatenate([result.model_columns[name] for result in results])
                for name in results[0].model_columns}),
        )

    def to_table(self):
        """An Astropy table in input order, including objects without a fit."""
        outcomes = self.outcomes
        if outcomes is None:
            outcomes = pd.DataFrame({'snid': self.snids, 'fit_index': np.arange(self.n_sne)})
        table = Table.from_pandas(outcomes)
        if 'reason' in table.colnames:
            table['reason_description'] = [FIT_REASON_DESCRIPTIONS[code]
                                           for code in table['reason']]
        rows = np.asarray(outcomes.fit_index)
        valid = rows >= 0
        parameters = (self.model_columns if self.model_columns is not None else
                      {**self.params, **{name+'_err': value for name, value in self.param_errors.items()}})
        columns = {**parameters,
                   'chi_squared': self.chi_squared, 'prior_chi2': self.prior_chi2,
                   'ndof': self.ndof, 'edm': self.edm}
        for name, values in columns.items():
            data = np.zeros(len(rows))
            data[valid] = np.asarray(values)[rows[valid]]
            table[name] = MaskedColumn(data, mask=~valid)
        return table

    def write_table(self, path, **kwargs):
        """Explicitly export a result table using Astropy's supported formats."""
        self.to_table().write(path, **kwargs)
