"""Storage and indexing of simulated observations, from padded batches to flat epochs."""

import jax.numpy as jnp
import numpy as np

from jax_snlc_sim.simlib import SIMLIB_OBSERVATION_KEYS


# flattened per-epoch arrays: event i's epochs start at epoch_start[i]
PER_EPOCH_RESULT_KEYS = (
    'magnitudes', 'fluxcals', 'fluxcals_noisy',
    'fluxcalerrs', 'snrs', 'snrs_model', 'snrs_mon', 'fluxgal_pe',
    'filter_z_valid')
PER_EPOCH_SELECTION_KEYS = (
    'detection_mask', 'valid_mask', 'trigger_mask', 'saturated_mask',
    'snrs_sat', 'fluxcals_noisy_sat', 'fluxcalerrs_sat', 'photprob_2d',
    'snrmax_epoch', 'nearpeak_epoch')
PER_EPOCH_SCHEDULE_KEYS = SIMLIB_OBSERVATION_KEYS + ('phases', 'simlib_obs_idx')


class LookupColumn:
    """A column looked up from a table by a per-row index, materialised only
    where it is read: a slice at a time by the FITS writer, or whole by
    ``np.asarray``. The written BAND and FIELD are 20-character strings on
    every epoch, and the SIMLIB quantities are copies of the entry's
    epochs; held whole they would be most of a write's memory, and as an
    index they are four bytes. ``transform`` and ``dtype`` apply to what is
    read, never to the table."""

    ndim = 1

    def __init__(self, table, index, transform=None, dtype=None):
        self.table = np.asarray(table)
        self.index = np.asarray(index)
        self.transform = transform
        self.dtype = np.dtype(dtype) if dtype is not None else self.table.dtype

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        out = self.table[self.index[item]]
        if self.transform is not None:
            out = self.transform(out)
        return out.astype(self.dtype, copy=False)

    def __array__(self, dtype=None, copy=None):
        out = self[:]
        return out if dtype is None else out.astype(dtype)

    def astype(self, dtype):
        return LookupColumn(self.table, self.index, self.transform, dtype)


# per-epoch SIMLIB copies a finished run keeps only as simlib_obs_idx
SIMLIB_EPOCH_KEYS = tuple(k for k in SIMLIB_OBSERVATION_KEYS if k != 'filter_indices')


def simlib_epochs(resources, results, key, transform=None):
    """A per-epoch SIMLIB quantity of the run's kept epochs, flattened: the
    schedule's own array when the run kept it, else a lookup by SIMLIB
    index, materialised where it is read; ``transform`` applies to the
    values either way."""
    schedule = results['observation_schedule']
    if key in schedule:
        values = np.asarray(schedule[key]).reshape(-1)
        return values if transform is None else transform(values)
    return LookupColumn(resources.flat_obs_data[key],
                        np.asarray(schedule['simlib_obs_idx']).reshape(-1),
                        transform)


def span_indices(start, count):
    """The flat indices of the spans ``(start, count)``, in order."""
    total = int(count.sum())
    seg = np.repeat(np.arange(len(count)), count)
    within = np.arange(total) - np.repeat(np.cumsum(count) - count, count)
    return start[seg] + within, seg


def keep_events(results, selection, keep, written_only):
    """The results and selection of the events ``keep`` only; with
    ``written_only`` the per-epoch arrays hold just the passing events'
    valid epochs, the ones the writer reports, so the packed arrays are
    the written columns. Whatever reads a rejected event or an unwritten
    epoch (the generated record, the NOISE rows, the SED-true epochs) has
    taken it from the batch before this."""
    n_events = len(keep)
    rows = np.flatnonzero(keep)
    sched, seg = span_indices(np.asarray(results['epoch_start'])[rows],
                              np.asarray(results['epoch_count'])[rows])
    if written_only:
        written = (np.asarray(selection['valid_mask']).reshape(-1)[sched]
                   & np.asarray(selection['passing_mask'])[rows][seg])
        sched, seg = sched[written], seg[written]
    count = np.bincount(seg, minlength=len(rows))

    def take(value):
        if isinstance(value, dict):
            return {k: take(v) for k, v in value.items()}
        if hasattr(value, 'shape') and value.ndim >= 1 and value.shape[0] == n_events:
            return np.asarray(value)[rows]
        return value

    def pack(value):
        return None if value is None else np.asarray(value).reshape(-1)[sched]

    out = {'epoch_start': np.cumsum(count) - count, 'epoch_count': count}
    for key, value in results.items():
        if key in out:
            continue
        if key in PER_EPOCH_RESULT_KEYS:
            out[key] = pack(value)
        elif key == 'observation_schedule':
            out[key] = {k: pack(v) if k in PER_EPOCH_SCHEDULE_KEYS else take(v)
                        for k, v in value.items()}
        elif key == 'sed_true_mjds':
            n_spec = np.asarray(results['sed_true_count'])
            spans, _ = span_indices((np.cumsum(n_spec) - n_spec)[rows], n_spec[rows])
            out[key] = value[spans]
        else:
            out[key] = take(value)
    sel = {k: (pack(v) if k in PER_EPOCH_SELECTION_KEYS
               else v if k == 'generated' else take(v))
           for k, v in selection.items()}
    return out, sel


def padded_rows(results, array, rows, fill):
    """The spans of ``rows`` of a per-epoch array as a (len(rows),
    widest span) array, ``fill`` past each row's epochs; for a consumer
    that works on an event-by-epoch grid."""
    count = np.asarray(results['epoch_count'])[rows]
    flat, seg = span_indices(np.asarray(results['epoch_start'])[rows], count)
    out = np.full((len(rows), int(count.max()) if len(count) else 0), fill,
                  dtype=np.asarray(array).dtype)
    within = np.arange(len(flat)) - np.repeat(np.cumsum(count) - count, count)
    out[seg, within] = np.asarray(array).reshape(-1)[flat]
    return out


def row_slice(results, row):
    """The slice of a flattened per-epoch array holding row ``row``'s
    scheduled epochs."""
    start = int(results['epoch_start'][row])
    return slice(start, start + int(results['epoch_count'][row]))


def epoch_layout(n_events, max_obs, n_obs_per_sn):
    """The span of each event's row of a (n_events, max_obs) per-epoch
    array; traceable, for the photometry."""
    return {'epoch_start': jnp.arange(n_events, dtype=jnp.int32) * max_obs,
            'epoch_count': jnp.asarray(n_obs_per_sn, dtype=jnp.int32)}
