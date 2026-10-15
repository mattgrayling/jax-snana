# Unsupported input keys

The tables below list SNANA input keys explicitly rejected by jax-snana,
organised by task. Each entry gives the reason or the setting to use instead.
Keys are listed alphabetically within each table.

Some light-curve fitting keys are accepted when their value leaves the
option inactive, allowing existing input files to retain unused switches.
The corresponding operation is not supported. Restrictions on particular
values of otherwise supported keys, including bit masks and model-specific
options, are given in the task settings pages linked below.

```{contents} On this page
:local:
:depth: 1
```

(unsupported-simulation-keys)=
## Simulation

For accepted keys and flag values, see {doc}`simulation_keys`.

```{eval-rst}
.. rejectedkeys:: jax_snlc_sim.config_parser._REJECTED_KEYS
```

(unsupported-fitting-keys)=
## Light-curve fitting

For accepted keys and flag values, see {doc}`fitting_keys`.

```{eval-rst}
.. rejectedkeys:: jax_snlc_fit.config_parser._REJECTED_KEYS
```

The following `FITWIN_` forms are also rejected:

```{eval-rst}
.. rejectedkeys:: jax_snlc_fit.config_parser._FITWIN_REFUSED
   :key-prefix: FITWIN_
```

The parameter suffixes below are not supported in parameter settings such
as `INIVAL_<PARAM>`, `INISTP_<PARAM>` and `USESIM_<PARAM>`:

```{eval-rst}
.. rejectedkeys:: jax_snlc_fit.config_parser._UNSUPPORTED_PARAM_KEYS
```

(unsupported-distance-keys)=
## Standardisation

For accepted keys, numbered parameters and correction options, see
{doc}`bias_correction_keys` and {doc}`bias_corrections`.
These are keys passed to BBC itself. Pipeline options such as `NSPLITRAN`
are handled separately by the {doc}`Pippin interface <snana_and_pippin>`.

```{eval-rst}
.. rejectedkeys:: jax_salt2mu.config_parser._REJECTED_KEYS
```

`datafile_override` is also rejected when it names a file; use `datafile`.
The survey- and sample-specific forms `CUTWIN_SURVEY(...)` and
`CUTWIN_IDSAMPLE(...)` are not supported. `CUTWIN <column> <low> <high>`
applies a window to all events. `nmax` accepts a total count,
but not counts specified separately by survey or field.

(unsupported-covariance-keys)=
## Covariance construction

For accepted settings, see {doc}`covariance_keys`.

```{eval-rst}
.. rejectedkeys:: jax_create_cov.config_parser._REJECTED_KEYS
```

(unsupported-cosmology-keys)=
## Cosmology fitting

For accepted settings, see {doc}`cosmology_keys`. The names below are written
with a single leading dash when used as SNANA command-line flags, for example
`-fitswrite`.

```{eval-rst}
.. rejectedkeys:: jax_wfit.config_parser._REJECTED_KEYS
```
