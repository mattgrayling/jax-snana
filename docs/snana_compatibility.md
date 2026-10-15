# Differences from SNANA

jax-snana accepts SNANA simulation inputs, light-curve fitting input files and
SALT2mu inputs, as well as SNANA light-curve data and analysis tables.

## Light-curve fitting

SNANA uses MINUIT for light-curve fitting; jax-SNANA uses a damped Newton
method with derivatives calculated through automatic differentiation.

## Bias corrections

jax-SNANA supports fitting the intrinsic scatter jointly with the other BBC
parameters using `sigint_mode=joint`. SNANA estimates the scatter by
iterating until the reduced χ² is approximately one. Both the iterative
approach and a fixed scatter are also available in jax-SNANA; see
{doc}`bias_corrections`.

## Simulation

jax-SNANA applies host-dependent magnitude shifts from HOSTLIB `SNMAGSHIFT`
values to every model, including BayeSN, SIMSED and NON1ASED. SNANA applies
these shifts to SALT-family models.

## Cosmology fitting

SNANA's `wfit` evaluates a χ² grid. jax-SNANA also supports optimisation,
which is the default, and NUTS posterior sampling. See
{doc}`cosmology` for the fitting methods.

## Cosmological distances

The simulator and BBC include radiation in their reference distances by
default, whereas SNANA omits it in those stages. Set `radiation: false`
in YAML to omit it in jax-SNANA.

## Unsupported features

Some SNANA features are not yet implemented:

- Strongly-lensed supernova simulations.
- LCLIB models, which simulate from libraries of precomputed light curves.
- Grid simulations (`GENSOURCE: GRID`).

{doc}`unsupported_input_keys` lists the explicitly rejected keys by task.

```{toctree}
:maxdepth: 1

unsupported_input_keys
```
