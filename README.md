# jax-snana

[![Docs](https://readthedocs.org/projects/jax-snana/badge/?version=latest)](https://jax-snana.readthedocs.io)
[![Tests](https://github.com/mattgrayling/jax-snana/actions/workflows/tests.yml/badge.svg)](https://github.com/mattgrayling/jax-snana/actions/workflows/tests.yml)
[![Coverage](https://coveralls.io/repos/github/mattgrayling/jax-snana/badge.svg?branch=main)](https://coveralls.io/github/mattgrayling/jax-snana?branch=main)
[![PyPI](https://img.shields.io/pypi/v/jax-snana)](https://pypi.org/project/jax-snana/)
[![arXiv](https://img.shields.io/badge/arXiv-2610.19000-b31b1b.svg)](https://arxiv.org/abs/2610.19000)
[![SNANA](https://img.shields.io/badge/SNANA-v12__02d--12--g74dfb2c5-blue)](https://github.com/RickKessler/SNANA/commit/74dfb2c5)

GPU-accelerated supernova simulation, light curve fitting and cosmology in JAX

Developed and maintained by Matt Grayling (@mattgrayling).

## About
jax-snana is a JAX reimplementation of the supernova analysis framework SNANA, supporting survey
simulations, light curve fitting, BBC distance fitting and cosmological inference on CPU or GPU. It
also provides a compiled simulator for simulation-based inference, as discussed in Grayling 2026
(https://arxiv.org/abs/2610.19000).

jax-snana is validated against SNANA v12_02d-12-g74dfb2c5.

## Installation and usage
jax-snana can be pip-installed via the command `pip install jax-snana`.

Detailed instructions on how to install and run jax-snana can be found here:
https://jax-snana.readthedocs.io

## Support
If you find something unclear, come across any bugs or would like a feature that is not yet
implemented, please raise a GitHub issue.

## Citing the code
If you use jax-snana, please cite Grayling 2026 (arXiv:2610.19000) as well as SNANA, Kessler et al.
2009 (PASP 121, 1028). If you use any of the models included within jax-snana, please also cite the
papers that present them, listed here: https://jax-snana.readthedocs.io/en/latest/citing.html
