# Vendored Pippin config

`valid_create_cov.yml` and the `data_files/surveys` tree below it are copied
verbatim from the Pippin repository so that
`tests/unit/test_pippin_reader.py::test_a_real_pippin_config_builds_the_whole_chain`
reads a config written by Pippin rather than by this suite.

Source: https://github.com/dessn/Pippin at commit
`e15ee2161292821d4f7ecf56fc010cde4080bff6`.

| vendored path | Pippin path |
|---|---|
| `valid_create_cov.yml` | `tests/config_files/valid_create_cov.yml` |
| `data_files/surveys/sdss/sims_ia/sn_ia_g10_sdss_3yr.input` | same under `data_files/` |
| `data_files/surveys/sdss/sims_cc/sn_ii_templates.input` | same under `data_files/` |
| `data_files/surveys/sdss/sims_cc/sn_ibc_templates.input` | same under `data_files/` |
| `data_files/surveys/des/lcfit_nml/des_5yr.nml` | same under `data_files/` |
| `data_files/surveys/des/bbc/bbc_5yr.input` | same under `data_files/` |

Only the files the config names are vendored; the reader resolves paths and
does not open them. Refresh by copying from a newer Pippin checkout and
recording its commit here.
