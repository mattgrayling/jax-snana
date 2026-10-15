"""Sanity-check imports + CLI scripts run end-to-end against bundled fixtures."""

import os
import subprocess
import sys

import pytest

from jax_salt2mu.config_parser import parse_bbc_input
from jax_snlc_core.utils.error_handling import SNANAError
from jax_snlc_core.utils.precision import DEFAULT_DOUBLE_PRECISION


def test_cli_help_runs(repo_root) -> None:
    """CLI scripts respond to --help without crashing."""
    for script in ("snlc_sim.py", "snlc_fit.py", "wfit.py", "salt2mu.py",
                   "create_covariance.py", "replay_snana_dump.py",
                   "lcplot_from_fitres.py"):
        result = subprocess.run(
            [sys.executable, str(repo_root / script), "--help"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, (
            f"{script} --help exited {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


_IMPORTABLE = [
    "jax_snlc_core", "jax_snlc_sim", "jax_snlc_fit", "jax_salt2mu",
    "jax_wfit", "jax_create_cov", "snlc_sim", "snlc_fit", "salt2mu",
    "wfit", "create_covariance",
]


def _probe(repo_root, code, jax_x64=None):
    """Run `code` in a fresh process with JAX's own x64 variable controlled."""
    env = os.environ.copy()
    env.pop("JAX_ENABLE_X64", None)
    if jax_x64 is not None:
        env["JAX_ENABLE_X64"] = jax_x64
    env["PYTHONPATH"] = str(repo_root)
    return subprocess.run([sys.executable, "-c", code], capture_output=True,
                          env=env, text=True, timeout=120)


@pytest.mark.parametrize("module", _IMPORTABLE)
def test_importing_leaves_the_precision_flag_alone(repo_root, module) -> None:
    """The frame comes from an input key, so no import may choose it.

    JAX's own default is the witness: with nothing set, x64 is off, and a
    package that still fixed the frame on import would report True here.
    """
    result = _probe(repo_root,
                    f"import jax; import {module}; print(jax.config.x64_enabled)")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "False"


def test_importing_does_not_overrule_jaxs_own_variable(repo_root) -> None:
    """The other direction: an import must not clear the flag either."""
    result = _probe(repo_root,
                    "import jax; import jax_wfit; print(jax.config.x64_enabled)",
                    jax_x64="1")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "True"


# BBC takes its keys as command-line tokens with no input file, so it is the
# cheapest parser to ask; every parser reads the key in the same first stage.
_PARSE_PROBE = (
    "import jax\n"
    "from jax_salt2mu.config_parser import parse_bbc_input\n"
    "config = parse_bbc_input(None, {tokens!r})\n"
    "print(config.double_precision, jax.config.x64_enabled)\n"
)


@pytest.mark.parametrize("tokens, expected", [
    (["double_precision=1"], "True"),
    (["double_precision=0"], "False"),
])
def test_parsing_an_input_records_the_frame_and_leaves_it_alone(
        repo_root, tokens, expected) -> None:
    result = _probe(repo_root, _PARSE_PROBE.format(tokens=tokens))
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == f"{expected} False"


def test_an_input_that_omits_precision_preserves_it_until_cli_setup(repo_root) -> None:
    result = _probe(repo_root, _PARSE_PROBE.format(tokens=[]) + (
        'from jax_snlc_core.utils.precision import configure_precision\n'
        'configure_precision(config.double_precision)\n'
        'print(jax.config.x64_enabled)\n'))
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-2:] == [
        'None False', str(DEFAULT_DOUBLE_PRECISION)]


@pytest.mark.parametrize("argv, expected", [
    (["in.input", "a=1"], ("in.input", ["a=1"], None)),
    # SNANA's "no parFile" sentinels (parse_parFile, SALT2mu.c:19321)
    (["NULL", "datafile=x"], (None, ["datafile=x"], None)),
    # a key in the positional slot is an override, not a filename
    (["datafile=x"], (None, ["datafile=x"], None)),
    # cat_only names a mode; the optional /N is its prescale (:20154)
    (["cat_only", "datafile=x"], (None, ["datafile=x"], 1)),
    (["cat_only/5", "datafile=x"], (None, ["datafile=x"], 5)),
    # words without '=' pass through, so CUTWIN keeps its limits (:19409)
    (["in.input", "CUTWIN", "x1", "-3", "3"],
     ("in.input", ["CUTWIN", "x1", "-3", "3"], None)),
    # with --input, the first trailing token is an override too
    (["--input", "in.input", "CUTWIN", "x1", "-3", "3"],
     ("in.input", ["CUTWIN", "x1", "-3", "3"], None)),
    (["--input", "in.input", "zmax=0.7", "nzbin=5"],
     ("in.input", ["zmax=0.7", "nzbin=5"], None)),
])
def test_salt2mu_positional_forms(argv, expected) -> None:
    from salt2mu import _parse_cli_args

    input_file, overrides, _verbose, cat_prescale, settings = _parse_cli_args(argv)
    assert settings == {}
    assert (input_file, overrides, cat_prescale) == expected


@pytest.mark.parametrize("argv, expected", [
    # SNANA's own form: the input file is positional, overrides follow
    (["my.input"], ("my.input", [])),
    (["my.input", "NGENTOT_LC", "500"], ("my.input", ["NGENTOT_LC", "500"])),
    (["my.input", "GENRANGE_REDSHIFT", "0.2", "0.9"],
     ("my.input", ["GENRANGE_REDSHIFT", "0.2", "0.9"])),
    # --input keeps working, and the positional slot then holds an override
    (["-i", "my.input", "NGENTOT_LC", "500"], ("my.input", ["NGENTOT_LC", "500"])),
    (["--input", "my.input"], ("my.input", [])),
])
def test_snlc_sim_input_and_override_forms(argv, expected) -> None:
    from snlc_sim import _parse_cli_args, _resolve_input_and_overrides

    assert _resolve_input_and_overrides(_parse_cli_args(argv)) == expected


@pytest.mark.parametrize("argv, flag, value", [
    (["my.input", "--verbose", "NGENTOT_LC", "500"], "verbose", True),
    (["my.input", "NGENTOT_LC", "500", "--n-sne", "10"], "n_sne", 10),
])
def test_snlc_sim_flags_survive_alongside_overrides(argv, flag, value) -> None:
    """argparse.REMAINDER would swallow these into the override list; nargs='*'
    does not. The overrides must still come through intact."""
    from snlc_sim import _parse_cli_args, _resolve_input_and_overrides

    args = _parse_cli_args(argv)
    assert (args.settings[flag] if flag in args.settings else getattr(args, flag)) == value
    assert _resolve_input_and_overrides(args) == ("my.input", ["NGENTOT_LC", "500"])


@pytest.mark.parametrize("command", [
    ["salt2mu.py"], ["wfit.py"], ["create_covariance.py"],
    ["-m", "jax_pipeline.pippin"],
])
def test_missing_cli_input_has_one_diagnostic(repo_root, tmp_path, command):
    missing = tmp_path / "missing.input"
    result = subprocess.run(
        [sys.executable, *command, str(missing)], cwd=repo_root,
        env={**os.environ, "PYTHONPATH": str(repo_root)},
        capture_output=True, text=True, timeout=60)
    diagnostic = result.stdout + result.stderr
    assert result.returncode == 1, diagnostic
    assert str(missing) in diagnostic
    assert diagnostic.count("FATAL ERROR ABORT") == 1, diagnostic
    assert "Traceback" not in diagnostic


def test_mistyped_sim_override_has_one_diagnostic(repo_root):
    result = subprocess.run(
        [sys.executable, "snlc_sim.py", "examples/quickstart/sim.input",
         "NGENTOT_LC", "1", "RANSED", "2"], cwd=repo_root,
        env={**os.environ, "PYTHONPATH": str(repo_root)},
        capture_output=True, text=True, timeout=60)
    diagnostic = result.stdout + result.stderr
    assert result.returncode == 1, diagnostic
    assert "RANSED" in diagnostic
    assert "command-line overrides" in diagnostic
    assert diagnostic.count("FATAL ERROR ABORT") == 1, diagnostic
    assert "Traceback" not in diagnostic
