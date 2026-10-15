"""Command-line refresh requests must be explicit and reach the runner intact."""

import pytest

from jax_pipeline import pippin


@pytest.mark.parametrize("flags,expected", [
    ([], ()),
    (["--refresh", "BBC", "COV"], ["BBC", "COV"]),
    (["--refresh-all"], True),
])
def test_refresh_request_reaches_runner(monkeypatch, flags, expected):
    calls = []
    monkeypatch.setattr(pippin, "run_pippin", lambda *a, **kw: calls.append((a, kw)))
    monkeypatch.setattr(pippin, "configure_precision", lambda value: None)
    pippin.main(["analysis.yml", "--outdir", "results", *flags])
    assert len(calls) == 1
    assert calls[0][0][:2] == ("analysis.yml", "results")
    assert calls[0][1]["refresh"] == expected


@pytest.mark.parametrize("flags,message", [
    (["--refresh"], "expected at least one argument"),
    (["--refresh", "BBC", "--refresh-all"], "not allowed with argument"),
    (["--check", "--refresh", "BBC"], "--check does not run stages"),
    (["--check", "--refresh-all"], "--check does not run stages"),
])
def test_invalid_refresh_fails_before_running(monkeypatch, capsys, flags, message):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid CLI arguments must not start an analysis")
    monkeypatch.setattr(pippin, "run_pippin", unexpected)
    with pytest.raises(SystemExit) as error:
        pippin.main(["analysis.yml", *flags])
    assert error.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("via_env", [False, True])
def test_missing_absolute_path_names_task(tmp_path, monkeypatch, via_env):
    from jax_snlc_core.utils.error_handling import SNANAError
    missing = tmp_path / "absent.input"
    monkeypatch.setenv("PIPELINE_TEST_INPUTS", str(tmp_path))
    path = "$PIPELINE_TEST_INPUTS/absent.input" if via_env else str(missing)
    with pytest.raises(SNANAError) as error:
        pippin._locate(path, [], "SIM", "survey")
    assert "SIM survey" in str(error.value)
    assert str(missing) in str(error.value)
    assert "does not exist" in str(error.value)
