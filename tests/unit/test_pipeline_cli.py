"""Command-line refresh requests must be explicit and reach the runner intact."""

import os

import pytest
import yaml

from jax_pipeline import pippin


@pytest.mark.parametrize("flags,refreshed", [
    ([], set()),
    (["--refresh", "first", "second"], {"first", "second"}),
    (["--refresh-all"], {"first", "second", "third"}),
])
def test_refresh_request_reaches_runner(tmp_path, flags, refreshed):
    """The real CLI reruns exactly the requested stages in a cached pipeline."""
    sources = tmp_path / 'sources'
    sources.mkdir()
    tasks = {}
    for name in ('first', 'second', 'third'):
        directory = sources / name
        directory.mkdir()
        tasks[name] = {'OPTS': {'RAW_DIR': str(directory)}}
    config = tmp_path / 'analysis.yml'
    config.write_text(yaml.safe_dump({'DATAPREP': tasks}))
    outdir = tmp_path / 'results'
    argv = [str(config), '--outdir', str(outdir), '--double-precision']
    pippin.main(argv)
    pippin.main([*argv, *flags])
    manifest = yaml.safe_load((outdir / 'MANIFEST.yml').read_text())
    assert manifest['status'] == 'completed'
    assert set(manifest['stages']) == set(tasks)
    for name, stage in manifest['stages'].items():
        assert stage['status'] == ('completed' if name in refreshed else 'loaded')
        assert (outdir / name / 'DONE.yml').is_file()


@pytest.mark.parametrize("flags,message", [
    (["--refresh"], "expected at least one argument"),
    (["--refresh", "BBC", "--refresh-all"], "not allowed with argument"),
    (["--check", "--refresh", "BBC"], "--check does not run stages"),
    (["--check", "--refresh-all"], "--check does not run stages"),
])
def test_invalid_refresh_fails_before_running(capsys, flags, message):
    with pytest.raises(SystemExit) as error:
        pippin.main(["analysis.yml", *flags])
    assert error.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("via_env", [False, True])
def test_missing_absolute_path_names_task(tmp_path, via_env):
    """Missing paths retain the task identity after environment expansion."""
    from jax_snlc_core.logging import SNANAError
    missing = tmp_path / "absent.input"
    previous = os.environ.get('PIPELINE_TEST_INPUTS')
    os.environ['PIPELINE_TEST_INPUTS'] = str(tmp_path)
    try:
        path = "$PIPELINE_TEST_INPUTS/absent.input" if via_env else str(missing)
        with pytest.raises(SNANAError) as error:
            pippin._locate(path, [], "SIM", "survey")
        assert "SIM survey" in str(error.value)
        assert str(missing) in str(error.value)
        assert "does not exist" in str(error.value)
    finally:
        if previous is None:
            os.environ.pop('PIPELINE_TEST_INPUTS', None)
        else:
            os.environ['PIPELINE_TEST_INPUTS'] = previous
