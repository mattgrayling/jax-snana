"""The pipeline's driver, with stages that do no science.

What is under test is the driver's contract: the order stages run in, that
each runs once, where each writes, what it is handed, the manifest, and that
a finished stage is loaded rather than rerun until something it depends on
changes. The built-in stages are covered by tests that simulate and fit.
"""

from contextlib import chdir
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Mapping

import pytest
import yaml

from jax_pipeline import BBC, Classify, Cosmology, Covariance, Data, Fit, Sim, Stage, run
from jax_pipeline.run import file_digest, validate_graph
from jax_pipeline.systematics import cells, key, overrides_for, scales
from jax_snlc_core.logging import SNANAError
from jax_snlc_core.snana_files import fitopt_muopt_name


@dataclass(eq=False)
class Recorder(Stage):
    """A stage that writes its name to a file and hands the file's text on."""

    name: str
    consumed: Mapping[str, Stage] = field(default_factory=dict)
    setting: int = 0
    source: Path | None = None
    runs: ClassVar[dict[str, int]] = {}
    given: ClassVar[dict[str, Any]] = {}

    def inputs(self):
        return self.consumed

    def run(self, inputs, outdir):
        Recorder.runs[self.name] = Recorder.runs.get(self.name, 0) + 1
        Recorder.given[self.name] = dict(inputs)
        (Path(outdir) / "result.txt").write_text(f"{self.name}-result")
        return self.load(inputs, outdir)

    def load(self, inputs, outdir):
        return (Path(outdir) / "result.txt").read_text()


@dataclass(eq=False)
class NoLoader(Recorder):
    """A stage that cannot read its result back, so it runs every time."""

    def run(self, inputs, outdir):
        Recorder.runs[self.name] = Recorder.runs.get(self.name, 0) + 1
        return f"{self.name}-result"

    def load(self, inputs, outdir):
        return Stage.load(self, inputs, outdir)


@pytest.fixture(autouse=True)
def clear_counts():
    Recorder.runs.clear()
    Recorder.given.clear()


def test_stages_run_after_what_they_consume(tmp_path):
    first = Recorder("first")
    middle = Recorder("middle", {"data": first})
    last = Recorder("last", {"data": middle, "extra": first})

    results = run(last, outdir=tmp_path)

    assert list(results) == ["first", "middle", "last"]
    assert Recorder.given["middle"] == {"data": "first-result"}
    assert Recorder.given["last"] == {"data": "middle-result", "extra": "first-result"}
    assert results["last"] == "last-result"


def test_a_shared_stage_runs_once(tmp_path):
    """Two consumers of one stage must not simulate or fit it twice."""
    shared = Recorder("shared")
    left = Recorder("left", {"data": shared})
    right = Recorder("right", {"data": shared})

    run(left, right, outdir=tmp_path)

    assert Recorder.runs["shared"] == 1


def test_processing_phases_group_realisations(tmp_path):
    sims = [Sim(f"sim{i}", tmp_path / "sim.input") for i in range(2)]
    fit = Fit("fit", tmp_path / "fit.nml", datasets=sims)
    correction = Fit("correction", fitres=tmp_path / "biascor.FITRES")
    ends = []
    for i, sim in enumerate(sims):
        classified = Classify(f"classify{i}", fit=fit.version(sim),
                              predictions=tmp_path / "predictions.csv")
        bbc = BBC(f"bbc{i}", data=classified, biascor=correction)
        cov = Covariance(f"cov{i}", distances=bbc)
        ends.append(Cosmology(f"cosmo{i}", covariance=cov))

    ordered = validate_graph(ends)

    assert [stage.name for stage in ordered] == [
        "sim0", "sim1", "fit", "correction",
        "classify0", "classify1", "bbc0", "bbc1", "cov0", "cov1",
        "cosmo0", "cosmo1",
    ]
    assert validate_graph(ends) == ordered
    assert ordered.count(fit) == ordered.count(correction) == 1


def test_a_fit_of_several_versions_must_be_narrowed_to_one(tmp_path):
    sims = [Sim(f"sim{i}", tmp_path / "sim.input") for i in range(2)]
    fit = Fit("fit", tmp_path / "fit.nml", datasets=sims)

    with pytest.raises(SNANAError, match=r"fits 2 versions \(sim0, sim1\)"):
        BBC("bbc", data=fit)
    bbc = BBC("bbc", data=fit.version(sims[0]), biascor=fit.version(sims[1]))
    assert set(bbc.inputs()) == {"data:fit/sim0", "biascor:fit/sim1"}
    assert validate_graph([bbc]) == [*sims, fit, bbc]


def test_phases_group_chains_with_different_depths(tmp_path):
    external = Fit("external", fitres=tmp_path / "data.FITRES")
    short = BBC("short", data=external)
    sim = Sim("sim", tmp_path / "sim.input")
    fit = Fit("fit", tmp_path / "fit.nml", datasets=[sim])
    custom = Recorder("custom", {"fit": fit})
    long = BBC("long", data=custom)

    assert [stage.name for stage in validate_graph([short, long])] == [
        "sim", "external", "fit", "custom", "short", "long",
    ]


def test_dependencies_override_processing_phase(tmp_path):
    """A data-stage subclass still runs after its later-phase dependency."""
    @dataclass(eq=False)
    class DerivedData(Data):
        source: Stage

        def inputs(self):
            return {"custom": self.source}

    fit = Fit("fit", fitres=tmp_path / "data.FITRES")
    bbc = BBC("bbc", data=fit)
    custom = Recorder("custom", {"bbc": bbc})
    data = DerivedData("later_data", tmp_path, custom)
    independent = Sim("independent", tmp_path / "sim.input")

    ordered = validate_graph([data, independent])

    assert [stage.name for stage in ordered] == [
        "independent", "fit", "bbc", "custom", "later_data",
    ]
    positions = {stage: i for i, stage in enumerate(ordered)}
    assert all(positions[used] < positions[stage]
               for stage in ordered for used in stage.inputs().values())


def test_each_stage_writes_under_its_own_directory(tmp_path):
    first = Recorder("first")
    second = Recorder("second", {"data": first})

    run(second, outdir=tmp_path)

    assert (tmp_path / "first" / "result.txt").read_text() == "first-result"
    assert (tmp_path / "second" / "result.txt").read_text() == "second-result"
    assert (tmp_path / "first" / "DONE.yml").is_file()


def test_stage_list(tmp_path):
    stages = [Recorder("first"), Recorder("second")]
    with pytest.raises(SNANAError, match=r"run\(\*stages\)"):
        run(stages, outdir=tmp_path)
    assert list(run(*stages, outdir=tmp_path)) == ['first', 'second']


@pytest.mark.parametrize("name", ["", ".", "..", "../outside", "/absolute"])
def test_stage_names_cannot_escape_output(tmp_path, name):
    marker = tmp_path / "keep.txt"
    marker.write_text("unrelated data")
    with pytest.raises(SNANAError, match="single directory name"):
        run(Recorder(name), outdir=tmp_path / "results")
    assert marker.read_text() == "unrelated data"
    assert not (tmp_path / "results").exists()


def test_the_manifest_records_the_run(tmp_path):
    first = Recorder("first", setting=3)
    second = Recorder("second", {"data": first})

    run(second, outdir=tmp_path)
    written = yaml.safe_load((tmp_path / "MANIFEST.yml").read_text())

    assert list(written["stages"]) == ["first", "second"]
    entry = written["stages"]["second"]
    assert entry["stage"] == "Recorder"
    assert entry["inputs"] == {"data": "first"}
    assert entry["loaded"] is False
    assert written["stages"]["first"]["setting"] == 3
    assert "double_precision" in written and "generated" in written


def test_a_finished_stage_is_loaded_not_rerun(tmp_path):
    first = Recorder("first")
    second = Recorder("second", {"data": first})

    once = run(second, outdir=tmp_path)
    again = run(second, outdir=tmp_path)
    written = yaml.safe_load((tmp_path / "MANIFEST.yml").read_text())

    assert Recorder.runs == {"first": 1, "second": 1}
    assert again == once
    assert all(entry["loaded"] for entry in written["stages"].values())


@pytest.mark.parametrize("refresh", [("second",), "second"])
def test_refresh_reruns_a_stage_and_everything_after_it(tmp_path, refresh):
    first = Recorder("first")
    second = Recorder("second", {"data": first})
    third = Recorder("third", {"data": second})

    run(third, outdir=tmp_path)
    run(third, outdir=tmp_path, refresh=refresh)

    assert Recorder.runs == {"first": 1, "second": 2, "third": 2}


def test_a_changed_setting_upstream_reruns_downstream(tmp_path):
    run(Recorder("second", {"data": Recorder("first", setting=1)}), outdir=tmp_path)
    run(Recorder("second", {"data": Recorder("first", setting=2)}), outdir=tmp_path)

    assert Recorder.runs == {"first": 2, "second": 2}


@pytest.mark.parametrize("failure,status", [(RuntimeError("fit failed"), "failed"),
                                           (KeyboardInterrupt(), "interrupted")])
def test_manifest_tracks_failed_rerun_and_resume(tmp_path, failure, status):
    """A failed or interrupted stage can resume after its failure is resolved."""
    failure_marker = tmp_path / "fail"

    class InterruptibleRecorder(Recorder):
        def run(self, inputs, outdir):
            if failure_marker.exists():
                progress = yaml.safe_load((tmp_path / "MANIFEST.yml").read_text())
                assert progress["status"] == "running"
                assert progress["stages"]["first"]["status"] == "loaded"
                assert progress["stages"]["second"]["status"] == "running"
                assert progress["stages"]["last"]["status"] == "pending"
                raise failure
            return super().run(inputs, outdir)

    first = Recorder("first")
    second = InterruptibleRecorder("second", {"data": first})
    last = Recorder("last", {"data": second})
    run(last, outdir=tmp_path)
    failure_marker.touch()
    with pytest.raises(type(failure)) as caught:
        run(last, outdir=tmp_path, refresh="second")
    assert caught.value is failure
    progress = yaml.safe_load((tmp_path / "MANIFEST.yml").read_text())
    assert progress["status"] == status
    assert progress["current_stage"] == "second"
    assert progress["stages"]["second"]["status"] == status
    assert type(failure).__name__ in progress["stages"]["second"]["error"]
    assert not (tmp_path / "second" / "DONE.yml").exists()
    failure_marker.unlink()
    run(last, outdir=tmp_path)
    progress = yaml.safe_load((tmp_path / "MANIFEST.yml").read_text())
    assert progress["status"] == "completed" and "error" not in progress
    assert "current_stage" not in progress
    assert Recorder.runs == {"first": 1, "second": 2, "last": 2}


def test_manifest_records_failure_before_stage_execution(tmp_path):
    stage = Recorder("missing", source=tmp_path / "absent.txt")
    with pytest.raises(SNANAError, match="does not exist"):
        run(stage, outdir=tmp_path)
    progress = yaml.safe_load((tmp_path / "MANIFEST.yml").read_text())
    assert progress["status"] == "failed"
    assert progress["stages"]["missing"]["status"] == "failed"
    assert Recorder.runs == {}


def test_manifest_records_cached_load_failure(tmp_path):
    """An unreadable cached result is reported as a stage failure."""
    stage = Recorder("cached")
    run(stage, outdir=tmp_path)
    (tmp_path / 'cached' / 'result.txt').write_bytes(b'\xff')
    with pytest.raises(UnicodeDecodeError):
        run(stage, outdir=tmp_path)
    progress = yaml.safe_load((tmp_path / "MANIFEST.yml").read_text())
    assert progress["stages"]["cached"]["status"] == "failed"


def test_completion_logs_existing_result_locations(tmp_path, caplog):
    """A pipeline without cosmology reports only the outputs it writes."""
    caplog.set_level("INFO", logger="jax_pipeline.run")
    run(Recorder("first"), outdir=tmp_path)
    assert f"Outputs: {tmp_path}" in caplog.text
    assert f"Run manifest: {tmp_path / 'MANIFEST.yml'}" in caplog.text
    assert "Cosmology summary:" not in caplog.text
    assert not (tmp_path / "COSMOLOGY_SUMMARY.csv").exists()


def test_a_changed_input_file_reruns_the_stage(tmp_path):
    source = tmp_path / "in.txt"
    source.write_text("a")
    run(Recorder("only", source=source), outdir=tmp_path / "run")
    run(Recorder("only", source=source), outdir=tmp_path / "run")
    source.write_text("b")
    run(Recorder("only", source=source), outdir=tmp_path / "run")

    assert Recorder.runs == {"only": 2}


@pytest.mark.parametrize("subdir", ["", "nested"])
def test_data_edits_invalidate_dependents(tmp_path, subdir):
    import os
    import numpy as np
    from astropy.io import fits

    source = tmp_path / "data"
    folder = source / subdir
    folder.mkdir(parents=True)
    path = folder / "PHOT.FITS"
    fits.BinTableHDU.from_columns([
        fits.Column(name="FLUXCAL", format="D", array=np.array([1., 2.]))
    ]).writeto(path)
    original = path.stat()
    first = Recorder("data", source=source)
    fitted = Recorder("fit", {"data": first})
    other = Recorder("unrelated")
    outdir = tmp_path / "run"
    run(fitted, other, outdir=outdir)
    run(fitted, other, outdir=outdir)
    assert Recorder.runs == {"data": 1, "fit": 1, "unrelated": 1}
    with fits.open(path, mode="update") as table:
        table[1].data["FLUXCAL"][0] = 3.
    assert path.stat().st_size == original.st_size
    os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
    run(fitted, other, outdir=outdir)
    assert Recorder.runs == {"data": 2, "fit": 2, "unrelated": 1}


def test_a_stage_without_a_loader_runs_every_time(tmp_path):
    run(NoLoader("again"), outdir=tmp_path)
    run(NoLoader("again"), outdir=tmp_path)

    assert Recorder.runs == {"again": 2}


def test_a_cycle_is_named_not_hung(tmp_path):
    first = Recorder("first")
    second = Recorder("second", {"data": first})
    first.consumed = {"back": second}

    with pytest.raises(SNANAError, match="cycle"):
        run(second, outdir=tmp_path)


def test_two_stages_may_not_share_a_name(tmp_path):
    """Their output directories would be the same one."""
    first = Recorder("same")
    second = Recorder("same", {"data": first})

    with pytest.raises(SNANAError, match="named same"):
        run(second, outdir=tmp_path)


def test_the_grid_is_nominal_first_and_snana_numbered():
    grid = cells(("a", "b"), ("m",))

    assert [fitopt_muopt_name(cell) for cell in grid] == [
        "FITOPT000_MUOPT000", "FITOPT000_MUOPT001",
        "FITOPT001_MUOPT000", "FITOPT001_MUOPT001",
        "FITOPT002_MUOPT000", "FITOPT002_MUOPT001",
    ]
    assert fitopt_muopt_name((2, 13)) == "FITOPT002_MUOPT013"
    assert [key(cell, ("a", "b"), ("m",)) for cell in grid] == [
        "nominal", "m", "a", ("a", "m"), "b", ("b", "m")]


def test_the_nominal_cell_applies_no_overrides():
    variants = {"shift": {"KEY": 1}}

    assert overrides_for(variants, 0) == {}
    assert overrides_for(variants, 1) == {"KEY": 1}


def test_scales_carry_the_nominal_and_every_label():
    """The nominal is DEFAULT, the label create_covariance and Pippin use."""
    numbered = scales(("cal", "dust"), {"cal": 0.5})

    assert numbered == {0: ("DEFAULT", 1.0), 1: ("cal", 0.5), 2: ("dust", 1.0)}


def test_an_include_line_with_a_trailing_comment_is_hashed(tmp_path):
    """A SNANA input line ends at its first '#', so the hash must not take the
    comment for another included file."""
    included = tmp_path / "shared.input"
    included.write_text("GENMODEL: SALT3\n")
    main = tmp_path / "main.input"
    main.write_text("INPUT_FILE_INCLUDE: shared.input   # the shared block\n")

    first = file_digest(main)
    assert set(first["includes"]) == {"shared.input"}

    included.write_text("GENMODEL: SALT2\n")
    assert file_digest(main) != first


def test_a_binary_model_file_is_hashed_by_its_bytes(tmp_path):
    """A trained model given by path, such as SuperNNova's .pt, is not text;
    it hashes by its bytes and includes nothing."""
    model = tmp_path / "model.pt"
    model.write_bytes(bytes([0x80, 0x97, 0xff, 0x00]) * 64)

    first = file_digest(model)
    assert set(first) == {"file"}

    model.write_bytes(bytes([0x80, 0x97, 0xfe, 0x00]) * 64)
    assert file_digest(model) != first



def test_nested_include_edits_invalidate_the_stage(tmp_path):
    from jax_snlc_sim.inputs import read_keys

    with chdir(tmp_path):
        inputs = tmp_path / "inputs"
        (inputs / "sub").mkdir(parents=True)
        primary = inputs / "primary.input"
        primary.write_text("INPUT_FILE_INCLUDE: sub/mid.input\n")
        (inputs / "sub/mid.input").write_text("INPUT_INCLUDE_FILE: leaf.input\n")
        leaf = inputs / "leaf.input"
        leaf.write_text("GENPREFIX: FIRST\n")
        # A nearby file is not the leaf the parser reads.
        (inputs / "sub/leaf.input").write_text("GENPREFIX: WRONG\n")
        assert read_keys(primary)["GENPREFIX"] == "FIRST"
        stage = Recorder("sim", source=primary)
        outdir = tmp_path / "run"
        run(stage, outdir=outdir)
        run(stage, outdir=outdir)
        leaf.write_text("GENPREFIX: SECOND\n")
        assert read_keys(primary)["GENPREFIX"] == "SECOND"
        run(stage, outdir=outdir)
        assert Recorder.runs == {"sim": 2}


def test_cached_stage_keeps_its_producing_version(tmp_path):
    """Loading older results preserves their provenance until they are refreshed."""
    from jax_snlc_core.logging import version_string

    stage = Recorder('first')
    run(stage, outdir=tmp_path)
    done_path = tmp_path / 'first' / 'DONE.yml'
    record = yaml.safe_load(done_path.read_text())
    record['version'] = 'jax-original'
    original = yaml.safe_dump(record)
    done_path.write_text(original)
    run(stage, outdir=tmp_path)
    manifest = yaml.safe_load((tmp_path / 'MANIFEST.yml').read_text())
    assert manifest['version'] == version_string()
    assert manifest['stages']['first']['produced_version'] == 'jax-original'
    assert done_path.read_text() == original
    assert Recorder.runs == {'first': 1}
    run(stage, outdir=tmp_path, refresh=True)
    assert yaml.safe_load(done_path.read_text())['version'] == version_string()


def test_legacy_completion_record_has_unknown_producing_version(tmp_path):
    stage = Recorder('first')
    run(stage, outdir=tmp_path)
    path = tmp_path / 'first' / 'DONE.yml'
    record = yaml.safe_load(path.read_text())
    record.pop('version')
    path.write_text(yaml.safe_dump(record))
    run(stage, outdir=tmp_path)
    manifest = yaml.safe_load((tmp_path / 'MANIFEST.yml').read_text())
    assert manifest['stages']['first']['produced_version'] is None
    assert Recorder.runs == {'first': 1}


@pytest.mark.parametrize('fitopts,muopts', [(['DEFAULT'], []), ([], ['DEFAULT'])])
def test_systematics_can_use_default_label(fitopts, muopts):
    """DEFAULT remains available for SNANA-compatible covariance labels."""
    from jax_pipeline.systematics import check_labels

    check_labels('fit', fitopts, muopts)
