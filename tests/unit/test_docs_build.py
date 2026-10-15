"""The Sphinx documentation must build with no warnings.

Catches broken cross-references, malformed MyST directives, and config fields
that lose their `#:` description (which `configtable` reports as a warning).

The build runs in its own process. Autodoc documents every package, and in
doing so copies the annotations it finds inside an `__init__` onto the class
itself, which `model_input_keys` would then read as declared input keys; in
the same process that would break every test of the model classes that
followed.
"""

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("sphinx.cmd.build")

DOCS = Path(__file__).resolve().parents[2] / "docs"


def test_docs_build_clean(tmp_path):
    # Local documentation checks must not depend on remote inventories.
    script = """
import sys
from sphinx.application import Sphinx

source, output, doctrees = sys.argv[1:]
app = Sphinx(source, source, output, doctrees, "html",
             confoverrides={"intersphinx_mapping": {}}, warningiserror=True)
app.build()
sys.exit(app.statuscode)
"""
    result = subprocess.run([
        sys.executable, "-c", script, str(DOCS), str(tmp_path / "html"),
        str(tmp_path / "doctrees"),
    ], capture_output=True, text=True)
    assert result.returncode == 0, (
        "sphinx build failed:\n" + result.stdout[-4000:] + result.stderr[-4000:])

    bbc_html = (tmp_path / "html" / "bias_correction_keys.html").read_text()
    for spelling in ("nzbin", "opt_biascor", "varname_pIa", "uM0", "typeIa_ccprior"):
        assert f">{spelling}</span>" in bbc_html
    assert ">NZBIN</span>" not in bbc_html
    populations = (tmp_path / "html" / "supernova_populations.html").read_text()
    assert 'class="math notranslate nohighlight"' in populations
    assert '$\\Delta t$' not in populations

    from importlib import import_module
    unsupported = (tmp_path / "html" / "unsupported_input_keys.html").read_text()
    for package in ("jax_snlc_sim", "jax_snlc_fit", "jax_salt2mu",
                    "jax_create_cov", "jax_wfit"):
        parser = import_module(f"{package}.config_parser")
        for key in parser._REJECTED_KEYS:
            assert f">{key}</span>" in unsupported, (package, key)
    fit_parser = import_module("jax_snlc_fit.config_parser")
    for suffix in fit_parser._FITWIN_REFUSED:
        assert f">FITWIN_{suffix}</span>" in unsupported


def test_key_tables_follow_source_changes(tmp_path):
    """Reuse doctrees across CLI builds while changing imported declarations."""
    source = tmp_path / "docs"
    source.mkdir()
    (source / "conf.py").write_text(
        f"import sys\nsys.path[:0] = {[str(DOCS.parent), str(DOCS / '_ext'), str(tmp_path)]!r}\n"
        "extensions = ['configtable']\nmaster_doc = 'index'\n")
    pages = {
        "index": "Reference\n=========\n\n.. toctree::\n\n   config\n   model\n   refused\n   explicit\n",
        "config": "Configuration\n=============\n\n.. configtable:: declaration.Example\n"
                  "   :field-keys:\n   :keys-from: parser_fixture\n   :aliases-from: aliases_fixture\n",
        "model": "Model\n=====\n\n.. modeltable:: model_fixture.ExampleModel\n",
        "refused": "Refused\n=======\n\n.. rejectedkeys:: parser_fixture.REFUSED\n",
        "explicit": "Explicit keys\n=============\n\n.. configtable:: declaration.NamedSettings\n"
                    "   :field-keys:\n   :explicit-keys:\n",
    }
    for name, text in pages.items():
        (source / f"{name}.rst").write_text(text)
    modules = {
        "declaration": "from dataclasses import dataclass\n\n@dataclass\nclass Example:\n"
                       "    count: int = 1  #: Original count description.\n"
                       "    limit: int = 10  #: Maximum number of objects.\n"
                       "\n@dataclass\nclass NamedSettings:\n"
                       "    count: int = 1  #: SNANA OLD_COUNT; Event count.\n"
                       "    python_only: int = 2  #: Named setting without a SNANA spelling.\n",
        "key_specs": "FIELDS = {'count': ('COUNT', int)}\n",
        "parser_fixture": "from declaration import Example\nfrom key_specs import FIELDS\n\n"
                          "def parse() -> 'Example':\n"
                          "    return _supplied(raw=None, spec=FIELDS)\n\n"
                          "REFUSED = {'OLD': 'Original rejection reason.'}\n",
        "aliases_fixture": "_ALIASES = dict(COUNT_ALIAS='COUNT')\n",
        "model_fixture": "class ExampleModel:\n"
                         "    scale: float = 1.0  #: Original model description.\n",
    }
    for name, text in modules.items():
        (tmp_path / f"{name}.py").write_text(text)

    def build():
        result = subprocess.run([
            sys.executable, "-m", "sphinx", "-b", "html", "-W",
            "-d", str(tmp_path / "doctrees"), str(source), str(tmp_path / "html"),
        ], capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr

    def html(page):
        return (tmp_path / "html" / f"{page}.html").read_text()

    build()
    assert "Original count description." in html("config")
    assert "COUNT_ALIAS" in html("config")
    assert "LIMIT" in html("config")
    assert "Maximum number of objects." in html("config")
    assert "Original rejection reason." in html("refused")
    assert "Original model description." in html("model")
    assert "Python / YAML" in html("explicit")
    assert ">count</span>" in html("explicit")
    assert "OLD_COUNT" in html("explicit")
    assert "python_only" in html("explicit")
    assert "PYTHON_ONLY" not in html("explicit")
    assert "—" in html("explicit")

    # Change one dependency at a time, without touching the reStructuredText.
    # Different source lengths invalidate Python's bytecode cache even if two
    # writes fall in the same second; CLI processes also refresh sys.modules.
    changes = [
        ("declaration", "1  #: Original count description.",
         "222  #: Updated count description.", "config",
         ("Updated count description.", ">222<")),
        ("aliases_fixture", "COUNT_ALIAS", "UPDATED_COUNT_ALIAS", "config",
         ("UPDATED_COUNT_ALIAS",)),
        ("parser_fixture", "Original rejection reason.",
         "Updated parser rejection reason.", "refused",
         ("Updated parser rejection reason.",)),
        ("key_specs", "COUNT", "TOTAL_COUNT", "config", ("TOTAL_COUNT",)),
        ("model_fixture", "1.0  #: Original model description.",
         "222.0  #: Updated model description.", "model",
         ("Updated model description.", ">222.0<")),
    ]
    for module, old, new, page, expected in changes:
        path = tmp_path / f"{module}.py"
        path.write_text(path.read_text().replace(old, new))
        build()
        rendered = html(page)
        for text in expected:
            assert text in rendered, f"{page} did not refresh after {module} changed"
