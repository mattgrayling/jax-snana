"""Documentation metadata survives ordinary parser and declaration refactors."""

import importlib
import sys
import textwrap

import pytest

pytest.importorskip("sphinx")


@pytest.fixture
def extension():
    from docs._ext import configtable
    return configtable


@pytest.fixture
def parser(tmp_path):
    """Import a real Python source file through Python's normal module loader."""
    module_name = 'metadata_parser_' + tmp_path.name
    sys.path.insert(0, str(tmp_path))

    def create(source):
        path = tmp_path / f'{module_name}.py'
        path.write_text(textwrap.dedent(source))
        return importlib.import_module(module_name)

    try:
        yield create
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop(module_name, None)


def test_comments_follow_python_syntax(extension):
    source = '''
    class Example:
        \'\'\'A class description.
        bogus: int
        # Not a section
        \'\'\'
        # Controls
        #: Real field description.
        name: str = "#: not a comment"
        limit: int = (
            3
        )  #: Multi-line default description.

        def method(self):
            local: str = "not a field"
    '''
    assert extension._groups(source) == [
        ("Controls", [("name", "Real field description."),
                      ("limit", "Multi-line default description.")])]


def test_parser_metadata_refactors(extension, parser):
    module = parser('''
        from dataclasses import dataclass
        @dataclass
        class Example:
            count: int = 1
            scale: float = 1.
        Result = Example
        def declarations():
            raise AssertionError("Unrelated module functions must not be called")
        ALTERNATES = ("COUNT", "COUNT_OLD")
        BASE = {"count": (ALTERNATES, int)}
        FIELDS = {**BASE, "scale": ("SCALE", float)}
        def renamed(raw: "RuntimeOnlyInput") -> "Result":
            unused = 0
            unused = 1
            spec = {**FIELDS}
            return _supplied(raw=raw, spec=spec)
    ''')
    # _supplied is deliberately undefined: discovery must not execute the parser.
    assert extension._parser_keys(module.__name__, module.Example) == {
        "count": ["COUNT", "COUNT_OLD"], "scale": ["SCALE"]}


def test_runtime_alias_declarations(extension, parser):
    module = parser('''
        BASE = (("COUNT", "OLD_COUNT"),)
        _KEY_ALIAS_GROUPS = [*BASE, ("SCALE", "OLD_SCALE")]
        _ALIASES = dict(ANOTHER_COUNT="COUNT")
    ''')
    assert extension._alias_groups(module.__name__) == {
        "COUNT": ["OLD_COUNT", "ANOTHER_COUNT"], "SCALE": ["OLD_SCALE"]}


def test_unsupported_metadata_is_reported(extension, parser):
    module = parser('''
        class Example:
            pass
        def parse(raw) -> Example:
            return _supplied(raw, make_spec(raw))
    ''')
    with pytest.raises(ValueError, match=rf"{module.__name__}\.parse:.*make_spec"):
        extension._parser_keys(module.__name__, module.Example)


def test_malformed_alias_is_reported(extension, parser):
    module = parser('_KEY_ALIAS_GROUPS = (("COUNT", 12),)')
    with pytest.raises(ValueError, match="input key"):
        extension._alias_groups(module.__name__)


@pytest.mark.parametrize("body", [
    'spec = {"count": ("COUNT", int)}\n'
    'spec["count"] = ("UPDATED", int)',
    'spec = {"count": ("COUNT", int)}\n'
    'spec.update({"count": ("UPDATED", int)})',
    'spec = {"count": ("COUNT", int)}\n'
    'if raw:\n    spec = {"count": ("UPDATED", int)}',
    'spec = spec',
])
def test_modified_specs_are_rejected(extension, parser, body):
    source = (
        'class Example:\n    pass\n'
        'spec = {"count": ("GLOBAL", int)}\n'
        'def parse(raw) -> Example:\n'
        + textwrap.indent(body, '    ')
        + '\n    return _supplied(raw, spec)\n'
    )
    module = parser(source)
    with pytest.raises(ValueError, match="local specification binding: spec"):
        extension._parser_keys(module.__name__, module.Example)


def test_explicit_filter_keys_are_not_repeated_as_aliases(extension):
    groups = [["MJD_TEMPLATE"], ["MJD_TEMPLATE_FILTER"]]
    aliases = {"MJD_TEMPLATE": ["MJD_TEMPLATE_FILTER", "TEMPLATE_MJD"]}
    assert extension._with_aliases(groups, aliases) == [
        ["MJD_TEMPLATE", "TEMPLATE_MJD"], ["MJD_TEMPLATE_FILTER"]]


def test_bbc_documented_spellings_are_accepted(extension, tmp_path):
    from jax_salt2mu.config import BBCConfig
    from jax_salt2mu.config import parse_bbc_input

    mapping = extension._parser_keys("jax_salt2mu.config", BBCConfig)
    assert mapping["varname_pia"] == ["varname_pIa"]
    assert mapping["select_trueia"] == ["select_trueIa"]
    path = tmp_path / "bbc.input"
    path.write_text(
        "model_lcfit=SALT2\ndatafile=example.FITRES\nprefix=example\n"
        "nzbin=7\nzmin=0.05\nzmax=1.2\nopt_biascor=1\n"
        f"{mapping['varname_pia'][0]}=PROB_IA\n")
    config = parse_bbc_input(str(path))
    assert config.nzbin == 7
    assert config.opt_biascor == 1
    assert config.varname_pia == "PROB_IA"
