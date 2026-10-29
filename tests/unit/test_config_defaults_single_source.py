"""A key's default must live on its dataclass field and nowhere else.

The parser used to restate every default in three further places, so a fix
applied to one layer was silently overridden by another (HOSTLIB_GALID_PRIORITY,
GENMODEL_MSKOPT). These tests fail if a second default reappears.
"""

import ast
import dataclasses
import glob
import pickle
from pathlib import Path

import jax_snlc_sim.simulation as config_module
from jax_snlc_sim import inputs as config_parser
from jax_snlc_sim.builder import build_simulation


def _parser_ast():
    return ast.parse(open(config_parser.__file__).read())


def _claimed_keys(tree):
    """Every input key a dataclass field claims, from the `_supplied` specs."""
    keys = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == '_supplied' and len(node.args) == 2):
            continue
        spec = node.args[1]
        if not isinstance(spec, ast.Dict):   # the helper's own `_supplied(raw, spec)`
            continue
        for entry in spec.values:
            key = entry.elts[0]
            if isinstance(key, ast.Constant):
                keys.add(key.value)
            elif isinstance(key, ast.Subscript):
                owner = getattr(config_module, key.value.value.id)
                keys.update(owner.input_names[ast.literal_eval(key.slice)])
            else:
                keys.update(e.value for e in key.elts)
    return keys


def _is_literal(node):
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return all(_is_literal(e) for e in node.elts)
    if isinstance(node, ast.Dict):
        return all(_is_literal(v) for v in node.values)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return _is_literal(node.operand)
    return False


def test_parser_does_not_restate_a_field_default():
    """No `<config>.get(KEY, <literal>)` for a key a dataclass field owns.

    A second lookup as the fallback (`get(NEW, get(OLD))`) is an alias chain,
    not a default, so it is allowed.
    """
    tree = _parser_ast()
    claimed = _claimed_keys(tree)
    offenders = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'get' and len(node.args) == 2):
            continue
        source = ast.unparse(node.func.value)
        # `raw` is keyed by SNANA key, `supplied` by field name — a literal
        # fallback on either restates a default the field already owns
        if source not in ('raw', 'supplied'):
            continue
        key = node.args[0]
        if not isinstance(key, ast.Constant):
            continue
        if source == 'raw' and key.value not in claimed:
            continue
        if _is_literal(node.args[1]):
            offenders.append(f"line {node.lineno}: {ast.unparse(node)}")
    assert not offenders, (
        "these keys are defaulted in the parser as well as on their dataclass "
        "field; delete the parser copy and let the field own it:\n  "
        + "\n  ".join(offenders))


def test_every_claimed_key_reaches_a_field_with_a_default():
    """Each spec entry names a real field, and that field declares a default —
    otherwise omitting the key from the input would raise instead of default."""
    tree = _parser_ast()
    missing = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id.endswith('Config')):
            continue
        cls = getattr(config_module, node.func.id, None)
        if cls is None or not dataclasses.is_dataclass(cls):
            continue
        defaulted = {
            f.name for f in dataclasses.fields(cls)
            if f.default is not dataclasses.MISSING
            or f.default_factory is not dataclasses.MISSING
        }
        for kw in node.keywords:
            spec = kw.value
            if not (kw.arg is None and isinstance(spec, ast.Call)
                    and isinstance(spec.func, ast.Name)
                    and spec.func.id == '_supplied' and len(spec.args) == 2):
                continue
            args = spec.args[1]
            if not isinstance(args, ast.Dict):
                continue
            for name in args.keys:
                if name.value not in defaulted:
                    missing.append(f"{node.func.id}.{name.value}")
    assert not missing, (
        "spec entries whose field is absent or has no default: "
        + ", ".join(sorted(missing)))


def test_omitted_settings_use_declared_defaults(tmp_path, fixture_root):
    source = fixture_root / 'snana/atmosphere_replay/input.input'
    required = {'GENMODEL', 'RANSEED', 'GENRANGE_REDSHIFT', 'SIMLIB_FILE',
                'KCOR_FILE', 'GENVERSION', 'GENFILTERS', 'GENRANGE_PEAKMJD',
                'GENRANGE_TREST', 'NGENTOT_LC', 'SOLID_ANGLE', 'GENSOURCE'}
    lines = [line for line in source.read_text().splitlines()
             if line.split(':')[0].strip() in required]
    minimal = tmp_path / 'minimal.input'
    minimal.write_text('\n'.join(lines) + '\nOPT_MWEBV: 0\n')
    config = config_parser.parse_snana_input_file(str(minimal))
    assert config.host.galid_priority == config_module.Host().galid_priority
    assert config.model.genmodel_mskopt == config_module.Model().genmodel_mskopt
    assert config.mw_dust.sigma_ratio == config_module.MWDust().sigma_ratio
    assert config.mw_dust.scale == config_module.MWDust().scale

    with minimal.open('a') as stream:
        stream.write('HOSTLIB_GALID_PRIORITY: 1,10\nGENMODEL_MSKOPT: 16\n'
                     'GENSIGMA_MWEBV_RATIO: 0.07\nGENSCALE_MWEBV: 0.9\n')
    explicit = config_parser.parse_snana_input_file(str(minimal))
    assert explicit.host.galid_priority == '1,10'
    assert explicit.model.genmodel_mskopt == 16
    assert explicit.mw_dust.sigma_ratio == 0.07
    assert explicit.mw_dust.scale == 0.9


# ---------------------------------------------------------------------------
# One vocabulary: the parser's key namespace is SNANA's, and the only untyped
# thing that leaves the config is the dict the models read.
# ---------------------------------------------------------------------------


def test_no_invented_key_names():
    """Every key the parser reads or writes is a SNANA key.

    Intermediate names of the parser's own invention ('_take_spectrum',
    '_non1a_entries', ...) used to travel alongside the real ones, so a reader
    could not tell which spelling a value lived under.
    """
    tree = _parser_ast()
    invented = set()
    for node in ast.walk(tree):
        key = None
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in ('get', 'setdefault', 'pop')
                and ast.unparse(node.func.value) == 'raw' and node.args):
            key = node.args[0]
        elif (isinstance(node, ast.Subscript)
              and ast.unparse(node.value) == 'raw'):
            key = node.slice
        if isinstance(key, ast.Constant) and str(key.value).startswith('_'):
            invented.add(key.value)
    assert not invented, (
        "these are the parser's own names, not SNANA's; pool the values under "
        "the SNANA key they came from: " + ", ".join(sorted(invented)))


def _model_config_keys_read():
    """Every key the five model classes read from the dict they are handed.

    The dict arrives as ``config`` at construction and as ``snana_config`` in
    ``configure_generation``; both names are scanned.
    """
    dict_names = {'snana_config', 'config'}
    keys = set()
    for path in glob.glob('jax_snlc_core/models/*.py'):
        tree = ast.parse(open(path).read())
        for node in ast.walk(tree):
            key = None
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == 'get'
                    and ast.unparse(node.func.value) in dict_names
                    and node.args):
                key = node.args[0]
            elif (isinstance(node, ast.Subscript)
                  and ast.unparse(node.value) in dict_names):
                key = node.slice
            if isinstance(key, ast.Constant):
                keys.add(key.value)
    return keys


def _structured_keys_declared():
    """The dict keys the models' own structured_input returns build; the
    model hands itself these, so the static offer list need not carry them."""
    keys = set()
    for path in glob.glob('jax_snlc_core/models/*.py'):
        tree = ast.parse(open(path).read())
        for node in ast.walk(tree):
            if (isinstance(node, ast.FunctionDef)
                    and node.name == 'structured_input'):
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Dict):
                        keys.update(k.value for k in sub.keys
                                    if isinstance(k, ast.Constant))
    return keys


def test_model_config_carries_exactly_what_models_read():
    """`model_config()` is the models' whole view of the sim input.

    It exists because `jax_snlc_core` sits below the simulator and cannot take
    a `SimulationConfig`. Handing over more than the models read would let one
    of them quietly start depending on a key it never declared — which is how
    the whole-config dict grew in the first place.
    """
    offered = set(config_module.SimulationConfig().model_config())
    read = _model_config_keys_read() - _structured_keys_declared()
    assert offered == read, (
        f"model_config() offers {sorted(offered - read)} that no model reads, "
        f"and withholds {sorted(read - offered)} that some model reads without "
        f"declaring through structured_input")


def test_build_does_not_mutate_the_parse_result():
    """`build_simulation` reads the parse result and returns `LightCurveSimulator`;
    nothing in the build may write back onto the config. Pinned by value:
    the parsed config serialises identically before and after the build."""
    cfg = config_parser.parse_snana_input_file(
        'tests/fixtures/snana/salt3_k21_replay/input.input')
    before = pickle.dumps(cfg)
    build_simulation(cfg, 500)
    assert pickle.dumps(cfg) == before
