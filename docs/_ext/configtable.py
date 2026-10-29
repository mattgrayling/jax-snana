"""Directives rendering the input keys from the code that owns them.

``configtable`` renders a config dataclass as grouped tables. ``modeltable``
renders the keys a model class declares as annotated attributes, which are the
keys the config dataclasses deliberately do not hold. ``rejectedkeys`` renders
the keys a parser refuses, with the reason it gives.

All three read their text from the source rather than repeating it here: the
dataclasses carry their grouping as ``# Section`` comments and their per-key
text as ``#:`` comments, and a model class comments its annotated attributes
the same way.

A config whose field names are not the input keys, and which holds fields no
input key sets, supplies the difference through two options::

    .. configtable:: jax_snlc_sim.simulation.Run
       :keys-from: jax_snlc_sim.inputs
       :omit-sections: Set by the program rather than by an input key

``keys-from`` names the parser module whose ``_supplied`` specifications map a
field to the input key or keys that set it, and ``omit-sections`` drops the
sections holding fields no input key reaches.

For configurations whose public field names are also accepted input keys,
``:field-keys:`` reads those names directly from the dataclass, in uppercase
for the SNANA reference tables. ``keys-from`` may also be given to add the
SNANA spellings described by the parser.

``:explicit-keys:`` with ``:field-keys:`` keeps the Python/YAML column but
lists only SNANA spellings stated in field descriptions. Fields without
a SNANA spelling show a dash in that column.

A field the parser reads through bespoke code states its key at the head of
its ``#:`` text, as ``SNANA <KEY>; text``. Several spellings of one key are
separated by ``/`` and render as alternatives (``GENTYPE or GENTYPES``);
several distinct keys which all set the field are separated by ``, `` and
render one to a line, so ``SNANA GENPEAK_<PAR>, GENSIGMA_<PAR>; text`` lists
two keys where ``SNANA GENTYPE/GENTYPES; text`` lists one. A key may carry
any character but whitespace, the comma, the slash and the semicolon, so
``HOSTLIB_SNR_SCALE(<MJDMIN>:<MJDMAX>)`` is a key. Spellings the parser
itself rewrites, declared in ``_KEY_ALIAS_GROUPS`` or ``_ALIASES``, are
appended to the key cell as alternatives without being written in the text.

A key that sets several fields is one row, in the section of the first, whose
description is each field's text in turn; the first field's text therefore
carries the whole description and the later fields' texts are short
continuations. Where every field carrying a value is a scalar the type and
default cells list them in order, as the successive values the key takes
(``CUTWIN_NEPOCH`` reads ``int, float``); otherwise the first field carrying a
value stands for the key. A ``bool`` field beside others is a presence flag
and carries no value of its own.

The Default cell reads *required* for a field with no default and for one
whose text carries the sentence ``Required.`` at any point, the sentence
itself being dropped, so a placeholder default the parser never lets stand
is not shown as the default; a sentence which starts ``Required`` and goes
on to say why stays in the text. A default of ``None``, ``''`` or an empty
collection reads *unset*, since all three mean the key was not given.

``modeltable`` takes the model class and the context asking, ``fit`` by
default, ``simulation`` for the keys the simulator reads, or ``bbc`` for the
keys the distance fit reads::

    .. modeltable:: jax_snlc_core.models.redback.RedbackModel
       :context: simulation

``rejectedkeys`` takes the parser's dict of refused key to reason, and with
``:group-by-reason:`` renders the keys refused for one reason as one row::

    .. rejectedkeys:: jax_snlc_sim.inputs._REJECTED_KEYS
       :group-by-reason:
"""

import ast
import importlib
import importlib.util
import inspect
import io
import re
import sys
import textwrap
import tokenize
import types
import typing
from collections.abc import Mapping
from dataclasses import MISSING, fields as dataclass_fields
from enum import Enum
from pathlib import PurePath

from docutils import nodes
from docutils.parsers.rst import directives
from docutils.statemachine import StringList
from sphinx.util import logging as sphinx_logging
from sphinx.util.docutils import SphinxDirective


logger = sphinx_logging.getLogger(__name__)

_SECTION = re.compile(r"^\s{4}#\s+(?P<title>[A-Z].*?)\s*$")
_TRAILING = re.compile(r"#:\s*(?P<text>.+?)\s*$")
_STANDALONE = re.compile(r"^\s{4}#:\s*(?P<text>.+?)\s*$")
_COMMENT = re.compile(r"^\s{4}#(?!:)")
_KEY_TOKEN = r"[^\s,;/]+"
_KEY_PREFIX = re.compile(
    rf"^SNANA\s+(?P<keys>{_KEY_TOKEN}(?:\s*[,/]\s*{_KEY_TOKEN})*)\s*;\s*(?P<text>.*)$")
# the bare sentence marking a key the input must give, and any sentence
# opening with the word, whether or not it goes on to say why
_REQUIRED_BARE = re.compile(r"(?:^|(?<=[.!?] ))Required\.(?: |$)")
_REQUIRED = re.compile(r"(?:^|(?<=[.!?] ))Required\b")
_SCALARS = (int, float, str, bool)
# a string annotation the module left unresolved, shown without its module
_MODULE_PREFIX = re.compile(r"\btyping\.|\bjax_\w+\.config\.")

# A heading names a group of fields, so it is short. One running to the width of
# the line is prose about a single field, which would otherwise render as a
# rubric with the field it describes filed underneath it.
_MAX_TITLE = 72


def _groups(source: str, sections: bool = True):
    """Split a class body into (section title, [(attribute, description)]).

    A description is a ``#:`` comment either trailing the attribute or on the
    line above it; Sphinx accepts both spellings.

    With ``sections``, a plain ``#`` comment opens a section, and since a
    heading is one line, a second plain ``#`` line beneath it is a description
    written with the wrong marker. A class that is mostly code rather than a
    declaration passes False: its ordinary comments are neither headings nor
    descriptions, and only ``#:`` says anything about a key.

    Docstrings are skipped, since a wrapped line of prose starting with a word
    and a colon otherwise reads as an attribute the class does not have.
    """
    source = textwrap.dedent(source)
    tree = ast.parse(source)
    declaration = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    attributes = {node.lineno: node for node in declaration.body
                  if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)}
    indent = next(iter(attributes.values())).col_offset if attributes else 4
    comments = {token.start[0]: token for token in
                tokenize.generate_tokens(io.StringIO(source).readline)
                if token.type == tokenize.COMMENT}
    collected = [(None, [])]
    pending = ""
    heading = False
    lines = source.splitlines()
    for lineno, line in enumerate(lines, 1):
        comment = comments.get(lineno)
        standalone = (comment is not None and comment.start[1] == indent
                      and not line[:indent].strip())
        # Normalise indentation for the comment conventions, not Python syntax.
        comment_line = "    " + comment.string if standalone else ""
        if _COMMENT.match(comment_line):
            if not sections:
                pending = ""
                continue
            if heading:
                raise ValueError(
                    "a section heading is one line, so what follows describes "
                    f"a field: spell it '#:' instead: {line.strip()!r}")
            section = _SECTION.match(comment_line)
            if section:
                title = section.group("title")
                if len(title) > _MAX_TITLE:
                    raise ValueError(
                        f"a heading of {len(title)} characters reads as a field "
                        f"description; spell it '#:' instead: {title!r}")
                collected.append((title, []))
                heading = True
            pending = ""
            continue
        heading = False
        description = _STANDALONE.match(comment_line)
        if description:
            text = description.group("text")
            pending = f"{pending} {text}" if pending else text
            continue
        field = attributes.get(lineno)
        if field:
            trailing = next((match for number in (field.lineno, field.end_lineno)
                             if number in comments and
                             (match := _TRAILING.match(comments[number].string))), None)
            collected[-1][1].append((
                field.target.id, trailing.group("text") if trailing else pending))
        pending = ""
    return [(title, entries) for title, entries in collected if entries]


def _default_value(field):
    """The field's default, or MISSING where it has none."""
    if field.default is not MISSING:
        return field.default
    if field.default_factory is not MISSING:
        return field.default_factory()
    return MISSING


def _unset(value) -> bool:
    """Whether a default says the key was not given."""
    return (value is None or value == ""
            or (isinstance(value, (tuple, list, dict, set, frozenset))
                and not value))


def _default_cell(value, required: bool = False) -> str:
    """A default as a table cell: *required* where the input must give it,
    *unset* where nothing was given, and the value otherwise."""
    if required or value is MISSING:
        return "*required*"
    if _unset(value):
        return "*unset*"
    if isinstance(value, Enum):
        return f"``{value.value!r}``"
    return f"``{value!r}``"


def _required(description: str):
    """Whether the text marks the key required, and the text with the bare
    ``Required.`` sentence dropped."""
    text = description or ""
    required = bool(_REQUIRED.search(text))
    text = " ".join(_REQUIRED_BARE.sub("", text).split())
    return required, text


def _key_names(value):
    """Validate metadata rather than silently rendering a partial key list."""
    names = [value] if isinstance(value, str) else value
    if (not isinstance(names, (tuple, list)) or not names
            or any(not isinstance(name, str) or not name for name in names)):
        raise ValueError(f"expected an input key or nonempty tuple/list of keys, got {value!r}")
    return list(names)


def _spec_keys(node, namespace, assignments):
    """Read a literal, local, or module-owned _supplied specification.

    Only metadata expressions are resolved. Parser bodies and converters are
    never executed to discover their keys. Unsupported expressions fail closed.
    """
    def value(expr, resolving=()):
        if isinstance(expr, ast.Name):
            if expr.id in resolving:
                raise ValueError(f"cyclic specification reference: {expr.id}")
            if expr.id in assignments:
                if assignments[expr.id] is None:
                    raise ValueError(f"ambiguous local specification binding: {expr.id}")
                return value(assignments[expr.id], (*resolving, expr.id))
            if expr.id in namespace:
                return namespace[expr.id]
        elif isinstance(expr, ast.Attribute):
            owner = value(expr.value, resolving)
            if isinstance(owner, types.ModuleType) or (isinstance(owner, type) and expr.attr == 'input_names'):
                return getattr(owner, expr.attr)
        elif isinstance(expr, ast.Subscript):
            mapping = value(expr.value, resolving)
            if isinstance(mapping, Mapping):
                return mapping[value(expr.slice, resolving)]
        elif isinstance(expr, (ast.Constant, ast.Tuple, ast.List)):
            if isinstance(expr, ast.Constant):
                return expr.value
            return tuple(value(part, resolving) for part in expr.elts)
        raise ValueError(f"cannot resolve specification expression {ast.unparse(expr)!r}")

    if isinstance(node, ast.Name) and node.id in assignments:
        if assignments[node.id] is None:
            raise ValueError(f"ambiguous local specification binding: {node.id}")
        # Keep a sentinel so cycles cannot fall back to a same-named global.
        remaining = {**assignments, node.id: None}
        return _spec_keys(assignments[node.id], namespace, remaining)
    if isinstance(node, ast.Dict):
        keys = {}
        for field, entry in zip(node.keys, node.values):
            if field is None:
                keys.update(_spec_keys(entry, namespace, assignments))
                continue
            name = value(field)
            if not isinstance(name, str):
                raise ValueError(f"specification field must be a string, got {name!r}")
            if not isinstance(entry, (ast.Tuple, ast.List)) or len(entry.elts) != 2:
                raise ValueError(f"{name}: expected (input key, converter)")
            keys[name] = _key_names(value(entry.elts[0]))
        return keys
    spec = value(node)
    if not isinstance(spec, Mapping):
        raise ValueError("_supplied specification must be a mapping")
    keys = {}
    for field, entry in spec.items():
        if not isinstance(field, str) or not isinstance(entry, (tuple, list)) or len(entry) != 2:
            raise ValueError(f"invalid _supplied entry {field!r}: expected (input key, converter)")
        keys[field] = _key_names(entry[0])
    return keys


def _function_nodes(body):
    """Walk this function without reading nested functions or classes."""
    pending = list(reversed(body.body))
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        yield node
        pending.extend(reversed(list(ast.iter_child_nodes(node))))


def _local_bindings(body, nodes):
    """Resolve single local assignments; reject modified metadata bindings."""
    assignments = {}
    targets = set()
    for statement in body.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
        elif isinstance(statement, ast.AnnAssign):
            target = statement.target
        else:
            continue
        if isinstance(target, ast.Name):
            targets.add(target)
            assignments[target.id] = None if target.id in assignments else statement.value
    for node in nodes:
        # Conditional writes and mutations cannot be resolved from an initializer.
        target = None
        if isinstance(node, (ast.Name, ast.Attribute, ast.Subscript)):
            if isinstance(node.ctx, (ast.Store, ast.Del)) and node not in targets:
                target = node
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            target = node.func.value
        while isinstance(target, (ast.Attribute, ast.Subscript)):
            target = target.value
        if isinstance(target, ast.Name):
            if not isinstance(node, ast.Call) or target.id in assignments:
                assignments[target.id] = None
    return assignments


def _parser_keys(module_name: str, config, note_dependency=None):
    """Discover _supplied metadata using the function's resolved return type.

    Function names, quoted/qualified return annotations, argument spelling,
    and module-level mappings or simple local bindings do not affect discovery.
    """
    module = importlib.import_module(module_name)
    keys = {name: _key_names(key)
            for name, key in getattr(module, "_FIELD_KEYS", {}).items()}
    for function in vars(module).values():
        if not inspect.isfunction(function):
            continue
        # Input annotations may refer to types imported only for type checking.
        result = inspect.get_annotations(function).get("return")
        if isinstance(result, str):
            result = eval(result, function.__globals__)
        if result is not config:
            continue
        if note_dependency:
            note_dependency(function)
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        body = next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
        nodes = list(_function_nodes(body))
        assignments = _local_bindings(body, nodes)
        for call in nodes:
            if not (isinstance(call, ast.Call) and
                    getattr(call.func, "id", getattr(call.func, "attr", "")) == "_supplied"):
                continue
            spec = call.args[1] if len(call.args) > 1 else next(
                (kw.value for kw in call.keywords if kw.arg == "spec"), None)
            if spec is None:
                raise ValueError(f"{function.__name__}: _supplied has no specification")
            try:
                keys.update(_spec_keys(spec, function.__globals__, assignments))
            except (ValueError, AttributeError) as error:
                raise ValueError(f"{module_name}.{function.__name__}: {error}") from error
    return keys


def _alias_groups(module_name: str):
    """Read the same alias declarations as the parser, regardless of syntax."""
    module = importlib.import_module(module_name)
    groups = {}
    for group in getattr(module, "_KEY_ALIAS_GROUPS", ()):
        canonical, *alternates = _key_names(group)
        groups.setdefault(canonical, []).extend(alternates)
    aliases = getattr(module, "_ALIASES", {})
    if not isinstance(aliases, Mapping):
        raise ValueError(f"{module_name}._ALIASES must be a mapping")
    for alternate, canonical in aliases.items():
        if (not isinstance(alternate, str) or not alternate
                or not isinstance(canonical, str) or not canonical):
            raise ValueError(f"{module_name}._ALIASES must map strings to strings")
        groups.setdefault(canonical, []).append(alternate)
    return groups


def stated_key(description: str):
    """The key groups a ``SNANA <KEY>; text`` description states, and the
    text after the prefix, or None where the text states no key.

    A group is the spellings of one key, as the ``/`` in the prefix
    separates them; the groups are the distinct keys the ``,`` separates.
    """
    stated = _KEY_PREFIX.match(description or "")
    if not stated:
        return None
    groups = [[k.strip() for k in group.split("/")]
              for group in stated.group("keys").split(",")]
    return groups, stated.group("text")


def _key(name: str, description: str, mapping):
    """The key groups for one field, and the description with any prefix cut.

    A field the parser reads through bespoke code rather than a ``_supplied``
    specification carries its key in the description, spelled
    ``SNANA <KEY>; text`` as the configs already write it. Cutting the prefix
    leaves the text opening mid-sentence, so it starts a sentence again, unless
    its first word is cased like a name such as ``zHOST``.
    """
    stated = stated_key(description)
    if stated:
        groups, text = stated
        return groups, _sentence(text)
    spellings = mapping.get(name)
    return ([spellings] if spellings else None), description


def _sentence(text: str) -> str:
    """``text`` with a leading plain lowercase word capitalised.

    A first word spelled as code, such as a literal value or an attribute name,
    is left as it is, since its case is what it means.
    """
    first = text.split(" ", 1)[0].rstrip(",;:.").replace("-", "")
    if first.isalpha() and first.islower():
        return text[0].upper() + text[1:]
    return text


def _alternatives(spellings) -> str:
    """The spellings of one key as a cell fragment: ``A``, ``A or B``, or
    ``A, B or C``."""
    coded = [f"``{k}``" for k in spellings]
    if len(coded) == 1:
        return coded[0]
    return ", ".join(coded[:-1]) + " or " + coded[-1]


def _key_cell(groups) -> str:
    """The key cell: one key's spellings as alternatives, several keys one
    to a line."""
    if not groups:
        return "—"
    lines = [_alternatives(group) for group in groups]
    if len(lines) == 1:
        return lines[0]
    return "\n".join(f"| {line}" for line in lines)


def _table(rows, field_names=False):
    """One list-table of (key, type, default, description) rows.

    Every description is a sentence, whichever config it was written in, so the
    reference pages read alike. A key cell of several lines is a line block,
    whose continuation lines sit at the cell's own indent.
    """
    lines = [
        ".. list-table::",
        "   :header-rows: 1",
        "   :widths: 22 24 14 40",
        "",
        "   * - Key",
        "     - Type",
        "     - Default",
        "     - Description",
    ]
    if field_names:
        lines[2] = "   :widths: 18 18 18 12 34"
        lines[4:5] = ["   * - Python / YAML", "     - SNANA key"]
    for cells in rows:
        if field_names:
            name, *cells = cells
            lines.append(f"   * - {name}")
        key, kind, default, description = cells
        first, *rest = key.split("\n")
        lines += [f"{'     -' if field_names else '   * -'} {first}"] + [f"       {line}" for line in rest] + [
            f"     - {kind}",
            f"     - {default}",
            f"     - {_sentence(description) if description else '*undocumented*'}",
        ]
    return lines + [""]


class _Row:
    """One table row, gathering the fields one key sets."""

    def __init__(self, groups):
        self.groups = [list(group) for group in groups]
        self.fields = []
        self.descriptions = []
        self.required = False

    def spellings(self):
        return {k for group in self.groups for k in group}

    def add(self, field, annotation, description, required):
        self.fields.append((field, annotation))
        if description:
            self.descriptions.append(description)
        self.required = self.required or required

    def merge_keys(self, groups):
        """Take in the spellings of another field naming a key of this row."""
        for group in groups:
            for known in self.groups:
                if set(group) & set(known):
                    known.extend(k for k in group if k not in known)
                    break
            else:
                self.groups.append(list(group))

    def cells(self, field_names=False):
        valued = [(f, a) for f, a in self.fields
                  if _scalar_of(a) is not bool] or self.fields
        if all(_scalar_of(a) is not None for _, a in valued):
            kind = ", ".join(_type_cell(a) for _, a in valued)
            default = ", ".join(_default_cell(_default_value(f), self.required)
                                for f, _ in valued)
        else:
            field, annotation = valued[0]
            kind = _type_cell(annotation)
            default = _default_cell(_default_value(field), self.required)
        description = " ".join(
            [self.descriptions[0]] + [_sentence(d) for d in self.descriptions[1:]]
        ) if self.descriptions else ""
        cells = (_key_cell(self.groups), kind, default, description)
        return (f"``{self.fields[0][0].name}``", *cells) if field_names else cells


def _note_dependencies(directive, *sources):
    """Rebuild this page when a declaration or parser it reads changes.

    Imports are refreshed by each Sphinx command; the dependency paths also
    invalidate doctrees retained from the preceding build.
    """
    directive.env.note_dependency(__file__)
    for source in sources:
        if source is None:
            continue
        if isinstance(source, str):
            source = importlib.import_module(source)
        declarations = source.__mro__[:-1] if isinstance(source, type) else (source,)
        for declaration in declarations:
            path = inspect.getsourcefile(declaration)
            if path:
                directive.env.note_dependency(path)
        # A parser can import its specifications or aliases from another module.
        # Track direct imports already loaded by Python, without executing any
        # additional imports or depending on where a dictionary was constructed.
        module = source if inspect.ismodule(source) else (
            inspect.getmodule(source) if inspect.isfunction(source) else None)
        if module is None:
            continue
        for node in ast.parse(inspect.getsource(module)).body:
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                name = "." * node.level + (node.module or "")
                base = importlib.util.resolve_name(name, module.__package__)
                names = [base, *(f"{base}.{alias.name}" for alias in node.names)]
            else:
                continue
            for name in names:
                imported = sys.modules.get(name)
                path = getattr(imported, "__file__", None)
                if path and path.endswith(".py"):
                    directive.env.note_dependency(path)


class ConfigTable(SphinxDirective):
    required_arguments = 1
    option_spec = {
        "keys-from": directives.unchanged_required,
        "field-keys": directives.flag,
        "explicit-keys": directives.flag,
        "field-key-case": lambda value: directives.choice(value, ("upper", "preserve")),
        "aliases-from": directives.unchanged_required,
        "omit-sections": directives.unchanged_required,
    }

    def run(self):
        module_name, _, class_name = self.arguments[0].rpartition(".")
        config = getattr(importlib.import_module(module_name), class_name)
        annotations = _resolved_annotations(config)
        explicit_keys = "explicit-keys" in self.options
        try:
            preserve_case = self.options.get("field-key-case") == "preserve"
            mapping = ({name: [name if preserve_case else name.upper()] for name in annotations
                        if not name.startswith("_")}
                       if "field-keys" in self.options and not explicit_keys else None)
            if "keys-from" in self.options:
                parser_keys = _parser_keys(
                    self.options["keys-from"], config,
                    lambda source: _note_dependencies(self, source))
                mapping = {**(mapping or {}), **parser_keys}
        except (ValueError, NameError) as failure:
            raise self.error(str(failure)) from failure
        alias_module = self.options.get("aliases-from",
                                        self.options.get("keys-from"))
        try:
            aliases = _alias_groups(alias_module) if alias_module else {}
            for name, spellings in getattr(config, 'input_names', {}).items():
                aliases.setdefault(name.upper(), []).extend(spellings)
        except ValueError as failure:
            raise self.error(str(failure)) from failure
        _note_dependencies(self, config, self.options.get("keys-from"), alias_module)
        omitted = [title.strip()
                   for title in self.options.get("omit-sections", "").split(",")
                   if title.strip()]

        try:
            grouped = _groups(inspect.getsource(config))
        except ValueError as failure:
            raise self.error(f"{self.arguments[0]}: {failure}")
        seen = {name for _, entries in grouped for name, _ in entries}
        missing = set(annotations) - seen
        if missing:
            raise self.error(
                f"{self.arguments[0]}: {len(missing)} field(s) were not found "
                f"in the source scan and would be omitted: {sorted(missing)}"
            )

        # a key that sets several fields is one row, in the section of the
        # first, whatever the fields answer to: two fields naming any
        # spelling in common are the same row
        field_names = "field-keys" in self.options
        sections = []
        rows = []
        undocumented = []
        unkeyed = []
        for title, entries in grouped:
            if title and any(title.startswith(prefix) for prefix in omitted):
                continue
            section = []
            for name, description in entries:
                if name.startswith("_"):
                    continue
                field, annotation = annotations[name]
                groups, description = _key(name, description, mapping or {})
                required, description = _required(description)
                if not description:
                    undocumented.append(name)
                if mapping is not None and not groups:
                    unkeyed.append(name)
                groups = _with_aliases(
                    groups or ([] if explicit_keys else [[name]]), aliases)
                spellings = {k for group in groups for k in group}
                row = None if field_names else next(
                    (r for r in rows if r.spellings() & spellings), None)
                if row is None:
                    row = _Row(groups)
                    rows.append(row)
                    section.append(row)
                else:
                    row.merge_keys(groups)
                row.add(field, annotation, description, required)
            sections.append((title, section))

        lines = []
        for title, section in sections:
            if not section:
                continue
            if title:
                lines += [f".. rubric:: {title}", ""]
            lines += _table((row.cells(field_names) for row in section), field_names)

        if undocumented:
            logger.warning(
                "%s: %d key(s) without a description: %s",
                self.arguments[0], len(undocumented), ", ".join(undocumented),
            )
        if unkeyed:
            logger.warning(
                "%s: %d field(s) with no input key; give each one a "
                "'SNANA <KEY>;' description: %s",
                self.arguments[0], len(unkeyed), ", ".join(unkeyed),
            )

        return _parse(self, lines)


def _with_aliases(groups, aliases):
    """Each key group extended by the spellings the parser rewrites to any
    key in it."""
    explicit = {key for group in groups for key in group}
    extended = []
    for group in groups:
        spellings = list(group)
        for key in group:
            spellings += [a for a in aliases.get(key, [])
                          if a not in spellings and a not in explicit]
        extended.append(spellings)
    return extended


def _resolved_annotations(config):
    """Each field with its annotation as an object, resolving the strings a
    module written under ``from __future__ import annotations`` leaves."""
    try:
        hints = typing.get_type_hints(config)
    except Exception:
        hints = {}
    return {f.name: (f, hints.get(f.name, f.type))
            for f in dataclass_fields(config)}


def _parse(directive, lines):
    """Render generated reStructuredText in place of the directive."""
    node = nodes.section()
    node.document = directive.state.document
    directive.state.nested_parse(
        StringList(lines), directive.content_offset, node)
    return node.children


class ModelTable(SphinxDirective):
    """The input keys a model class declares as annotated attributes.

    The declaration is authoritative, as it is for the whole codebase: a model
    owns its keys, and the general simulator and fitter must not name them.
    A key declared on a base class, whether the SED contract or one of the
    roles, is shared by every model that inherits it, so it belongs on that
    base's own table rather than repeated under each model.
    """

    required_arguments = 1
    option_spec = {"context": directives.unchanged_required}

    def run(self):
        module_name, _, class_name = self.arguments[0].rpartition(".")
        from jax_snlc_core.models import model_input_keys

        model = getattr(importlib.import_module(module_name), class_name)
        _note_dependencies(self, model, model_input_keys)
        context = self.options.get("context", "fit")
        consumer = {"fit": "fit", "simulation": "sim", "bbc": "bbc"}.get(context)
        if consumer is None:
            raise self.error(
                f"{self.arguments[0]}: context is 'fit', 'simulation' or "
                f"'bbc', not {context!r}")

        # a key declared on a base class, whether the SED contract or one of
        # the roles, is documented once on that base's own table
        declared = vars(model).get("__annotations__", {})
        keys = {k: v for k, v in model_input_keys(model, consumer=consumer).items()
                if v.attribute in declared}

        described = {name: text
                     for _, entries in _groups(inspect.getsource(model),
                                               sections=False)
                     for name, text in entries}

        rows = []
        undocumented = []
        for key, spec in keys.items():
            required, description = _required(described.get(spec.attribute, ""))
            if not description:
                undocumented.append(key)
            kind = f"``{spec.kind.__name__}``"
            if spec.allowed:
                kind += ", " + _alternatives(repr(v) for v in spec.allowed)
            rows.append((f"``{key}``", kind,
                         _default_cell(getattr(model, spec.attribute), required),
                         description))

        if undocumented:
            logger.warning(
                "%s: %d key(s) without a description: %s",
                self.arguments[0], len(undocumented), ", ".join(undocumented),
            )
        if not rows:
            return _parse(self, [
                f"{class_name} declares no input keys of its own."])
        return _parse(self, _table(rows))


def _scalar_of(annotation):
    """The scalar type an annotation names, ``None`` being unwrapped, or
    None where it names a collection or nothing so simple."""
    origin = typing.get_origin(annotation)
    if origin in (typing.Union, types.UnionType):
        members = [a for a in typing.get_args(annotation) if a is not type(None)]
        return _scalar_of(members[0]) if len(members) == 1 else None
    if origin is typing.Literal:
        return type(typing.get_args(annotation)[0])
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return str
    return annotation if annotation in _SCALARS else None


def _type_cell(annotation) -> str:
    """An annotation as a table cell, in one dialect whatever the source
    spelled: ``dict[str, float]``, ``str | list[str]``, and the values of a
    ``Literal`` or an ``Enum`` in place of their name. ``None`` is dropped,
    since the Default cell says *unset* where that is the default. A ``Path``
    is a ``path``, an array an ``array``, and a class is its bare name.
    """
    if isinstance(annotation, str):
        return f"``{_MODULE_PREFIX.sub('', annotation)}``"
    if isinstance(annotation, typing.ForwardRef):
        return f"``{annotation.__forward_arg__}``"
    if annotation is typing.Any:
        return "``Any``"
    if annotation is Ellipsis:
        return "``...``"
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin in (typing.Union, types.UnionType):
        members = [_type_cell(a) for a in args if a is not type(None)]
        return " | ".join(dict.fromkeys(members))
    if origin is typing.Literal:
        return _alternatives(repr(v) for v in args)
    if origin is not None:
        inner = ", ".join(_type_cell(a).strip("`") for a in args)
        name = _type_name(origin)
        return f"``{name}[{inner}]``" if inner else f"``{name}``"
    if isinstance(annotation, type):
        if issubclass(annotation, Enum):
            return _alternatives(repr(m.value) for m in annotation)
        return f"``{_type_name(annotation)}``"
    return f"``{annotation}``"


def _type_name(kind) -> str:
    """A class as the table spells it."""
    if isinstance(kind, type):
        if issubclass(kind, PurePath):
            return "path"
        if kind.__name__ == "ndarray":
            return "array"
        return kind.__name__
    return str(kind).replace("typing.", "").lower()


def _rejected(module_name: str, name: str):
    """The keys a parser refuses, each with the reason it refuses them with.

    Every parser declares them as one dict of key to reason, the same dict
    its abort message reads, so the table and the message cannot disagree.
    """
    table = getattr(importlib.import_module(module_name), name, None)
    if not isinstance(table, dict):
        raise ValueError(f"{module_name}.{name} is not a dict of key to reason")
    return [(key, reason or "") for key, reason in table.items()]


class RejectedKeys(SphinxDirective):
    """The keys a stage refuses, and why, from the parser that refuses them.

    The list is the parser's own, so a key added to it appears here and one
    removed from it disappears; a key refused without a reason is a build
    warning, since a refusal the page cannot explain is the failure the page
    exists to prevent. With ``group-by-reason`` the keys refused for one
    reason share a row, in the order the first of them appears.
    """

    required_arguments = 1
    option_spec = {"group-by-reason": directives.flag,
                   "key-prefix": directives.unchanged_required}

    def run(self):
        module_name, _, name = self.arguments[0].rpartition(".")
        _note_dependencies(self, module_name)
        try:
            rows = _rejected(module_name, name)
        except ValueError as failure:
            raise self.error(str(failure))
        unexplained = [key for key, reason in rows if not reason]
        if unexplained:
            logger.warning(
                "%s: %d refused key(s) with no reason beside them: %s",
                self.arguments[0], len(unexplained), ", ".join(unexplained))
        if "group-by-reason" in self.options:
            grouped = {}
            for key, reason in rows:
                grouped.setdefault(reason, []).append(key)
            rows = [(keys, reason) for reason, keys in grouped.items()]
        else:
            rows = [([key], reason) for key, reason in sorted(rows)]
        lines = [
            ".. list-table::",
            "   :header-rows: 1",
            "   :widths: 30 70",
            "",
            "   * - Key",
            "     - Reason or alternative",
        ]
        prefix = self.options.get("key-prefix", "")
        for keys, reason in rows:
            keys = [prefix + key for key in keys]
            lines += [f"   * - {', '.join(f'``{key}``' for key in keys)}",
                      f"     - {_sentence(reason) if reason else '*no reason recorded*'}"]
        return _parse(self, lines + [""])


def setup(app):
    app.add_directive("configtable", ConfigTable)
    app.add_directive("modeltable", ModelTable)
    app.add_directive("rejectedkeys", RejectedKeys)
    return {"env_version": 2, "parallel_read_safe": True}
