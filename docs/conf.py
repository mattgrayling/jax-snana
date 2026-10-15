"""Sphinx configuration for the jax-snana documentation."""

import dataclasses
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent / "_ext"))

from sphinx.ext.napoleon.docstring import (  # noqa: E402
    _single_colon_regex, _xref_or_code_regex)

from configtable import stated_key  # noqa: E402

project = "jax-snana"
author = "Matthew Grayling"
copyright = "2026, Matthew Grayling"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx.ext.mathjax",
    "configtable",
]

exclude_patterns = ["_build"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["navigation.css"]

# Docstrings quote input keys and carry URLs whose paths hold '...', and a
# code example quotes strings; typographic replacement would mangle all three.
smartquotes = False

# Keys are documented as trailing `#:` comments on the config dataclasses, so
# autodoc must read them and keep declaration order rather than sorting.
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "member-order": "bysource",
}
# an Attributes: section otherwise renders as attribute directives that
# collide with the same attributes documented from the class body
napoleon_use_ivar = True

autodoc_typehints = "description"
# Link to source for definitions, without following re-exports or generated members.
viewcode_follow_imported_members = False
# a 64-field dataclass otherwise renders its whole constructor signature
# as a wall of text before the per-field entries
autodoc_class_signature = "separated"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

myst_enable_extensions = ["colon_fence", "deflist", "dollarmath"]


def _name_the_input_key(app, what, name, obj, options, lines):
    """A config field's ``SNANA <KEY>; text`` comment, as the key tables read
    it, opens the API entry by naming the key instead of carrying the prefix
    raw; and a dataclass without a docstring of its own, whose entry is the
    constructor signature, shows what each default factory makes.

    Runs before napoleon, which reads the first line of an attribute's
    docstring as ``type: description`` wherever it holds a colon and would
    file the opening words of a sentence under a Type heading; a first line
    which holds a colon is therefore broken at the word before it, so that
    every ``#:`` comment stays a description."""
    if not lines:
        return
    if what == "class" and isinstance(obj, type) and dataclasses.is_dataclass(obj):
        lines[:] = [_factory_values(obj, line) for line in lines]
        return
    if what not in ("attribute", "data", "property"):
        return
    stated = stated_key(lines[0])
    if stated:
        groups, text = stated
        keys = "; ".join(" or ".join(f"``{k}``" for k in group) for group in groups)
        text = text[0].upper() + text[1:] if text[:1].islower() else text
        lines[0] = f"Input key {keys}. {text}"
    lines[0:1] = _before_any_colon(lines[0])


def _before_any_colon(line):
    """``line`` as one line, or as two broken at the last space before the
    first colon napoleon would read as a type separator."""
    for i, piece in enumerate(_xref_or_code_regex.split(line)):
        if i % 2 or not _single_colon_regex.search(piece):
            continue
        at = sum(len(p) for p in _xref_or_code_regex.split(line)[:i])
        at += _single_colon_regex.search(piece).start()
        cut = line.rfind(" ", 0, at)
        if cut <= 0:
            return [line]
        return [line[:cut], line[cut + 1:]]
    return [line]


_FACTORY = re.compile(r"(\w+)(:[^=]*)?=\s*<factory>")


def _factory_values(cls, text):
    """``text`` with each ``name=<factory>`` of the dataclass ``cls`` replaced
    by the value the factory makes, ``{}`` or ``[]``."""
    factories = {f.name: f.default_factory for f in dataclasses.fields(cls)
                 if f.default_factory is not dataclasses.MISSING}
    return _FACTORY.sub(
        lambda m: f"{m.group(1)}{m.group(2) or ''}={factories[m.group(1)]()!r}"
        if m.group(1) in factories else m.group(0), text)


def _dataclass_factory_defaults(app, what, name, obj, options, signature,
                                return_annotation):
    """A dataclass field with a default factory shows the value the factory
    makes, ``{}`` or ``()``, in place of autodoc's ``<factory>``."""
    if not signature or "<factory>" not in signature:
        return None
    cls = obj if isinstance(obj, type) else None
    if cls is None and what == "method":
        # the separated __init__ of a dataclass, resolved by its dotted name
        module_name, _, rest = name.rpartition(".")
        module_name, _, class_name = module_name.rpartition(".")
        cls = getattr(sys.modules.get(module_name), class_name, None)
    if not (isinstance(cls, type) and dataclasses.is_dataclass(cls)):
        return None
    return _factory_values(cls, signature), return_annotation


def setup(app):
    app.connect("autodoc-process-docstring", _name_the_input_key, priority=400)
    app.connect("autodoc-process-signature", _dataclass_factory_defaults)
