"""Guard: no model-name dispatch scattered through the BBC subsystem.

Model differences live in the model classes' StandardisedModel role, resolved
once from MODEL_LCFIT by the model registry. Any ``if model == 'salt'`` /
``ISMODEL_`` conditional in ``jax_salt2mu`` is a leak of the exact pattern this
design avoids.
"""

import pathlib
import re

import jax_salt2mu

# config.py may spell "SALT2": it is BBCConfig's default MODEL_LCFIT
# (SALT2mu.c:6477), a value the registry resolves, not a branch on it.
_EXEMPT = {"config.py"}

_PATTERNS = [
    re.compile(r"""ISMODEL_"""),
    re.compile(r"""model_lcfit\s*=="""),
    re.compile(r"""['"]SALT2['"]"""),
    re.compile(r"""['"]BAYESN['"]"""),
]


def test_no_model_dispatch_outside_registry():
    pkg_dir = pathlib.Path(jax_salt2mu.__file__).parent
    offenders = []
    for path in pkg_dir.rglob("*.py"):
        if path.name in _EXEMPT:
            continue
        text = path.read_text()
        for pat in _PATTERNS:
            for m in pat.finditer(text):
                line = text[: m.start()].count("\n") + 1
                offenders.append(f"{path.name}:{line}: {pat.pattern}")
    assert not offenders, (
        "model-name dispatch leaked into the BBC package:\n" + "\n".join(offenders))
