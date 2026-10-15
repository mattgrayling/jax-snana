"""Key-usage tracking for SNANA-style config parsers.

Shared by the simulator config parser and the SALT2mu/BBC config parser: a
dict that records which keys were looked up, so a parser can report input keys
it never consumed (typos, unsupported features).

Plain reads see the last occurrence of a key (SNANA's sequential overwrite);
:meth:`all` returns every occurrence of one or more spellings in file order,
for the keys that accumulate.
"""


class UsedTrackingDict(dict):
    """A dict that remembers which keys were looked up, so the parser can
    report input keys it never used (typos, unsupported features)."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.used = set()
        self._seq = 0
        self._occ = {}
        self._lines = []
        if args and isinstance(args[0], UsedTrackingDict):
            self._seq = args[0]._seq
            self._occ = {key: list(occ) for key, occ in args[0]._occ.items()}
            self._lines = list(args[0]._lines)
            super().update(args[0])
            self.update(**kwargs)
        else:
            self.update(*args, **kwargs)

    def __getitem__(self, key):
        self.used.add(key)
        return super().__getitem__(key)

    def __contains__(self, key):
        self.used.add(key)
        return super().__contains__(key)

    def get(self, key, default=None):
        self.used.add(key)
        return super().get(key, default)

    def add_occurrence(self, key, value):
        """Record one occurrence of ``key`` in file order; plain reads of the
        key see this value until a later occurrence or an assignment."""
        self._seq += 1
        self._occ.setdefault(key, []).append((self._seq, value))
        super().__setitem__(key, value)

    def __setitem__(self, key, value):
        # assignment replaces every occurrence
        self._seq += 1
        self._occ[key] = [(self._seq, value)]
        super().__setitem__(key, value)

    def pop(self, key, *default):
        self._occ.pop(key, None)
        return super().pop(key, *default)

    def __delitem__(self, key):
        self._occ.pop(key, None)
        super().__delitem__(key)

    def update(self, *args, **kwargs):
        for key, value in dict(*args, **kwargs).items():
            self[key] = value

    def record_line(self, key, text):
        """The input line that set ``key``, kept as read for the README."""
        self._lines.append((key, text))

    def lines(self):
        """Every recorded input line as (key, text), in reading order."""
        return list(self._lines)

    def all(self, *keys):
        """Every occurrence of the named key spellings, merged in file order."""
        self.used.update(keys)
        merged = [entry for key in keys for entry in self._occ.get(key, [])]
        return [value for _, value in sorted(merged, key=lambda e: e[0])]


def refused(reason, *keys):
    """A parser's refused-key table entry: every key, with the one reason."""
    return dict.fromkeys(keys, reason)
