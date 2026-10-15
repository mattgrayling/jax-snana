"""Unit tests for UsedTrackingDict — the shared config key-usage tracker."""

from jax_snlc_core.utils.config_tracking import UsedTrackingDict


def test_getitem_records_use():
    d = UsedTrackingDict({"A": 1, "B": 2})
    assert d["A"] == 1
    assert d.used == {"A"}


def test_contains_records_use():
    d = UsedTrackingDict({"A": 1})
    assert "A" in d
    assert "MISSING" not in d
    assert d.used == {"A", "MISSING"}


def test_get_records_use_including_default():
    d = UsedTrackingDict({"A": 1})
    assert d.get("A") == 1
    assert d.get("B", 99) == 99
    assert d.used == {"A", "B"}


def test_unlooked_keys_are_not_marked_used():
    d = UsedTrackingDict({"A": 1, "B": 2, "C": 3})
    _ = d["A"]
    unused = set(d) - d.used
    assert unused == {"B", "C"}


def test_add_occurrence_reads_as_last_wins():
    d = UsedTrackingDict()
    d.add_occurrence("A", 1)
    d.add_occurrence("A", 2)
    assert d["A"] == 2
    assert d.all("A") == [1, 2]


def test_all_merges_spellings_in_file_order():
    d = UsedTrackingDict()
    d.add_occurrence("NON1A", "row1")
    d.add_occurrence("NONIA", "row2")
    d.add_occurrence("NON1A", "row3")
    assert d.all("NON1A", "NONIA") == ["row1", "row2", "row3"]
    assert {"NON1A", "NONIA"} <= d.used


def test_assignment_replaces_every_occurrence():
    d = UsedTrackingDict()
    d.add_occurrence("A", 1)
    d.add_occurrence("A", 2)
    d["A"] = 9
    assert d.all("A") == [9]
    d.pop("A")
    assert d.all("A") == []


def test_copy_construction_preserves_occurrences():
    d = UsedTrackingDict()
    d.add_occurrence("A", 1)
    d.add_occurrence("A", 2)
    probe = UsedTrackingDict(d)
    assert probe["A"] == 2
    assert probe.all("A") == [1, 2]
    assert probe.used == {"A"}


def test_update_and_delete_replace_occurrences():
    d = UsedTrackingDict()
    d.add_occurrence("A", 1)
    d.update({"A": 2})
    assert d.all("A") == [2]
    del d["A"]
    assert d.all("A") == []
