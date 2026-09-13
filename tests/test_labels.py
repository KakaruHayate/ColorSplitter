"""Cluster-label editing: the operations behind the WebUI's point/簇 editing."""

from __future__ import annotations

import numpy as np
import pytest

from colorsplitter.core import labels as L


def test_cluster_sizes_counts_points() -> None:
    assert L.cluster_sizes(np.array([0, 0, 1, 2, 2, 2])) == {0: 2, 1: 1, 2: 3}


def test_compact_renumbers_by_smallest_current_id() -> None:
    out = L.compact(np.array([7, 7, 3, 9]))
    assert out.tolist() == [1, 1, 0, 2]


def test_compact_is_stable_and_idempotent() -> None:
    labels = np.array([5, 5, 1])
    once = L.compact(labels)
    assert np.array_equal(L.compact(once), once)


def test_relabel_points_moves_only_the_selected_indices() -> None:
    labels = np.array([0, 0, 1, 1])
    out = L.relabel_points(labels, [0, 3], 5)
    assert out.tolist() == [5, 0, 1, 5]
    assert labels.tolist() == [0, 0, 1, 1], "input must not be mutated"


def test_relabel_points_rejects_out_of_range() -> None:
    with pytest.raises(IndexError):
        L.relabel_points(np.array([0, 1]), [9], 0)


def test_new_cluster_id_fills_the_first_gap() -> None:
    assert L.new_cluster_id(np.array([0, 2])) == 1
    assert L.new_cluster_id(np.array([0, 1, 2])) == 3
    assert L.new_cluster_id(np.array([-1, 0])) == 1


def test_split_cluster_moves_points_to_a_fresh_id() -> None:
    labels = np.array([0, 0, 1])
    out = L.split_cluster(labels, 0, [0])
    assert out.tolist() == [2, 0, 1]


def test_split_cluster_requires_points_to_be_in_the_source() -> None:
    with pytest.raises(ValueError):
        L.split_cluster(np.array([0, 1]), 0, [1])


def test_merge_clusters_folds_sources_into_target() -> None:
    out = L.merge_clusters(np.array([0, 1, 2, 3]), [1, 3], 0)
    assert out.tolist() == [0, 0, 2, 0]


def test_rename_cluster_merges_when_target_exists() -> None:
    out = L.rename_cluster(np.array([0, 1, 1]), 1, 0)
    assert out.tolist() == [0, 0, 0]


def test_remove_cluster_marks_points_as_noise_by_default() -> None:
    out = L.remove_cluster(np.array([0, 1, 1]), 1)
    assert out.tolist() == [0, -1, -1]
    assert L.cluster_sizes(out) == {-1: 2, 0: 1}


def test_remove_cluster_can_reassign_instead() -> None:
    out = L.remove_cluster(np.array([0, 1, 1]), 1, reassign_to=0)
    assert out.tolist() == [0, 0, 0]


def test_history_undo_redo_round_trip() -> None:
    history = L.LabelHistory(np.array([0, 0, 1]))
    assert not history.can_undo

    history.push(L.relabel_points(history.current, [0], 1))
    assert history.current.tolist() == [1, 0, 1]
    assert history.can_undo

    assert history.undo().tolist() == [0, 0, 1]
    assert history.can_redo
    assert history.redo().tolist() == [1, 0, 1]


def test_history_ignores_no_op_pushes() -> None:
    history = L.LabelHistory(np.array([0, 1]))
    history.push(np.array([0, 1]))
    assert not history.can_undo, "an unchanged state should not be undoable"


def test_history_push_clears_redo() -> None:
    history = L.LabelHistory(np.array([0, 1]))
    history.push(np.array([1, 1]))
    history.undo()
    assert history.can_redo
    history.push(np.array([0, 0]))
    assert not history.can_redo


def test_history_respects_its_limit() -> None:
    history = L.LabelHistory(np.array([0]), limit=3)
    for step in range(10):
        history.push(np.array([step + 1]))
    depth = 0
    while history.can_undo:
        history.undo()
        depth += 1
    assert depth == 3


def test_history_reset_clears_both_stacks() -> None:
    history = L.LabelHistory(np.array([0, 1]))
    history.push(np.array([1, 1]))
    history.reset(np.array([2, 2]))
    assert not history.can_undo
    assert not history.can_redo
    assert history.current.tolist() == [2, 2]


def test_edits_do_not_alias_the_input_array() -> None:
    """A shared buffer here would corrupt the undo stack in subtle ways."""
    original = np.array([0, 1, 2])
    snapshot = original.copy()
    for fn in (
        lambda a: L.relabel_points(a, [0], 9),
        lambda a: L.compact(a),
        lambda a: L.merge_clusters(a, [1], 0),
        lambda a: L.rename_cluster(a, 2, 0),
        lambda a: L.remove_cluster(a, 1),
    ):
        fn(original)
    assert np.array_equal(original, snapshot)
