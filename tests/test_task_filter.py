"""Provide test task filter functionality."""

from benchmark.utils.task_filter import (
    SELECTED_BENCHMARK_TASKS,
    is_selected_benchmark_task,
    should_skip_benchmark_task,
)


def test_selected_task_allowlist_contains_expected_17_entries():
    """Execute the test selected task allowlist contains expected 17 entries operation.

    Returns:
        Any: The operation result."""
    assert len(SELECTED_BENCHMARK_TASKS) == 17
    expected = (
        "nsclc_gnn_hoebel2026", "OS", "cv", "c_index"
    )
    assert expected in SELECTED_BENCHMARK_TASKS
    assert is_selected_benchmark_task(*expected)
    assert not should_skip_benchmark_task(*expected)


def test_non_selected_task_is_skipped():
    """Execute the test non selected task is skipped operation.

    Returns:
        Any: The operation result."""
    not_selected = ("nsclc_gnn_hoebel2026", "other", "cv", "c_index")
    assert not is_selected_benchmark_task(*not_selected)
    assert should_skip_benchmark_task(*not_selected)
