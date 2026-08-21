"""
Description: Unit tests for backend.utils.batching_utils.split_into_batches().
    Each test checks one specific behavior/edge case in isolation, so if
    this function ever breaks, we know exactly which case failed.
"""
from backend.utils.batching_utils import split_into_batches


def test_split_into_batches_even_split():
    """Normal case: list length is exactly divisible by batch_size."""
    items = [1, 2, 3, 4, 5, 6]
    result = split_into_batches(items, 2)
    assert result == [[1, 2], [3, 4], [5, 6]]


def test_split_into_batches_uneven_split():
    """Edge case: last batch is smaller than batch_size when it doesn't divide evenly."""
    items = [1, 2, 3, 4, 5]
    result = split_into_batches(items, 2)
    assert result == [[1, 2], [3, 4], [5]]


def test_split_into_batches_empty_list():
    """Edge case: an empty input list should give an empty list of batches, not crash."""
    result = split_into_batches([], 3)
    assert result == []


def test_split_into_batches_batch_size_bigger_than_list():
    """Edge case: if batch_size is bigger than the whole list, we just get one batch."""
    items = [1, 2]
    result = split_into_batches(items, 10)
    assert result == [[1, 2]]
