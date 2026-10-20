"""Batch boundaries, shuffle, reset, and repeated epochs."""

import numpy as np

from mydezero import DataLoader
from tests.test_datasets import TinyDataset


def test_ordered_batches_include_tail_and_repeat_next_epoch():
    loader = DataLoader(TinyDataset(), batch_size=2, shuffle=False)
    for _ in range(2):
        batches = list(loader)
        assert [len(t) for x, t in batches] == [2, 2, 1]
        np.testing.assert_array_equal(np.concatenate([t for x, t in batches]), np.arange(5))
        np.testing.assert_array_equal(np.concatenate([x for x, t in batches]), np.arange(10).reshape(5, 2))


def test_shuffle_uses_fresh_permutation_each_epoch(monkeypatch):
    orders = iter([np.array([4, 2, 0, 3, 1]), np.array([1, 3, 0, 2, 4]), np.arange(5)])
    monkeypatch.setattr(np.random, 'permutation', lambda n: next(orders))
    loader = DataLoader(TinyDataset(), batch_size=3)
    for expected in ([4, 2, 0, 3, 1], [1, 3, 0, 2, 4]):
        batches = list(loader)
        np.testing.assert_array_equal(np.concatenate([t for x, t in batches]), expected)
        for x, t in batches:
            np.testing.assert_array_equal(x[:, 0], t * 2)


def test_reset_and_next_alias():
    loader = DataLoader(TinyDataset(), batch_size=10, shuffle=False)
    x, t = loader.next()
    assert x.shape == (5, 2) and t.shape == (5,)
    loader.reset()
    np.testing.assert_array_equal(loader.next()[1], t)


def test_empty_dataset_and_unlabeled_batches():
    data = TinyDataset()
    data.data = data.data[:0]
    data.label = data.label[:0]
    assert list(DataLoader(data, 2)) == []
    data = TinyDataset()
    data.label = None
    x, t = next(DataLoader(data, 2, shuffle=False))
    np.testing.assert_array_equal(t, [None, None])
