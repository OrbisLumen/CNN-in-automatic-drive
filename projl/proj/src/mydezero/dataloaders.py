"""NumPy mini-batches with optional shuffling and automatic epoch reset."""

import math
import random
import numpy as np


class DataLoader:
    """Iterate over a dataset in batches, including the final partial batch.

    Args:
        dataset (Dataset): Indexable collection of (sample, label) pairs.
        batch_size (int): Positive number of samples per batch.
        shuffle (bool): Shuffle indices initially and after each exhausted epoch.

    Iteration returns two NumPy arrays. Unlabeled samples produce an object
    label array containing None. Exhausting the iterator resets it for the next
    epoch; stopping early requires reset to restart from the first batch.
    """

    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_size = len(dataset)
        self.max_iter = math.ceil(self.data_size / batch_size)

        self.reset()

    def reset(self):
        """Restart the epoch and regenerate shuffled or sequential indices."""
        self.iteration = 0
        if self.shuffle:
            self.index = np.random.permutation(len(self.dataset))
        else:
            self.index = np.arange(len(self.dataset))

    def __iter__(self):
        return self

    def __next__(self):
        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration

        i, batch_size = self.iteration, self.batch_size
        batch_index = self.index[i * batch_size:(i + 1) * batch_size]
        batch = [self.dataset[i] for i in batch_index]
        x = np.array([example[0] for example in batch])
        t = np.array([example[1] for example in batch])

        self.iteration += 1
        return x, t

    def next(self):
        """Return the next batch; alias for the iterator protocol's next call."""
        return self.__next__()
