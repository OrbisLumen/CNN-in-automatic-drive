"""Dataset hooks and MNIST parsing from local IDX/GZIP fixtures."""

import gzip
import struct
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from mydezero.datasets import Dataset, MNIST
import mydezero.datasets as datasets


class TinyDataset(Dataset):
    def prepare(self):
        self.data = np.arange(10).reshape(5, 2)
        self.label = np.arange(5)


def test_dataset_prepares_and_transforms_samples_without_mutating_storage():
    data = TinyDataset(train=False, transform=lambda x: x + 10,
                       target_transform=lambda t: t * 2)
    assert len(data) == 5 and data.train is False
    x, t = data[2]
    np.testing.assert_array_equal(x, [14, 15])
    assert t == 4
    np.testing.assert_array_equal(data.data[2], [4, 5])


def test_dataset_identity_and_unlabeled_samples():
    data = TinyDataset()
    np.testing.assert_array_equal(data[0][0], [0, 1])
    assert data[0][1] == 0
    data.label = None
    assert data[0][1] is None


@pytest.fixture
def mnist_files(tmp_path, monkeypatch):
    images = np.zeros((2, 1, 28, 28), dtype=np.uint8)
    images[1] = 255
    labels = np.array([3, 7], dtype=np.uint8)
    for prefix in ('train', 't10k'):
        with gzip.open(tmp_path / f'{prefix}-images-idx3-ubyte.gz', 'wb') as f:
            f.write(struct.pack('>IIII', 2051, 2, 28, 28) + images.tobytes())
        with gzip.open(tmp_path / f'{prefix}-labels-idx1-ubyte.gz', 'wb') as f:
            f.write(struct.pack('>II', 2049, 2) + labels.tobytes())
    requests = []

    def local_file(url):
        requests.append(url)
        return str(tmp_path / Path(url).name)

    monkeypatch.setattr(datasets, 'get_file', local_file)
    return images, labels, requests


@pytest.mark.parametrize('train,prefix', [(True, 'train'), (False, 't10k')])
def test_mnist_reads_correct_split_and_transforms_on_access(mnist_files, train, prefix):
    images, labels, requests = mnist_files
    data = MNIST(train=train)
    np.testing.assert_array_equal(data.data, images)
    np.testing.assert_array_equal(data.label, labels)
    x, t = data[1]
    assert x.shape == (784,) and x.dtype == np.float32 and t == 7
    np.testing.assert_array_equal(x, np.ones(784))
    assert all(Path(url).name.startswith(prefix) for url in requests)
    assert len(requests) == 2
    assert data.data.dtype == np.uint8


def test_mnist_custom_transforms_and_labels(mnist_files):
    data = MNIST(transform=None, target_transform=lambda t: int(t) + 1)
    assert data[0][0].shape == (1, 28, 28)
    assert data[0][1] == 4
    assert MNIST.labels() == {i: str(i) for i in range(10)}


def test_mnist_show_can_select_last_image(mnist_files, monkeypatch):
    data = MNIST()
    # With one remaining image, the old exclusive upper bound was zero.
    data.data = data.data[-1:]
    imshow = Mock()
    monkeypatch.setattr(datasets.plt, 'imshow', imshow)
    monkeypatch.setattr(datasets.plt, 'axis', Mock())
    monkeypatch.setattr(datasets.plt, 'show', Mock())
    data.show(row=2, col=3)
    image = imshow.call_args.args[0]
    assert image.shape == (56, 84)
    np.testing.assert_array_equal(image, np.full((56, 84), 255))
