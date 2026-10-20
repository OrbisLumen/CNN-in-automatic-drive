"""In-memory datasets and MNIST loading from cached IDX/GZIP files."""

import gzip
import numpy as np
from matplotlib import pyplot as plt

from mydezero.utils import get_file
from mydezero.transforms import Compose, Flatten, ToFloat, Normalize


class Dataset:
    """Prepare data once and transform individual samples on access.

    Subclasses implement prepare, assigning data and optionally label arrays.

    Args:
        train (bool): Select the training split when supported by the subclass.
        transform (Callable or None): Transform applied to each data sample.
        target_transform (Callable or None): Transform applied to each label.

    Attributes:
        data (np.ndarray or None): Raw samples populated by prepare.
        label (np.ndarray or None): Labels; None denotes unlabeled data.
    """

    def __init__(self, train=True, transform=None, target_transform=None):
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        if self.transform is None:
            self.transform = lambda x: x
        if self.target_transform is None:
            self.target_transform = lambda x: x

        self.data = None
        self.label = None
        self.prepare()

    def __getitem__(self, index):
        """Return a transformed (sample, label) pair for a scalar index."""
        assert np.isscalar(index)
        if self.label is None:
            return self.transform(self.data[index]), None
        else:
            return (self.transform(self.data[index]),
                    self.target_transform(self.label[index]))

    def __len__(self):
        return len(self.data)

    def prepare(self):
        """Populate raw data and labels; override in concrete datasets."""
        pass


# =============================================================================
# MNIST-like dataset: MNIST
# =============================================================================
class MNIST(Dataset):
    """Load the training or test split of handwritten digits into memory.

    Downloads are cached under ~/.mydezero. Raw data has shape (N, 1, 28, 28)
    and dtype uint8. The default transform returns a float32 vector of length
    784 scaled to [0, 1]. Pass transform=None to access raw images instead.
    Labels are uint8 class indices. Arguments follow Dataset.
    """

    def __init__(self, train=True, transform=Compose([Flatten(), ToFloat(),
                                                      Normalize(0., 255.)]),
                 target_transform=None):
        super().__init__(train, transform, target_transform)

    def prepare(self):
        url = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
        train_files = {'target': 'train-images-idx3-ubyte.gz',
                       'label': 'train-labels-idx1-ubyte.gz'}
        test_files = {'target': 't10k-images-idx3-ubyte.gz',
                      'label': 't10k-labels-idx1-ubyte.gz'}

        files = train_files if self.train else test_files
        data_path = get_file(url + files['target'])
        label_path = get_file(url + files['label'])

        self.data = self._load_data(data_path)
        self.label = self._load_label(label_path)

    def _load_label(self, filepath):
        with gzip.open(filepath, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
        return labels

    def _load_data(self, filepath):
        with gzip.open(filepath, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28)
        return data

    def show(self, row=10, col=10):
        """Display a row-by-col grid of raw images sampled with replacement."""
        H, W = 28, 28
        img = np.zeros((H * row, W * col))
        for r in range(row):
            for c in range(col):
                img[r * H:(r + 1) * H, c * W:(c + 1) * W] = self.data[
                    np.random.randint(len(self.data))].reshape(H, W)
        plt.imshow(img, cmap='gray', interpolation='nearest')
        plt.axis('off')
        plt.show()

    @staticmethod
    def labels():
        """Return the digit class index to display-name mapping."""
        return {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}
