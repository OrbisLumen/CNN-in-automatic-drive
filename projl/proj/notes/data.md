# Datasets, Transforms, and Mini-Batches

## Dataset and MNIST

`Dataset(train=True, transform=None, target_transform=None)` calls the subclass's
`prepare()` during initialization. Set `self.data` and optional `self.label` there.
`len(dataset)` returns the sample count. `dataset[index]` accepts a scalar index,
applies transforms, and returns `(sample, label)`, or `(sample, None)` without labels.
A None transform leaves its input unchanged.

```python
import numpy as np
from mydezero import Dataset, DataLoader

class TinyDataset(Dataset):
    def prepare(self):
        self.data = np.arange(12, dtype=np.float32).reshape(6, 2)
        self.label = np.array([0, 1, 0, 1, 0, 1])

loader = DataLoader(TinyDataset(), batch_size=4, shuffle=False)
assert [len(t) for x, t in loader] == [4, 2]
```

`mydezero.datasets.MNIST(train=True)` selects 60,000 training images;
`train=False` selects 10,000 test images. The first use downloads Google-hosted
MNIST GZIP files to `~/.mydezero`; later calls reuse the cache.
HTTPS requires valid Python CA certificates.

Initialization decompresses the full split into memory:

| Item | Raw storage | Default `dataset[index]` output |
| --- | --- | --- |
| Images | `uint8`, shape `(N, 1, 28, 28)` | `float32`, shape `(784,)`, range `[0, 1]` |
| Labels | `uint8`, shape `(N,)` | Class index from 0 to 9 |

Raw training and test pixels use about 44.9 MiB and 7.5 MiB, plus temporary
memory during decompression. Raw arrays are NumPy views of immutable bytes and
are normally read-only. Default transforms create floating-point samples on access;
they do not convert the full storage to float32. Copy raw data before modifying it.

```python
from mydezero.datasets import MNIST

train = MNIST(train=True)
x, label = train[0]
assert x.shape == (784,)
train.show(row=2, col=5)  # Ten random raw grayscale images; repeats are allowed.
assert MNIST.labels()[3] == '3'

raw = MNIST(train=False, transform=None)
assert raw[0][0].shape == (1, 28, 28)
```

`show()` bypasses sample transforms and omits labels.
`labels()` maps integer classes to character names.

## Sample Transforms

Import from `mydezero.transforms`:

| Transform | Usage |
| --- | --- |
| `Compose(transforms=[])` | Runs transforms in order; an empty list returns the input unchanged |
| `Flatten()` | Flattens to a 1D array |
| `AsType(dtype=np.float32)` | Converts NumPy dtype |
| `ToFloat(dtype=np.float32)` | Alias for AsType |
| `ToInt(dtype=int)` | Converts to platform integers by default; accepts another integer dtype |
| `Normalize(mean=0, std=1)` | `(array - mean) / std`; std must be nonzero. Sequence arguments broadcast by channel along the first axis |

Convert images to floating point before normalization to avoid integer truncation
or overflow. Channel normalization expects `(C, H, W)` and should precede Flatten.

```python
from mydezero.transforms import Compose, Flatten, Normalize, ToFloat

transform = Compose([ToFloat(), Normalize([0.5], [0.5]), Flatten()])
# These statistics assume [0, 1] input; raw MNIST pixels are in [0, 255].
mnist_transform = Compose([Flatten(), ToFloat(), Normalize(0, 255)])
```

## DataLoader and Evaluation

`DataLoader(dataset, batch_size, shuffle=True)` returns `(x, t)` NumPy arrays,
including a partial final batch. Use a positive integer batch_size.
`shuffle=False` preserves order; otherwise a random permutation is generated
at initialization and after exhaustion.

Exhaustion calls `reset()`, so the loader can be reused across epochs.
Breaking early leaves the iterator at its current position; call `loader.reset()`
to restart. `loader.next()` equals `next(loader)`. Empty datasets yield no batches.
Unlabeled data returns object label arrays containing None, unsuitable for supervised loss.

```python
import mydezero
import mydezero.functions as F
from mydezero.models import MLP
from mydezero.optimizers import SGD

model = MLP((16, 2))
optimizer = SGD(lr=0.1).setup(model)
loader = DataLoader(TinyDataset(), batch_size=4, shuffle=True)
for epoch in range(2):
    for x, t in loader:
        loss = F.softmax_cross_entropy(model(x), t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

correct = total = 0
with mydezero.no_grad():
    for x, t in loader:
        correct += float(F.accuracy(model(x), t).data) * len(t)
        total += len(t)
print(correct / total)
```

Weight loss and accuracy by actual batch size, then divide by the evaluation
sample count. Do not use the training count for test metrics.
See the [MNIST example](../examples/neural_network/train_mnist_mlp.ipynb).

## Download Cache

`get_file` expands `~` in cache_dir and returns an absolute path. It supports nested
directories and custom file names. The default name comes from the URL path,
excluding query and fragment. Provide a valid URL filename or explicit file_name.

Existing files are reused without checking source, size, or contents. Identical
names share a cache path; use distinct file_name or cache_dir values when needed.
Failure or Ctrl+C removes the partial file and re-raises the original exception.
Downloads cannot resume. `show_progress` displays a percentage, or bytes if the
total is unknown, capped at 100% for a partial final block.
