"""Composable transforms for individual NumPy samples."""

import numpy as np


class Compose:
    """Composes several transforms together.

    Args:
        transforms (list): list of transforms
    """

    def __init__(self, transforms=[]):
        self.transforms = transforms

    def __call__(self, img):
        if not self.transforms:
            return img
        for t in self.transforms:
            img = t(img)
        return img


# =============================================================================
# Transforms for NumPy ndarray
# =============================================================================
class Normalize:
    """Compute (array - mean) / std using nonzero standard deviations.

    Channel sequences broadcast along the first axis (channel-first data).
    Convert integer pixels to floating point before normalization.

    Args:
        mean (float or sequence): mean for all values or sequence of means for
         each channel.
        std (float or sequence): standard deviation for each channel.
    """

    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std

    def __call__(self, array):
        mean, std = self.mean, self.std

        if not np.isscalar(mean):
            mshape = [1] * array.ndim
            mshape[0] = len(self.mean)
            mean = np.array(self.mean, dtype=array.dtype).reshape(*mshape)
        if not np.isscalar(std):
            rshape = [1] * array.ndim
            rshape[0] = len(self.std)
            std = np.array(self.std, dtype=array.dtype).reshape(*rshape)
        return (array - mean) / std


class Flatten:
    """Flatten a NumPy array.
    """

    def __call__(self, array):
        return array.flatten()


class AsType:
    """Return an array converted to dtype (float32 by default).

    ToFloat is an alias for this class and accepts the same dtype argument.
    """

    def __init__(self, dtype=np.float32):
        self.dtype = dtype

    def __call__(self, array):
        return array.astype(self.dtype)


ToFloat = AsType


class ToInt(AsType):
    """Convert an array to an integer dtype (platform int by default)."""

    def __init__(self, dtype=int):
        self.dtype = dtype
