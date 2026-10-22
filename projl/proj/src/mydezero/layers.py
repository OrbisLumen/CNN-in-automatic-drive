"""Trainable layers and recursive parameter collection."""

import os
import weakref

import numpy as np
from mydezero import Parameter, cuda
import mydezero.functions as F


class Layer:
    """Base class for layers that own parameters or other layers.

    Assign Parameter and Layer objects directly to attributes to register them.
    Parameters stored only in ordinary lists or dictionaries are not collected.

    Attributes:
        inputs (list[weakref.ReferenceType]): Weak references to the latest inputs.
        outputs (list[weakref.ReferenceType]): Weak references to the latest outputs.
    """

    def __init__(self):
        self._params = set()

    def __setattr__(self, name, value):
        """Register parameters and child layers when assigning attributes.

        Args:
            name (str): Attribute name.
            value (object): Value to assign.
        """
        if isinstance(value, (Parameter, Layer)):
            self._params.add(name)
        super().__setattr__(name, value)

    def __call__(self, *inputs):
        """Run forward and retain weak references to the latest inputs and outputs.

        Args:
            *inputs (Variable or np.ndarray): Inputs accepted by the subclass.

        Returns:
            Variable or list[Variable]: Single output or a list of outputs.
        """
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, inputs):
        """Compute layer outputs; subclasses provide their own input signatures.

        Args:
            inputs (Variable or np.ndarray): Layer input. Subclasses may accept
                multiple positional inputs.

        Returns:
            Variable or tuple[Variable, ...]: One or more output variables.

        Raises:
            NotImplementedError: If a subclass does not implement forward.
        """
        raise NotImplementedError

    def params(self):
        """Iterate through registered parameters, including those in child layers.

        Traversal order is unspecified because attribute names are stored in a set.

        Yields:
            Parameter: A registered parameter from this layer or a child layer.
        """
        for name in self._params:
            obj = self.__dict__[name]

            if isinstance(obj, Layer):
                yield from obj.params()
            else:
                yield obj

    def cleargrads(self):
        """Reset gradients on all registered parameters recursively."""
        for param in self.params():
            param.cleargrad()

    def to_cpu(self):
        """Move all parameters to the CPU."""
        for param in self.params():
            param.to_cpu()

    def to_gpu(self):
        """Move all parameters to the GPU."""
        for param in self.params():
            param.to_gpu()

    def _flatten_params(self, params_dict, parent_key=""):
        """Collect parameters of this layer and its child layers into a dict.

        Nested layers contribute entries under names joined with ``'/'``,
        producing keys such as ``'l0/W'`` that stay unique across the model
        and match the keys written by `save_weights`.

        Args:
            params_dict (dict): Dictionary updated in place with parameter
                name keys and Parameter values.
            parent_key (str): Name prefix for nested layers; empty for the
                top-level layer.
        """
        for name in self._params:
            obj = self.__dict__[name]
            key = parent_key + '/' + name if parent_key else name

            if isinstance(obj, Layer):
                obj._flatten_params(params_dict, key)
            else:
                params_dict[key] = obj

    def save_weights(self, path):
        """Save all parameters as a compressed .npz archive.

        Parameter names are flattened with ``'/'`` separators (see
        `_flatten_params`) so nested layers are preserved. The layer is
        moved to the CPU before saving. If saving fails, a partially
        written file is removed and the error is re-raised.

        Args:
            path (str): File path to save to. Use the ``.npz`` extension so
                the same path can be passed to `load_weights`.
        """
        self.to_cpu()

        params_dict = {}
        self._flatten_params(params_dict)
        array_dict = {key: param.data for key, param in params_dict.items()
                      if param is not None}
        try:
            np.savez_compressed(path, **array_dict)
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(path):
                os.remove(path)
            raise

    def load_weights(self, path):
        """Load parameters saved by `save_weights` into this layer.

        Each parameter's data is replaced with the value stored under the
        matching flattened name, so the layer structure must match the
        saved archive. Loaded data is on the CPU; call `to_gpu()`
        afterwards when training on the GPU.

        Args:
            path (str): File path to a .npz archive written by
                `save_weights`.
        """
        npz = np.load(path)
        params_dict = {}
        self._flatten_params(params_dict)
        for key, param in params_dict.items():
            param.data = npz[key]

class Linear(Layer):
    """Fully connected layer with optional bias and deferred weight initialization.

    Args:
        out_size (int): Number of output features.
        nobias (bool): Omit the bias when True.
        dtype (np.dtype or type): Data type for initialized parameters.
        in_size (int or None): Number of input features. None defers weight
            initialization until the first forward pass.

    Attributes:
        in_size (int or None): Input width, inferred on the first pass if omitted.
        out_size (int): Output width.
        dtype (np.dtype or type): Parameter initialization data type.
        W (Parameter): Weight matrix of shape (in_size, out_size), initially
            holding None when initialization is deferred.
        b (Parameter or None): Bias vector of shape (out_size,), or None.
    """

    def __init__(self, out_size, nobias=False, dtype=np.float32, in_size=None):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype

        self.W = Parameter(None, name="W")
        if self.in_size is not None:
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_size, dtype=dtype), name="b")

    def _init_W(self, xp = np):
        """Initialize weights with standard deviation sqrt(1 / in_size).

        Args:
            xp (module): NumPy or CuPy module used to sample the weights on
                the device of the input.
        """
        I, O = self.in_size, self.out_size
        W_data = xp.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)
        self.W.data = W_data

    def forward(self, x):
        """Apply the affine transform, initializing weights once if needed.

        Args:
            x (Variable or np.ndarray): Input batch of shape (N, in_size).

        Returns:
            Variable: Output batch of shape (N, out_size).
        """
        if self.W.data is None:
            self.in_size = x.shape[1]
            xp = cuda.get_array_module(x)
            self._init_W(xp)

        y = F.linear(x, self.W, self.b)
        return y
