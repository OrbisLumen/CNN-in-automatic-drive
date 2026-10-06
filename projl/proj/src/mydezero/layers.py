"""Trainable layers and recursive parameter collection."""

import weakref

import numpy as np

from mydezero import Parameter
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

    def _init_W(self):
        """Initialize weights with standard deviation sqrt(1 / in_size)."""
        I, O = self.in_size, self.out_size
        W_data = np.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)
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
            self._init_W()

        y = F.linear(x, self.W, self.b)
        return y
