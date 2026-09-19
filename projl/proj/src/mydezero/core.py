import numpy as np


class Config:
    """Stores global configuration options for the framework.

    This class is used as a centralized namespace for runtime settings
    that control framework behavior.

    Attributes:

    """
    pass


class Variable:
    """Represents a variable in a computational graph.

    A Variable wraps numerical data and stores information required for
    automatic differentiation, including gradients and the function that
    created it.

    Attributes:
        data (np.ndarray): Numerical data stored in a numpy array.
        grad (np.ndarray): Numerical gradient stored in a numpy array when back propagated.
        creator (Function): Function used to create the variable.
    """

    def __init__(self, data):
        # check input for numpy.ndarray
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"{type(data)} is not supported, data must be a numpy array.")

        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        # When being the backward startpoint, initializing the grad
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f: Function = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)


class Function:
    """Base class differentiable functions.

    Subclasses must implement `forward` and `backward`.

    Attributes:
        input (Variable): The variable function received (storing).
        output (Variable): The variable function produced (storing).
    """

    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y))
        output.set_creator(self)  # save creator in output
        self.input = input
        self.output = output
        return output

    def forward(self, x):
        raise NotImplementedError

    def backward(self, gy):
        raise NotImplementedError

def as_array(x):
    """Converts a numpy scalar to a numpy array.

    Also check for python int and float and convert it to a numpy array.
    """
    if np.isscalar(x):
        return np.array(x)
    return x