import heapq
import weakref
import contextlib
import numpy as np


# =============================================================================
# Config
# =============================================================================

class Config:
    """Stores global configuration options for the framework.

    This class is used as a centralized namespace for runtime settings
    that control framework behavior.

    Attributes:
        enable_backprop (bool): Whether or not to enable backpropagation.
    """
    enable_backprop = True


@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


# =============================================================================
# Variable, Function
# =============================================================================

class Variable:
    """Represents a variable in a computational graph.

    A Variable wraps numerical data and stores information required for
    automatic differentiation, including gradients and the function that
    created it.

    Attributes:
        data (np.ndarray): Numerical data stored in a numpy array.
        name (str): Name of the variable.
        grad (np.ndarray): Numerical gradient stored in a numpy array when back propagated.
        creator (Function): Function used to create the variable.
        generation (int): the generation number of the variable in the backpropagation graph.
        shape (np.ndarray.shape): Shape of the variable.
        ndim (np.ndarray.ndim): Dimension of the variable.
        size (np.ndarray.size): Size of the variable.
        dtype (np.ndarray.dtype): Data type of the variable.
    """

    __array_priority__ = 200  # make Variable computation priority larger than ndarray

    def __init__(self, data, name=None):
        # check input for numpy.ndarray
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"{type(data)} is not supported, data must be a numpy array.")

        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def backward(self, retain_grad=False):
        # When being the backward startpoint, initializing the grad
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set = set()

        def add_func(f):
            """Add a function to the local priority queue.

            Uses negative generation for max-priority behavior.
            ``id(f)`` is used as a tie-breaker when two functions have the same generation,
            avoiding direct comparison between Function objects.
            """
            if f is not None and f not in seen_set:
                heapq.heappush(funcs, (-f.generation, id(f), f))
                seen_set.add(f)

        add_func(self.creator)

        while funcs:
            _, _, f = heapq.heappop(funcs)

            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None

    def cleargrad(self):
        self.grad = None


def as_array(x):
    """Converts a numpy scalar to a numpy array.

    Also check for python int and float and convert it to a numpy array.
    """
    if np.isscalar(x):
        return np.array(x)
    return x


def as_variable(obj):
    """Converts a numpy array to a Variable."""
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


class Function:
    """Base class differentiable functions.

    Subclasses must implement `forward` and `backward`.

    Attributes:
        inputs (Iterable[Variable]): The variables this function received (storing).
        outputs (Iterable[weakref.ref(Variable)]): The variable this function produced (storing).
        generation (int): the generation number of the function in the backpropagation graph, equal to the maximum of all the inputs' generation.
    """

    def __call__(self, *inputs):
        """
        Args:
            *inputs (Variable): The input variables.

        Returns:
            Variable | list[Variable]: A single output variable if there is only one
                output; otherwise, a list of output variables.
        """
        inputs = [as_variable(x) for x in inputs]

        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        # only auto graph when require backprop
        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)  # save creator in every output
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError

    def backward(self, gys):
        raise NotImplementedError


# =============================================================================
# Arithmetic
# =============================================================================

class Add(Function):
    """Computes addition of the two input."""

    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy


def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)


class Mul(Function):
    """Computes multiplication of the two input."""

    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0


def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)


class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy


def neg(x):
    return Neg()(x)


class Sub(Function):
    def forward(self, x0, x1):
        y = x0 - x1
        return y

    def backward(self, gy):
        return gy, -gy


def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)


def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0)


class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        return gx0, gx1


def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)


def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0)


class Pow(Function):
    def __init__(self, exponent):
        self.exponent = exponent

    def forward(self, x):
        y = x ** self.exponent
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        exponent = self.exponent
        gx = exponent * x ** (exponent - 1) * gy
        return gx


def pow(x, c):
    return Pow(c)(x)


def setup_variable():
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow
