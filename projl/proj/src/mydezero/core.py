"""Variables, computation graphs, and differentiable arithmetic on NumPy arrays."""

import contextlib
import heapq
import weakref

import numpy as np

import mydezero


# =============================================================================
# Config
# =============================================================================

class Config:
    """Stores global configuration options for the framework.

    This class is used as a centralized namespace for runtime settings
    that control framework behavior.

    Attributes:
        enable_backprop (bool): Whether function calls record a computation graph.
    """
    enable_backprop = True


@contextlib.contextmanager
def using_config(name, value):
    """Temporarily set a global option and restore it even if the block raises.

    Args:
        name (str): Name of an existing attribute on Config.
        value (object): Value to use inside the context.

    Yields:
        None: Control to the context block with the option applied.

    Raises:
        AttributeError: If Config has no attribute with the given name.
    """
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def no_grad():
    """Create a context that disables recording of computation graphs.

    Returns:
        contextlib.AbstractContextManager: Context restoring the previous setting
            on exit. Use it as ``with no_grad():`` during inference.
    """
    return using_config('enable_backprop', False)


# =============================================================================
# Variable, Function
# =============================================================================

class Variable:
    """Represents a variable in a computational graph.

    A Variable wraps numerical data and stores information required for
    automatic differentiation, including gradients and the function that
    created it.

    Args:
        data (np.ndarray or None): Array to wrap, or None for an uninitialized
            parameter or an absent optional input. Scalars must be arrays.
        name (str or None): Optional name used in graph labels.

    Attributes:
        data (np.ndarray or None): Wrapped numerical data.
        name (str or None): Optional variable name.
        grad (Variable or None): Accumulated gradient. A Variable allows gradients
            themselves to be differentiated when create_graph is enabled.
        creator (Function or None): Function that produced this variable.
        generation (int): Graph depth used to order backward operations.
        shape (tuple[int, ...]): Shape of the initialized data.
        ndim (int): Number of data dimensions.
        size (int): Number of data elements.
        dtype (np.dtype): Data type of the initialized data.

    Raises:
        TypeError: If data is neither a NumPy array nor None.
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
        """Record the producing function and derive this variable's graph depth.

        Args:
            func (Function): Function whose forward pass produced this variable.
        """
        self.creator = func
        self.generation = func.generation + 1

    def backward(self, retain_grad=False, create_graph=False):
        """Accumulate gradients through the graph in decreasing generation order.

        If no output gradient is supplied, start with an array of ones. Calling
        backward repeatedly accumulates gradients; use cleargrad to reset them.

        Args:
            retain_grad (bool): Keep intermediate and output gradients when True.
                By default, only leaf gradients remain after propagation.
            create_graph (bool): Record gradient operations to support higher-order
                differentiation. Defaults to False.
        """
        # When being the backward startpoint, initializing the grad
        if self.grad is None:
            self.grad = Variable(np.ones_like(self.data))

        funcs = []
        seen_set = set()

        def add_func(f):
            """Add a function to the local priority queue.

            Uses negative generation for max-priority behavior.
            ``id(f)`` is used as a tie-breaker when two functions have the same generation,
            avoiding direct comparison between Function objects.

            Args:
                f (Function or None): Function to enqueue once, ignoring None.
            """
            if f is not None and f not in seen_set:
                heapq.heappush(funcs, (-f.generation, id(f), f))
                seen_set.add(f)

        add_func(self.creator)

        while funcs:
            _, _, f = heapq.heappop(funcs)
            gys = [output().grad for output in f.outputs]

            with using_config('enable_backprop', create_graph):
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
        """Reset the accumulated gradient before another backward pass."""
        self.grad = None

    def reshape(self, *shape):
        """Reshape the data while preserving its gradient connection.

        Args:
            *shape (int or tuple[int, ...] or list[int]): Target dimensions, passed
                separately or as one sequence. One dimension may be -1.

        Returns:
            Variable: Reshaped variable, or self if the shape already matches.
        """
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return mydezero.functions.reshape(self, shape)

    def transpose(self, *axes):
        """Permute data axes while preserving its gradient connection.

        Args:
            *axes (int or tuple[int, ...] or list[int] or None): Axis permutation,
                passed separately or as one sequence. Omission reverses all axes.

        Returns:
            Variable: Variable with the requested axis order.
        """
        if len(axes) == 0:
            axes = None
        elif len(axes) == 1:
            if isinstance(axes[0], (tuple, list)) or axes[0] is None:
                axes = axes[0]
        return mydezero.functions.transpose(self, axes)

    def sum(self, axis=None, keepdims=False):
        """Sum elements along the requested axes.

        Args:
            axis (int or tuple[int, ...] or None): Axes to reduce; None reduces all.
            keepdims (bool): Retain reduced axes as dimensions of size one.

        Returns:
            Variable: Sum with a gradient connection to this variable.
        """
        return mydezero.functions.sum(self, axis, keepdims)

    @property
    def T(self):
        """Variable: Transpose with all axes reversed, following NumPy's T."""
        return mydezero.functions.transpose(self)


def as_array(x):
    """Convert a Python or NumPy scalar to a zero-dimensional array.

    Args:
        x (object): Value to normalize.

    Returns:
        np.ndarray or object: Array for a scalar; otherwise the original object.
    """
    if np.isscalar(x):
        return np.array(x)
    return x


def as_variable(obj):
    """Wrap an array or None, preserving an existing Variable.

    Args:
        obj (Variable or np.ndarray or None): Value to wrap.

    Returns:
        Variable: Existing variable or a new wrapper around obj.

    Raises:
        TypeError: If obj is not a Variable, NumPy array, or None.
    """
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


class Parameter(Variable):
    """Mark a Variable as trainable so Layer can collect it for optimization.

    Accepts the same arguments as Variable. Data may initially be None to defer
    weight initialization until the first forward pass.
    """


class Function:
    """Base class for differentiable functions.

    Subclasses must implement `forward` and `backward`.

    Attributes:
        inputs (list[Variable]): Inputs retained when graph recording is enabled.
        outputs (list[weakref.ReferenceType]): Weak references to output variables,
            preventing reference cycles with their creator.
        generation (int): Maximum input generation when graph recording is enabled.
    """

    def __call__(self, *inputs):
        """Run the forward pass and record graph connections when enabled.

        Args:
            *inputs (Variable or np.ndarray or None): Inputs to wrap as variables.

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
        """Compute output data; subclasses define their own input signatures.

        Args:
            xs (np.ndarray or None): Input data. Subclasses may accept multiple
                inputs and None for optional data such as a missing bias.

        Returns:
            np.ndarray | tuple[np.ndarray, ...]: Output array(s).

        Raises:
            NotImplementedError: If a subclass does not implement the forward pass.
        """
        raise NotImplementedError

    def backward(self, gys):
        """Compute input gradients; subclasses define their output signatures.

        Args:
            gys (Variable): Output gradient. Subclasses with multiple outputs
                accept one argument per output.

        Returns:
            Variable or tuple[Variable or None, ...]: One gradient per input;
                None may represent an absent optional input.

        Raises:
            NotImplementedError: If a subclass does not implement the backward pass.
        """
        raise NotImplementedError


# =============================================================================
# Arithmetic
# =============================================================================

class Add(Function):
    """Add two inputs and reduce broadcast gradients to their original shapes."""

    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 + x1
        return y

    def backward(self, gy):
        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape:  # for broadcast
            gx0 = mydezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = mydezero.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1


def add(x0, x1):
    """Add two inputs with NumPy broadcasting.

    Args:
        x0 (Variable or np.ndarray): Left operand.
        x1 (Variable or np.ndarray or scalar): Right operand.

    Returns:
        Variable: Elementwise sum.
    """
    x1 = as_array(x1)
    return Add()(x0, x1)


class Mul(Function):
    """Multiply inputs and reduce broadcast gradients to their original shapes."""

    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy * x1
        gx1 = gy * x0
        if x0.shape != x1.shape:  # for broadcast
            gx0 = mydezero.functions.sum_to(gx0, x0.shape)
            gx1 = mydezero.functions.sum_to(gx1, x1.shape)
        return gx0, gx1


def mul(x0, x1):
    """Multiply two inputs elementwise with NumPy broadcasting.

    Args:
        x0 (Variable or np.ndarray): Left operand.
        x1 (Variable or np.ndarray or scalar): Right operand.

    Returns:
        Variable: Elementwise product.
    """
    x1 = as_array(x1)
    return Mul()(x0, x1)


class Neg(Function):
    """Negate an input and its incoming gradient."""

    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy


def neg(x):
    """Negate an input elementwise.

    Args:
        x (Variable or np.ndarray): Input to negate.

    Returns:
        Variable: Elementwise negative of x.
    """
    return Neg()(x)


class Sub(Function):
    """Subtract inputs and reduce broadcast gradients to their original shapes."""

    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 - x1
        return y

    def backward(self, gy):
        gx0, gx1 = gy, -gy
        if self.x0_shape != self.x1_shape:  # for broadcast
            gx0 = mydezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = mydezero.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1


def sub(x0, x1):
    """Subtract the right operand from the left with NumPy broadcasting.

    Args:
        x0 (Variable or np.ndarray): Left operand.
        x1 (Variable or np.ndarray or scalar): Right operand.

    Returns:
        Variable: Elementwise difference x0 - x1.
    """
    x1 = as_array(x1)
    return Sub()(x0, x1)


def rsub(x0, x1):
    """Implement reflected subtraction for Variable.__rsub__.

    Args:
        x0 (Variable): Variable on the right of the subtraction operator.
        x1 (Variable or np.ndarray or scalar): Left operand.

    Returns:
        Variable: Elementwise difference x1 - x0.
    """
    x1 = as_array(x1)
    return Sub()(x1, x0)


class Div(Function):
    """Divide inputs and reduce broadcast gradients to their original shapes."""

    def forward(self, x0, x1):
        y = x0 / x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        if x0.shape != x1.shape:  # for broadcast
            gx0 = mydezero.functions.sum_to(gx0, x0.shape)
            gx1 = mydezero.functions.sum_to(gx1, x1.shape)
        return gx0, gx1


def div(x0, x1):
    """Divide the left operand by the right with NumPy broadcasting.

    Args:
        x0 (Variable or np.ndarray): Numerator.
        x1 (Variable or np.ndarray or scalar): Denominator.

    Returns:
        Variable: Elementwise quotient x0 / x1.
    """
    x1 = as_array(x1)
    return Div()(x0, x1)


def rdiv(x0, x1):
    """Implement reflected division for Variable.__rtruediv__.

    Args:
        x0 (Variable): Variable on the right of the division operator.
        x1 (Variable or np.ndarray or scalar): Numerator on the left.

    Returns:
        Variable: Elementwise quotient x1 / x0.
    """
    x1 = as_array(x1)
    return Div()(x1, x0)


class Pow(Function):
    """Raise an input to a constant exponent.

    Args:
        exponent (int or float): Constant power; it is not differentiated.

    Attributes:
        exponent (int or float): Power used in the forward and backward passes.
    """

    def __init__(self, exponent):
        self.exponent = exponent

    def forward(self, x):
        y = x ** self.exponent
        return y

    def backward(self, gy):
        x, = self.inputs
        exponent = self.exponent
        gx = exponent * x ** (exponent - 1) * gy
        return gx


def pow(x, c):
    """Raise an input elementwise to a constant power.

    Args:
        x (Variable or np.ndarray): Base values.
        c (int or float): Constant exponent.

    Returns:
        Variable: Elementwise power x ** c.
    """
    return Pow(c)(x)


def setup_variable():
    """Bind arithmetic operators to Variable when the package is imported."""
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
