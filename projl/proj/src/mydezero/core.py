import heapq
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
        generation (int): the generation number of the variable in the backpropagation graph.
    """

    def __init__(self, data):
        # check input for numpy.ndarray
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"{type(data)} is not supported, data must be a numpy array.")

        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def backward(self):
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

            gys = [output.grad for output in f.outputs]
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

    def cleargrad(self):
        self.grad = None


class Function:
    """Base class differentiable functions.

    Subclasses must implement `forward` and `backward`.

    Attributes:
        inputs (Iterable[Variable]): The variables this function received (storing).
        outputs (Iterable[Variable]): The variable this function produced (storing).
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
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        self.generation = max([x.generation for x in inputs])
        for output in outputs:
            output.set_creator(self)  # save creator in every output
        self.inputs = inputs
        self.outputs = outputs
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError

    def backward(self, gys):
        raise NotImplementedError


def as_array(x):
    """Converts a numpy scalar to a numpy array.

    Also check for python int and float and convert it to a numpy array.
    """
    if np.isscalar(x):
        return np.array(x)
    return x
