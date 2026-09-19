import numpy as np
from mydezero import Function, Variable


def numerical_diff(f, x, eps = 1e-4) :
    """Numerical differentiation of a function with respect to its parameter.

    Args:
        f (Function()): Function to be differentiated.
        x (Variable): Variable to be differentiated.
        eps (float): Step size used for numerical differentiation. Defaults to 1e-4.

    Returns:
        np.ndarray: Differentiated value.
    """

    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)
