import numpy as np
from mydezero import Variable


def numerical_diff(function, variable, eps=1e-4):
    """Approximate a function's derivative with a centered difference."""
    x0 = Variable(np.array(variable.data - eps))
    x1 = Variable(np.array(variable.data + eps))
    y0 = function(x0)
    y1 = function(x1)
    return (y1.data - y0.data) / (2 * eps)
