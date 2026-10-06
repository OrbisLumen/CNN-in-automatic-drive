"""DOT graph generation without invoking the renderer."""

import numpy as np

from mydezero import Variable, no_grad
from mydezero.utils import get_dot_graph


def test_get_dot_graph_for_leaf_variable():
    x = Variable(np.array(1.0), name='x')
    graph = get_dot_graph(x)

    assert graph.startswith('digraph G {')
    assert 'label="x: () float64"' in graph
    assert '->' not in graph


def test_get_dot_graph_without_backprop():
    with no_grad():
        y = Variable(np.array(1.0)) + 2

    graph = get_dot_graph(y)
    assert str(id(y)) in graph
    assert '->' not in graph
