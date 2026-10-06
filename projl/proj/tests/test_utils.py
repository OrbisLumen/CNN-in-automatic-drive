import numpy as np
from pathlib import Path
from mydezero import Variable, no_grad
from mydezero.functions import goldstein
from mydezero.utils import get_dot_graph, plot_dot_graph

PROJECT_ROOT = Path(__file__).resolve().parents[1]


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

def test_plot_dot_graph_with_goldstein():

    x = Variable(np.array(1.0))
    y = Variable(np.array(1.0))
    z = goldstein(x, y)
    z.backward()

    x.name = 'x'
    y.name = 'y'
    z.name = 'z'

    to_file = 'goldstein.png'
    plot_dot_graph(z, verbose=False, to_file = to_file)

    print(f"Graph generated at: {(PROJECT_ROOT / 'draft' / to_file).resolve()}")
