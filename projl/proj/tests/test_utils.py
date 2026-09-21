import numpy as np
from pathlib import Path
from mydezero import Variable
from mydezero.functions import goldstein
from mydezero.utils import plot_dot_graph

PROJECT_ROOT = Path(__file__).resolve().parents[1]

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