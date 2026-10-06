"""Render a computation graph when the optional Graphviz executable is present."""

import shutil

import numpy as np
import pytest

from mydezero import Variable, utils
from mydezero.functions import goldstein


@pytest.mark.skipif(shutil.which('dot') is None, reason='Graphviz dot is not installed')
def test_plot_dot_graph_with_goldstein(tmp_path, monkeypatch):
    # The renderer derives its draft directory from the module's location.
    monkeypatch.setattr(utils, '__file__', str(tmp_path / 'src' / 'mydezero' / 'utils.py'))
    x = Variable(np.array(1.0), name='x')
    y = Variable(np.array(1.0), name='y')
    z = goldstein(x, y)
    z.name = 'z'
    z.backward()

    utils.plot_dot_graph(z, verbose=False, to_file='goldstein.png')

    draft = tmp_path / 'draft'
    assert (draft / 'goldstein.png').read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    assert (draft / 'temp.dot').read_text() == utils.get_dot_graph(z, verbose=False)
