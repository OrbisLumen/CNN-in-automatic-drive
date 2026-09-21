from pathlib import Path
import subprocess
import numpy as np
from mydezero import Variable


# =============================================================================
# graphviz visualization
# =============================================================================
def _dot_var(v, verbose=False):
    dot_var = '{} [label="{}", color=orange, style=filled]\n'

    name = '' if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name += ': '
        name += str(v.shape) + ' ' + str(v.dtype)
    return dot_var.format(id(v), name)


def _dot_func(f):
    dot_func = '{} [label="{}", color=lightblue, style=filled, shape=box]\n'
    txt = dot_func.format(id(f), f.__class__.__name__)

    dot_edge = '{} -> {}\n'
    for x in f.inputs:
        txt += dot_edge.format(id(x), id(f))
    for y in f.outputs:
        txt += dot_edge.format(id(f), id(y()))  # y is weakref
    return txt


def get_dot_graph(output, verbose=True):
    txt = ''
    funcs = []
    seen_set = set()

    def add_func(f):
        if f not in seen_set:
            funcs.append(f)
            seen_set.add(f)

    add_func(output.creator)
    txt += _dot_var(output, verbose)

    while funcs:
        func = funcs.pop()
        txt += _dot_func(func)
        for x in func.inputs:
            txt += _dot_var(x, verbose)

            if x.creator is not None:
                add_func(x.creator)
    return 'digraph G {\n' + txt + '\n}'


def plot_dot_graph(output, verbose=True, to_file='graph.png'):

    dot_graph = get_dot_graph(output, verbose)

    project_root = Path(__file__).resolve().parents[2]
    draft_dir = project_root / "draft"
    draft_dir.mkdir(exist_ok=True)

    graph_path = draft_dir / "temp.dot"
    output_path = draft_dir / to_file

    graph_path.write_text(dot_graph)

    extension = output_path.suffix[1:]

    subprocess.run(['dot', str(graph_path), '-T' + extension, '-o', str(output_path)])

    # Return the image as a Jupyter Image object, to be displayed in-line.
    try:
        from IPython import display
        return display.Image(filename=str(output_path))
    except:
        pass