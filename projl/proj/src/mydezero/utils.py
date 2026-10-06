"""Graphviz visualization and array helpers for reduction gradients."""

from pathlib import Path
import subprocess

# =============================================================================
# graphviz visualization
# =============================================================================
def _dot_var(v, verbose=False):
    """Build a DOT node declaration for a variable.

    Args:
        v (Variable): Variable to represent.
        verbose (bool): Include shape and dtype when data is initialized.

    Returns:
        str: DOT declaration using the variable's identity as its node ID.
    """
    dot_var = '{} [label="{}", color=orange, style=filled]\n'

    name = '' if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name += ': '
        name += str(v.shape) + ' ' + str(v.dtype)
    return dot_var.format(id(v), name)


def _dot_func(f):
    """Build a DOT function node and edges to its inputs and outputs.

    Args:
        f (Function): Recorded function whose output weak references are alive.

    Returns:
        str: DOT declarations for the function and its connecting edges.
    """
    dot_func = '{} [label="{}", color=lightblue, style=filled, shape=box]\n'
    txt = dot_func.format(id(f), f.__class__.__name__)

    dot_edge = '{} -> {}\n'
    for x in f.inputs:
        txt += dot_edge.format(id(x), id(f))
    for y in f.outputs:
        txt += dot_edge.format(id(f), id(y()))  # y is weakref
    return txt


def get_dot_graph(output, verbose=True):
    """Generate DOT source for the computation graph ending at output.

    Each function is visited once. Graph rendering does not require the generation
    ordering used by backward. A leaf variable produces a graph with one node.

    Args:
        output (Variable): Final output whose ancestors will be traversed.
        verbose (bool): Include variable shapes and dtypes in node labels.

    Returns:
        str: Complete Graphviz DOT source.
    """
    txt = ''
    funcs = []
    seen_set = set()

    def add_func(f):
        if f is not None and f not in seen_set:
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
    """Render a computation graph with Graphviz for notebook display.

    DOT source is written to the shared draft/temp.dot file under the project.
    Calls overwrite that file. The output extension selects the Graphviz format.

    Args:
        output (Variable): Final output whose graph will be rendered.
        verbose (bool): Include variable shapes and dtypes in node labels.
        to_file (str or pathlib.Path): Output path including a Graphviz-supported
            extension. Relative paths are resolved under the draft directory.

    Returns:
        IPython.display.Image or None: Displayable image if IPython can load the
            rendered file; otherwise None.

    Raises:
        FileNotFoundError: If the Graphviz dot executable is unavailable.
    """

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


# =============================================================================
# utils function for numpy
# =============================================================================

def sum_to(x, shape):
    """Sum elements along axes to output an array of a given shape.

    Args:
        x (np.ndarray): Input array.
        shape (tuple[int, ...]): Target shape compatible with reducing x.

    Returns:
        np.ndarray: Reduced array with the requested shape.
    """
    ndim = len(shape)
    lead = x.ndim - ndim
    lead_axis = tuple(range(lead))

    axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
    y = x.sum(lead_axis + axis, keepdims=True)
    if lead > 0:
        y = y.squeeze(lead_axis)
    return y


def reshape_sum_backward(gy, x_shape, axis, keepdims):
    """Reshape gradient appropriately for mydezero.functions.sum's backward.

    Args:
        gy (Variable): Gradient variable from the output by backprop.
        x_shape (tuple[int, ...]): Shape used at sum function's forward.
        axis (int or tuple[int, ...] or None): Axes used at sum function's
            forward.
        keepdims (bool): Keepdims used at sum function's forward.

    Returns:
        Variable: Gradient with singleton dimensions restored for broadcasting.
    """
    ndim = len(x_shape)
    tupled_axis = axis
    if axis is None:
        tupled_axis = None
    elif not isinstance(axis, tuple):
        tupled_axis = (axis,)

    if not (ndim == 0 or tupled_axis is None or keepdims):
        actual_axis = [a if a >= 0 else a + ndim for a in tupled_axis]
        shape = list(gy.shape)
        for a in sorted(actual_axis):
            shape.insert(a, 1)
    else:
        shape = gy.shape

    gy = gy.reshape(shape)  # reshape
    return gy
