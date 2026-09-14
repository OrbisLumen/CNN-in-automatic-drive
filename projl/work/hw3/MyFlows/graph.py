def collect(output):
    """迭代收集当前输出的祖先图；对象去重，输入使用次数不去重。"""
    variables, functions = [], []
    seen_v, seen_f = set(), set()
    pending = [output]
    while pending:
        v = pending.pop()
        if v in seen_v:
            continue
        seen_v.add(v)
        variables.append(v)
        f = v.creator
        if f is not None and f not in seen_f:
            seen_f.add(f)
            functions.append(f)
            pending.extend(reversed(f.inputs))
    return variables, functions


def trace(output):
    """返回 (nodes, adjacency)；本阶段 nodes 新增 object 字段。"""
    variables, functions = collect(output)
    vids = {v: "v" + str(i) for i, v in enumerate(variables)}
    fids = {f: "f" + str(i) for i, f in enumerate(functions)}
    nodes = {vids[v]: {"kind": "variable", "creator": fids.get(v.creator),
                      "object": v, "name": v.name} for v in variables}
    nodes.update({fids[f]: {"kind": "function", "output": vids[f.output],
                           "object": f} for f in functions})
    adjacency = {fids[f]: [vids[v] for v in f.inputs] for f in functions}
    return nodes, adjacency


def clear_graph_grads(output):
    """独立重算同一张图时，清除所有祖先变量及输出的梯度。"""
    variables, _ = collect(output)
    for v in variables:
        v.cleargrad()
