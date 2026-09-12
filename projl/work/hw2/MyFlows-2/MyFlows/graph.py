"""读图：把内存里互相引用的对象，变成两张只看得见编号的表。
对外两个函数：
    trace(output)          顺着 creator 往回走，返回 (nodes, adjacency)
    format_trace(n, adj)   把那两张表做成格式化输出
返回的格式参考 GRAPH_CONTRACT.md 
"""


def trace(output):
    """读取 output 的上游，返回 (nodes, adjacency)。

    nodes      用 v0 v1 记变量、f0 f1 记运算，值是各自的字段字典
    adjacency  {"f0": ["v0"]}，键只登记运算，值是它吃的那些变量
    本函数只读，不改动任何一个对象。
    """
    # 第一步：从 output 出发顺着 creator 往回走，把碰到的变量收进来
    visited = []
    seen = set()
    stack = [output]
    while stack:
        v = stack.pop()
        if id(v) in seen:
            continue
        seen.add(id(v))
        visited.append(v)
        f = v.creator
        if f is not None and id(f.input) not in seen:
            stack.append(f.input)

    # 第二步：倒过来排，就回到了前向计算的自然顺序
    visited.reverse()
    labels = {}
    for i, v in enumerate(visited):
        labels[id(v)] = "v%d" % i

    # 第三步：边走边给运算编号，顺手把两张表填好
    nodes = {}
    adjacency = {}
    fseq = 0
    for v in visited:
        key = labels[id(v)]
        f = v.creator
        if f is None:
            nodes[key] = {"kind": "variable", "name": v.name,
                          "shape": tuple(v.data.shape), "creator": None}
            continue
        fid = "f%d" % fseq
        fseq += 1
        nodes[fid] = {"kind": "function", "type": type(f).__name__,
                      "output": key}
        nodes[key] = {"kind": "variable", "name": v.name,
                      "shape": tuple(v.data.shape), "creator": fid}
        adjacency[fid] = [labels[id(f.input)]]
    return nodes, adjacency


def format_trace(nodes, adjacency):
    """把 trace 的返回结果排成两张小表格，方便肉眼看清楚结构。"""
    lines = ["变量节点", "编号  名称  shape  creator"]
    for key, item in nodes.items():
        if item["kind"] != "variable":
            continue
        name = item["name"] if item["name"] is not None else "-"
        lines.append("%s  %s  %s  %s" % (key, name, item["shape"],
                                        item["creator"]))
    lines.extend(["", "运算及有序输入"])
    for key, inputs in adjacency.items():
        item = nodes[key]
        lines.append("%s %s(%s) = %s" % (key, item["type"],
                                         ", ".join(inputs), item["output"]))
    return "\n".join(lines)
