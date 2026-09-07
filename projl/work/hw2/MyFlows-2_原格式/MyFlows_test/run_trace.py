import numpy as np
from MyFlows import Variable, Square, Exp, trace, format_trace


def build(name, value):
    """建一条三连的单输入链，中间变量起名是为了打印时看得清。"""
    x = Variable(np.array(value), name=name)
    a = Square()(x)
    a.name = "a"
    b = Exp()(a)
    b.name = "b"
    y = Square()(b)
    y.name = "y"
    return x, y


print("【第一段】一条链 x → Square → a → Exp → b → Square → y")
print("-" * 46)
_, y = build("x", 0.5)
nodes, adjacency = trace(y)
print(format_trace(nodes, adjacency))
print()
print("adjacency 它自己是这样：", adjacency)


print()
print("【第二段】换一个输入数值，再跑一次")
print("-" * 46)
_, y2 = build("x", 2.0)
nodes2, adjacency2 = trace(y2)
print(format_trace(nodes2, adjacency2))
print()
print("两次的 adjacency 一样吗：", adjacency == adjacency2)


print()
print("【第三段】旁边有别的计算时，会不会被捞进来")
print("-" * 46)
_, y3 = build("x", 0.5)
other = Exp()(Variable(np.array(3.0), name="other"))
nodes3, adjacency3 = trace(y3)
print("这次登记了几个节点：", len(nodes3))
print("名字里搜得到other吗：",
      "other" in [i.get("name") for i in nodes3.values()])
print("other 这个变量确实存在：", other.data)
