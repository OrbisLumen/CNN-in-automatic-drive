# 图检查返回格式

`trace(output)` 返回 `(nodes, adjacency)`

---

## 一、提交格式

### 例子：单输入链

代码在 `examples/chain.py`，链是 `x → Square → a → Exp → b → Square → y`。调用 `trace(y)` 要得到这两份东西。

```python
nodes = {
    "v0": {"kind": "variable", "name": "x", "shape": (), "creator": None},
    "f0": {"kind": "function", "type": "Square", "output": "v1"},
    "v1": {"kind": "variable", "name": "a", "shape": (), "creator": "f0"},
    "f1": {"kind": "function", "type": "Exp", "output": "v2"},
    "v2": {"kind": "variable", "name": "b", "shape": (), "creator": "f1"},
    "f2": {"kind": "function", "type": "Square", "output": "v3"},
    "v3": {"kind": "variable", "name": "y", "shape": (), "creator": "f2"},
}

adjacency = {
    "f0": ["v0"],
    "f1": ["v1"],
    "f2": ["v2"],
}
```

字典里键值对的顺序随便，验收只比内容。上面这个顺序是边走边登记的自然结果。

`graph_format.py` 里的 `format_trace` 会把这两份东西打成下面的样子，打印不用你自己写。

```
变量节点
编号  名称  shape  creator
v0  x  ()  None
v1  a  ()  f0
v2  b  ()  f1
v3  y  ()  f2

运算及有序输入
f0 Square(v0) = v1
f1 Exp(v1) = v2
f2 Square(v2) = v3
```

### 例子：只有一个变量

```python
x = Variable(1.0, name="leaf")
trace(x)
# nodes     = {"v0": {"kind": "variable", "name": "leaf", "shape": (), "creator": None}}
# adjacency = {}
```

变量没有创造者时 `adjacency` 是空字典，不要在表里给它留一个 `{"v0": []}` 的空行。

### 例子：一个运算接受两个输入

```python
adjacency = {"f0": ["v3", "v0"]}
```

自己接受自己也照样记两次。

```python
Add()(x, x)
adjacency = {"f0": ["v1", "v1"]}
```

---

## 二、规则

### 编号

变量编 `v0`、`v1`，运算编 `f0`、`f1`。

编号不能用对象的内存地址，也就是不能用 `id()`。原因很直接，地址每次运行都变，而验收里有一项专门比两次调用的结果是否相等，用 `id()` 必挂。

相同结构、相同输入顺序的图要产生相同编号。编号怎么排由你定，只要稳定，不要求是拓扑顺序。

同名或同值的不同对象仍然是不同节点。`Variable(1.0)` 建两次就是两个节点，各占一个编号。

### 字段

`nodes` 把变量和运算混在一张表里，靠 `kind` 区分。两类行的字段不通用。

变量行有四个字段。

| 字段 | 填什么 | 从哪来 |
|---|---|---|
| `kind` | 固定字符串 `"variable"` | 直接写 |
| `name` | 变量名字，没起名字就是 `None` | `v.name` |
| `shape` | 数据形状，是 tuple | `v.data.shape` |
| `creator` | 生出我的那个运算的编号，起点填 `None` | 遍历时记下来 |

运算行有三个字段。

| 字段 | 填什么 | 从哪来 |
|---|---|---|
| `kind` | 固定字符串 `"function"` | 直接写 |
| `type` | 运算的类名，是字符串 | `type(f).__name__` |
| `output` | 我生出的那个变量的编号 | 遍历时记下来 |

标量的 shape 是空 tuple，写成 `()`。`Variable([1.0, 2.0])` 的 shape 是 `(2,)`。

本讲每次运算只有一个输出，所以 `output` 是一个编号字符串，不是列表。

### 邻接表只记入边

`adjacency` 的键只包含运算节点，变量节点不出现在键里。值是输入变量的编号列表，按调用时写参数的顺序排，不能排序也不能去重。

同一变量被用两次就记两次，不转集合。对象本身只登记一次，但使用次数要留在表里。

运算到输出的那条边不在这张表里，它在 `nodes` 的 `output` 字段里。这张表是入边表，出边要去 `nodes` 查。要数一张图有几条边，入边看 `adjacency`，出边看 `nodes` 里的 `output`，两边加起来才对。本讲验收不考边数统计，后面讲反向传播时要用。

### 只走上游，而且只读

只追踪给定输出的上游依赖，不扫全局变量。从预测开始往回走，不会碰到旁边的损失分支。独立变量也是合法输入，见上面第二个例子。

`trace` 不能改任何东西。data、grad、creator、input、inputs、output 都不能写，也不能重新调用 `forward`。

### 这套格式不打算做什么

编号加运算名加连接关系，还不足以把任意模型原样跑起来。带阈值的运算还需要配置和输入值才行。本讲验收的是算式结构，不是模型序列化。

---

## 三、验收对应关系

跑 `python -m unittest discover -s exercises -p 'check_*.py' -v`，六项分别卡下面这些点。

| 测试 | 卡什么 | 对应哪一节规则 |
|---|---|---|
| `test_leaf` | 孤立变量进来，`nodes` 只有一项，`adjacency` 是空字典 | 字段、邻接表只记入边 |
| `test_chain` | 7 个节点、3 个运算、3 条入边，连接是 x 到 a、a 到 b、b 到 y | 编号、字段 |
| `test_unrelated_computation_is_excluded` | 旁边的计算不算进来 | 只走上游 |
| `test_trace_does_not_change_state` | 调完 trace 后 data、grad、creator 原封不动 | 只读 |
| `test_deterministic_labels` | 同一个 y 调两次完全相等，两条同构的链也相等 | 编号 |
| `test_shape_is_recorded` | `Variable([1.0, 2.0])` 的 shape 记成 `(2,)` | 字段 |

`tests/` 里另外 8 项是基础功能测试，跟你写的 `trace` 无关，拿到包就该全过。
