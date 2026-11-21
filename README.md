# CNN-in-automatic-drive

本仓库对应进阶式挑战性综合项目 I。每学期单独建仓。第一学期的目标是实现深度学习框架 MyDeZero，并将它用于 DonkeyCar 仿真训练与控制。当前已有框架、测试和 Notebook 示例，小车接口与训练流程是下一步工作。

## 目录

| 路径 | 用途 |
| --- | --- |
| `src/mydezero/` | 自动微分、网络层、模型、优化器与数据加载 |
| `tests/` | 单元测试与集成测试 |
| `examples/` | 自动微分、回归与神经网络 Notebook |
| `request/` | 本学期任务书、中期与期末材料模板 |
| `doc/` | [MyDeZero 文档](doc/mydezero/README.md)、[UML 设计稿](doc/uml/README.md)、环境与协作说明 |
| `reference/`、`tools/` | 本地参考材料、第三方接口源码和模拟器，Git 忽略 |

后续小车接口和可复用的训练代码放在 `src/` 下，与 `mydezero/` 并列。实验数据按原始和处理后分别放在 `data/raw/`、`data/processed/`，权重放在 `models/`，每次训练的日志和指标放在 `runs/`。这些本地产物目录按需创建。

## 安装与测试

从仓库根目录使用 Python 3.12 创建虚拟环境。

```bash
# macOS / Linux
python3.12 -m venv .venv
source .venv/bin/activate
```

```powershell
# Windows PowerShell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

按 [DonkeyCar 环境说明](doc/donkeycar-setup.md) 将固定版本的接口源码放到 `tools/gym-donkeycar/`，然后安装课程依赖并运行测试。

```bash
python -m pip install -r requirements.txt
python -m pytest
```

只使用框架时，可以直接执行 `python -m pip install -e .`。Notebook 和测试使用上面的课程环境。模拟器联通检查见 [DonkeyCar 环境说明](doc/donkeycar-setup.md)，Windows NVIDIA 环境见 [CuPy 安装说明](doc/cupy-setup.md)。

## 小组协作

从最新的 `origin/master` 建立任务分支，每项任务对应一个 PR。OrbisLumen、Booyean 负责审查与合并。详见 [协作规范](doc/collaboration.md) 和 [AGENTS.md](AGENTS.md)。
