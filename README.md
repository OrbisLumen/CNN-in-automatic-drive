# CNN-in-automatic-drive

CNN 自动驾驶课程仓库：实现 MyDeZero，并用于 DonkeyCar 仿真小车的训练与控制。

## 目录

```text
├── projl/       # 第一学期
├── projll/      # 第二学期
├── projlll/     # 第三学期
├── doc/         # 环境与协作说明
└── tools/       # 本机第三方源码、模拟器，不提交 Git
```

各学期按 `request/`（要求）、`reference/`（参考）、`proj/`（项目）、
`work/`（作业）整理。开始任务前阅读对应要求及 [AGENTS.md](AGENTS.md)，改动限于相关学期。

## 安装与测试

每台电脑使用一个根目录 `.venv`，供 MyDeZero 和小车控制共用。首次创建：

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

先按[环境说明](doc/donkeycar-setup.md)下载 tools 中的固定版本源码与模拟器，再使用环境中的 Python 安装依赖和两个本地项目：

```bash
python -m pip install -r requirements.txt
python -m pytest projl/proj/tests
```

环境基线为 Python 3.12，直接依赖版本见 `requirements.txt`；macOS 兼容性需单独验证。
DonkeyCar 下载、安装及 Windows 无需激活环境的用法见 [环境说明](doc/donkeycar-setup.md)。
MyDeZero 详情见 [项目 README](projl/proj/README.md)。

## 小组协作

个人分支 `dev/<github用户名>` → PR → 负责人审查 → 合并至 `master`。
OrbisLumen、Booyean 负责审查与合并；具体操作及权限配置状态见 [协作规范](doc/collaboration.md)。
个人环境、模拟器、数据和模型权重不提交 Git。
