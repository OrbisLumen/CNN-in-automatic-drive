# DonkeyCar 环境说明

MyDeZero 和小车控制共用根目录 `.venv`；每人本机创建环境，不复制虚拟环境目录。
Windows/macOS 使用相同 Python 代码及依赖版本，各自下载对应系统的模拟器。

## 文件与版本

| 内容 | 位置 |
| --- | --- |
| 第三方接口源码 | `tools/gym-donkeycar/` |
| 模拟器 | `tools/donkey-sim/` |
| 自写代码 | 对应学期 `proj/` |
| 要求与模板 | 对应学期 `request/` |
| 软件工程报告与图表 | 对应学期 `work/software-engineering/`（计划位置） |

`tools/`、`.venv` 不提交 Git。数据、权重和训练输出放在已忽略的 `data/` 或共享存储中，
仓库记录下载地址、校验值和对应实验。模拟器路径不要写死为个人电脑的绝对路径。

统一下载基线：

- [接口源码](https://github.com/tawnkramer/gym-donkeycar)：`a1f4ca6961e17a6929c1f2e883358a1edafb479d`
- 模拟器：`v25.10.06`，下载 [macOS 包](https://github.com/tawnkramer/gym-donkeycar/releases/download/v25.10.06/DonkeySimMac.zip) 或 [Windows 包](https://github.com/tawnkramer/gym-donkeycar/releases/download/v25.10.06/DonkeySimWin.zip)

版本更新通过 PR 同步记录，成员不要各自跟随最新 master。

## 下载与安装

以下命令在课程仓库根目录执行。已有目录或源码时，跳过对应创建、克隆步骤：

```bash
mkdir tools
git clone https://github.com/tawnkramer/gym-donkeycar.git tools/gym-donkeycar
git -C tools/gym-donkeycar checkout a1f4ca6961e17a6929c1f2e883358a1edafb479d
```

模拟器解压到 `tools/donkey-sim/`；Windows 保留 exe、数据目录及 DLL 的相对位置。
按 [根 README](../README.md) 创建 `.venv` 后，使用该环境安装两个项目：

```bash
# macOS
.venv/bin/python -m pip install -e ./projl/proj -e ./tools/gym-donkeycar
.venv/bin/python -c "import mydezero, gymnasium, gym_donkeycar; print('import OK')"
```

```powershell
# Windows PowerShell，无需激活环境
.\.venv\Scripts\python.exe -m pip install -e ./projl/proj -e ./tools/gym-donkeycar
.\.venv\Scripts\python.exe -c "import mydezero, gymnasium, gym_donkeycar; print('import OK')"
```

当前本机 `.venv` 为 Python 3.14，尚未完成上述安装与兼容性验证；不预先要求重建环境。
联调通过后，将统一的 Python 和依赖版本记录在项目中。

macOS 若因主程序缺少执行权限无法启动：

```bash
chmod u+x tools/donkey-sim/DonkeySimMac/donkey_sim.app/Contents/MacOS/donkey_sim
open tools/donkey-sim/DonkeySimMac/donkey_sim.app
```

## 验收与实验记录

当前 Mac 包已下载、校验、解压，补执行权限后已启动；支持 Intel 和 Apple Silicon。
Python 连接、控制及 Windows 运行仍待验证。Mac 压缩包 SHA-256：
`edc36077126257320d433a7d07266c1feae289245687c4b9139bac353e31dce0`。

至少各有一台 Mac 和 Windows 完成：启动模拟器 → Python 连接 → 收到图像 → 控制转向和油门 → 正常关闭。
当前接口为 Gymnasium：`reset()` 返回 `(obs, info)`，
`step()` 返回 `(obs, reward, terminated, truncated, info)`。

实验记录代码提交号、依赖版本、模拟器版本、赛道、随机种子、评价指标与推理耗时。
