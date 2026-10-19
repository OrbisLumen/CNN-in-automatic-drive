# Git 与 GitHub 协作规范

五人小组统一采用：**个人分支 → PR → 负责人审查 → 合并至 master**。
本文约定团队流程；远程权限是否生效，以 GitHub 实际配置为准。

## 分工与分支

| 成员 | 个人分支 |
| --- | --- |
| OrbisLumen | `dev/orbislumen` |
| Booyean | `dev/booyean` |
| yangchunyu-1 | `dev/yangchunyu-1` |
| Qiulin087 | `dev/qiulin087` |
| lucky-money-account | `dev/lucky-money-account` |

- 只向自己的个人分支推送，代码与文档均通过 PR 进入 `master`。
- OrbisLumen、Booyean 负责审查与合并，两人自己的 PR 由另一人审查；作者不能自批。
- 个人分支长期使用，PR 采用 **Create a merge commit**，合并后同步主分支。
- 分支名统一小写，不创建单独的 `dev` 分支；分支命名本身不限制权限。
- 每项任务明确负责人、交付内容和验收方法；Word 等二进制文档同一时间由一人编辑。

## 日常操作

以下以 OrbisLumen 为例，在仓库根目录执行，其他成员替换分支名。

首次加入并创建个人分支：

```bash
git clone https://github.com/OrbisLumen/CNN-in-automatic-drive.git
cd CNN-in-automatic-drive
git config user.name "你的名字"
git config user.email "你的邮箱"
git fetch origin
git switch --no-track -c dev/orbislumen origin/master
git push -u origin dev/orbislumen
```

本地已有分支时使用 `git switch dev/orbislumen`；仅远程已有时使用
`git switch --track origin/dev/orbislumen`。若要保留当前提交作为起点，改用
`git switch -c dev/orbislumen`，而非从 `origin/master` 创建。

开始任务及 PR 合并后同步主分支。切换、合并前先保存工作，确认工作区干净：

```bash
git status
git switch dev/orbislumen
git fetch origin
git merge origin/master
git push
```

修改前阅读对应 `request/` 和 `AGENTS.md`，完成后运行相关验证，再提交指定文件：

```bash
python -m pytest projl/proj/tests
git diff
git add README.md
git diff --cached
git diff --cached --check
git commit -m "docs(readme): 精简项目说明"
git push
```

示例中的测试和文件应按任务替换。暂存后再次修改需重新 `git add`；未提交内容不会随推送上传。
每次提交围绕一项完整改动，标题使用 `类型(范围): 具体内容`，如 `feat`、`fix`、`test`、
`docs`、`refactor`、`chore`，避免“update”“最终版”。不提交个人环境、数据、权重或无关改动。

## PR 与合并

创建前检查个人分支的全部差异，避免带入其他任务：

```bash
git fetch origin
git log --oneline origin/master..HEAD
git diff --stat origin/master...HEAD
```

在 GitHub 创建 PR：`base: master`，`compare: 个人分支`。
一个 PR 处理一项任务，说明目的、主要改动、验证命令与结果；未测试应说明原因，
跨平台改动注明已验证的系统与 Python 版本。

负责人检查实现、测试、范围和可维护性；`Comment` 不等于批准，`Request changes` 需处理，
`Approve` 才表示通过。使用 `Start a review` 后还需 `Submit review` 发布。
作者在原分支修改并推送即可更新 PR；新增改动后需重新批准。

负责人批准、讨论解决、相关验证及已配置的自动检查通过后，由 OrbisLumen 或 Booyean 合并。
前一任务未合并时，不往同一 PR 加入无关任务。

## 主分支权限：目标与历史记录

目标：全员可推个人分支，仅两位负责人能通过已批准的 PR 更新 `master`，负责人也不能直接推送。
批准人和合并人的限制需分别配置，CODEOWNERS 本身不能限制谁合并。

默认分支的 `.github/CODEOWNERS` 应包含：

```text
* @OrbisLumen @Booyean
```

两位负责人需有写权限。目标为 `master` 的规则拆成两条，避免维护者同时绕过审查：

| 规则 | 配置 | 例外 |
| --- | --- | --- |
| `Master review` | 必须 PR、1 次 Code Owner 批准、新提交取消旧批准、解决讨论、禁止强推和删除 | 无 |
| `Master maintainers` | 仅 `Restrict updates` | 两位具体负责人，`For pull requests only` |

不要把整个 Write 角色加入例外；审查规则中的管理员 `Always allow` 应移除。
若界面无法选择具体用户，需核对官方 API 支持情况，或在组织仓库使用负责人团队。
仓库所有者仍可修改规则。

**2026-10-06 历史核查，非实时状态：** PR、1 次批准、Code Owner 批准、解决讨论、
新提交取消旧批准、禁止强推及删除已启用；CODEOWNERS 经
[PR #3](https://github.com/OrbisLumen/CNN-in-automatic-drive/pull/3) 提交，当时待合并。
CODEOWNERS 须进入目标分支才生效。管理员 `Always allow` 仍存在，
仅负责人更新主分支的限制尚未配置和验证。修改本文不会改变远程规则。

配置后验证：普通成员可推个人分支但不能合并；负责人可合并已批准 PR，
不能合并未批准 PR，也不能直接推主分支。

## 常见问题

- 推送被主分支规则拒绝：改向个人分支推送并提 PR，不强推绕过。
- `non-fast-forward`：获取并检查远程同名分支，合并、验证后再推送。
- 合并冲突：用 `git status` 定位，理解双方改动、移除冲突标记，验证后提交。
- 个人分支推送后主分支没变化：需等 PR 合并。

参考：[PR 审查](https://docs.github.com/en/pull-requests/get-started/reviewing-pull-requests-quickstart)、
[CODEOWNERS](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners)、
[Rulesets](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/creating-rulesets-for-a-repository)、
[Rulesets API](https://docs.github.com/en/rest/repos/rules#create-a-repository-ruleset)。
