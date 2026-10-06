# Git 与 GitHub 协作规范

本规范适用于 CNN-in-automatic-drive 项目的五人小组，规定个人开发分支、提交记录、Pull Request（PR）审查及主分支更新流程。

团队采用以下流程：

```text
个人开发分支 → 提交 PR → 负责人审查 → 负责人合并 → master
```

本文中的权限配置为目标方案；文档修改不会自动创建分支、添加 CODEOWNERS 或更新 GitHub Rules。此前核查结果及尚需配置的项目见第 7 节。

## 1. 角色与职责

| 角色 | 成员 | 职责 |
| --- | --- | --- |
| 开发成员 | 全体五位成员 | 完成任务、验证改动、提交 PR，并处理审查意见 |
| PR 审查负责人 | OrbisLumen、Booyean | 检查实现、任务范围及验证结果，批准或要求修改 |
| 主分支维护负责人 | OrbisLumen、Booyean | 在审查要求满足后，将 PR 合并至 `master` |

OrbisLumen 提交的 PR 由 Booyean 审查；Booyean 提交的 PR 由 OrbisLumen 审查。其他成员的 PR 至少获得上述两位负责人之一的批准。作者不得批准自己的 PR。

两位负责人均参与开发，不因维护职责而免除 PR 审查要求。仓库所有者保留仓库及规则的管理权限。

## 2. 分支管理

| 分支 | 用途 |
| --- | --- |
| `master` | 保存经审查和验证的小组共同版本 |
| `dev/orbislumen` | OrbisLumen 的个人开发分支 |
| `dev/booyean` | Booyean 的个人开发分支 |
| `dev/yangchunyu-1` | yangchunyu-1 的个人开发分支 |
| `dev/qiulin087` | Qiulin087 的个人开发分支 |
| `dev/lucky-money-account` | lucky-money-account 的个人开发分支 |

分支名统一使用小写。`dev/` 为命名前缀，各分支相互独立，不存在父子继承或自动同步关系。不得另外创建名为 `dev` 的分支，以免与 `dev/用户名` 的引用路径冲突。

成员应仅向自己的个人分支推送。所有进入 `master` 的改动均应通过 PR，包括 MyDeZero、其他学期项目及共享文档。分支名称不构成访问权限限制；个人分支的使用范围由团队约定管理。

长期使用个人分支时，PR 合并方式采用 **Create a merge commit**。

## 3. 开发操作

以下命令以 OrbisLumen 为例，其他成员应替换为自己的分支名称。除测试命令明确切换目录外，所有命令均在仓库根目录执行。

### 3.1 首次加入

接受仓库协作者邀请后，下载仓库并配置个人提交身份：

```bash
git clone https://github.com/OrbisLumen/CNN-in-automatic-drive.git
cd CNN-in-automatic-drive
git config user.name "你的名字"
git config user.email "你的邮箱"
```

### 3.2 创建或切换个人分支

首次从最新远程主分支创建个人分支：

```bash
git status
git fetch origin
git switch --no-track -c dev/orbislumen origin/master
git push -u origin dev/orbislumen
```

`--no-track` 避免将新个人分支跟踪到远程主分支；首次推送的 `-u` 会建立与远程同名个人分支的跟踪关系，之后可直接使用 `git push`。

上述起点不包含仅存在于本地其他分支的提交。需要保留当前已提交工作时，应从当前 commit 创建个人分支，即改用 `git switch -c dev/orbislumen`，再执行首次推送。

本地个人分支已经存在时：

```bash
git switch dev/orbislumen
```

远程已有个人分支、本地尚未创建时：

```bash
git fetch origin
git switch --track origin/dev/orbislumen
```

### 3.3 同步主分支

开始新任务及 PR 合并后，应将最新共同版本整合到个人分支。执行前应妥善保存已有工作，并确认工作区无未提交改动：

```bash
git status
git switch dev/orbislumen
git fetch origin
git merge origin/master
```

`fetch` 更新远程信息，不修改当前分支的文件；`merge` 将远程主分支的提交整合进当前个人分支。处理可能出现的冲突并验证结果后，使用 `git push` 上传同步后的历史。

### 3.4 验证、提交与上传

修改前应阅读对应学期的 `request/`、适用的 `AGENTS.md` 及参考材料。修改完成后，执行与改动相关的验证。第一学期 MyDeZero 测试命令为：

```bash
cd projl/proj
python3 -m pytest tests
cd ../..
```

以下以修改根目录 `README.md` 为例，提交时应指定本次任务的实际文件：

```bash
git diff
git add README.md
git diff --cached
git diff --cached --check
git commit -m "docs(readme): 补充项目运行说明"
git push
```

| 命令 | 作用 |
| --- | --- |
| `git status` | 查看当前分支、文件状态及本地与远程的关系 |
| `git diff` | 查看已跟踪文件中尚未暂存的改动 |
| `git add 文件名` | 将指定文件的当前改动加入暂存区 |
| `git diff --cached` | 检查下一次提交将包含的内容 |
| `git diff --cached --check` | 检查差异中的空白错误等问题，不能替代功能测试 |
| `git commit` | 将暂存区保存为一次本地提交 |
| `git push` | 将已提交历史上传至远程分支 |

暂存后再次编辑的内容需要重新 `add`。已提交内容无需重复提交即可上传；未提交的编辑内容不会随 `push` 上传。

## 4. 提交记录规范

每条 commit 应对应一个完整、可解释的改动。功能及其验证测试可以共同提交，不应按文件机械拆分，也不应混入无关任务。

提交标题采用 `类型(范围): 具体改动`，范围可省略；标题可使用中文或英文，表述应准确、简洁。

| 类型 | 含义 | 示例 |
| --- | --- | --- |
| `feat` | 新增功能 | `feat(mydezero): 添加 ReLU 激活函数` |
| `fix` | 修复错误 | `fix(autograd): 修复广播输入的梯度形状` |
| `test` | 添加或调整测试 | `test(matmul): 补充矩阵乘法梯度测试` |
| `docs` | 文档更新 | `docs(readme): 补充项目运行说明` |
| `refactor` | 保持功能行为的重构 | `refactor(core): 提取重复的变量转换逻辑` |
| `chore` | 配置及日常维护 | `chore(gitignore): 忽略本地训练输出` |

必要时在标题后的空行添加正文，说明改动原因和行为变化。避免使用 `update`、`最终版` 等无法说明改动内容的标题。

提交前应确认变更范围，不包含无关学期文件、个人环境、数据集、模型权重或训练输出。

## 5. PR 审查与合并

### 5.1 提交要求

一个 PR 应聚焦一项任务。创建前检查个人分支相对主分支的整体差异：

```bash
git fetch origin
git log --oneline origin/master..HEAD
git diff --stat origin/master...HEAD
```

在 GitHub 的 `Pull requests → New pull request` 中选择 `base: master`，并将 `compare` 设置为个人分支。也可使用 GitHub CLI：

```bash
gh pr create --base master --head dev/orbislumen --title "docs(readme): 补充项目运行说明"
```

PR 说明应包含改动目的、主要内容、实际验证命令与结果，以及需重点审查的事项。未运行测试时应说明原因。前一项任务未合并时，不应在同一 PR 中继续加入不相关任务。

### 5.2 审查要求

负责人应检查任务完成情况、实现正确性、必要的边界情况、验证结果、变更范围及可维护性。审查意见应指出具体问题及其影响，并在必要时提出修改建议。

- `Comment`：一般反馈或讨论，不表示批准。
- `Request changes`：存在必须处理的问题。
- `Approve`：改动满足合并要求。

使用 `Start a review` 保存的意见需通过最终 `Submit review` 发布。作者应在原分支修改、提交并推送，原 PR 将自动更新。新增可审查改动后，应重新获得批准。

### 5.3 合并要求

最终合并由 OrbisLumen 或 Booyean 执行。合并前应满足负责人批准要求、解决审查讨论，并确认验证结果。配置自动检查后，还应通过指定检查。

对于长期个人分支，使用 `Create a merge commit`，合并后按第 3.3 节同步个人分支。

## 6. 主分支权限配置

团队目标是：全体成员可上传个人分支；仅 OrbisLumen 和 Booyean 可合并 PR、更新 `master`；两位负责人同样必须经过审查，不能直接推送至主分支。

“批准人”与“合并人”应分别控制。仅要求一次批准不能指定批准人；仅添加 CODEOWNERS 不能限制合并操作人。

### 6.1 指定审查负责人

应在默认分支配置 `.github/CODEOWNERS`：

```text
* @OrbisLumen @Booyean
```

两位负责人必须具备仓库写权限。两人位于同一行，任意一位批准即可满足 Code Owner 要求。应启用 `Require review from Code Owners`，并将批准数设置为 **1**。

该审查要求已启用，配置文件已在本地创建并通过 [PR #3](https://github.com/OrbisLumen/CNN-in-automatic-drive/pull/3) 提交。GitHub 使用 PR 目标分支中的 CODEOWNERS；该 PR 合并至 `master` 后，指定负责人的配置才会对主分支生效。

### 6.2 分离审查规则与更新权限规则

应同时启用两条目标为 `master` 的 Branch Ruleset：

| Ruleset | 规则 | Bypass list |
| --- | --- | --- |
| `Master review` | 必须 PR、1 次批准、Code Owner 批准、新提交取消旧批准、解决审查讨论、禁止强推及删除 | 空 |
| `Master maintainers` | 仅启用 `Restrict updates` | 仅 OrbisLumen 和 Booyean，模式为 `For pull requests only` |

第二条规则应使用具体用户身份，不应添加整个 `Write` 角色。例外只作用于更新权限规则；两位负责人在审查规则中没有例外，因此仍需经过批准。

GitHub 官方 Ruleset REST API 支持 `User` 类型的 bypass actor。如页面无法选择具体用户，应使用受支持的接口配置并验证，或在组织仓库使用仅包含两位负责人的团队。

不得将全部审查要求与更新权限放入同一条规则，再允许负责人绕过整条规则，否则审查要求也会被绕过。现有审查规则中的管理员 `Always allow` 例外应在实施时移除。

### 6.3 配置验证

管理员应分别验证以下行为：

| 身份与操作 | 预期结果 |
| --- | --- |
| 普通成员推送个人分支 | 允许 |
| 普通成员将已获批准的 PR 合并至 `master` | 拒绝 |
| 任一负责人合并未获有效批准的 PR | 拒绝 |
| 任一负责人合并满足全部要求的 PR | 允许 |
| 任一负责人直接推送至 `master` | 拒绝 |

权限限制适用于规则保持启用时的日常更新操作。仓库所有者仍可通过管理权限修改规则。

## 7. 配置状态记录

以下是 **2026-10-06 实施第 6.1 节后的核查记录**，不代表实时状态；规则变更或配置 PR 合并后应重新核查并更新本节。

| 项目 | 核查结果 |
| --- | --- |
| 仓库类型与默认分支 | OrbisLumen 的个人账号仓库；默认分支 `master` |
| 已启用要求 | PR、1 次批准、解决讨论、新提交取消旧批准、禁止强推和删除 |
| Code Owner 批准要求 | 已启用；批准数为 1 |
| CODEOWNERS 文件 | 本地已创建，远程 PR #3 待审查和合并；GitHub 校验无错误 |
| 管理员例外 | 存在 `Always allow` |
| 仅两位负责人更新主分支 | 尚未完成第 6.2 节的配置与第 6.3 节的验证 |

本次实施新增了 `.github/CODEOWNERS`，并启用了远程 Code Owner 审查要求。配置文件尚未进入默认分支。管理员例外和主分支更新者限制未在此次修改；第 6.2 节仍为待实施方案。

## 8. 异常处理

- **主分支规则拒绝推送**：向个人分支上传并创建 PR，不通过强推绕过审查。
- **个人分支推送提示 non-fast-forward**：获取远程更新，检查并合并远程同名分支，验证后重新推送。
- **合并冲突**：用 `git status` 定位冲突文件，理解双方改动，编辑为最终内容并删除冲突标记，然后暂存、提交、验证及推送。
- **个人分支上传后主分支未变化**：PR 合并前，主分支保持原版本。

## 9. 参考资料

- [Git push](https://git-scm.com/docs/git-push)
- [GitHub PR 审查](https://docs.github.com/en/pull-requests/get-started/reviewing-pull-requests-quickstart)
- [GitHub Code Owners](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners)
- [Ruleset 支持的规则](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/available-rules-for-rulesets)
- [Ruleset 创建与绕过权限](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/creating-rulesets-for-a-repository)
- [Ruleset REST API](https://docs.github.com/en/rest/repos/rules#create-a-repository-ruleset)
