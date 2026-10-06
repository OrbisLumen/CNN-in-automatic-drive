# 五人小组 Git 与 GitHub 协作说明

每个人在自己的 `dev/用户名` 分支开发并上传；完成一项任务后，通过 Pull Request（PR）将改动合并进 `master`。PR 审查负责人是 **OrbisLumen** 和 **Booyean**。

本文说明协作约定和建议的权限配置。文档上传本身不会创建五条个人分支，也不会自动修改 GitHub Rules。

## 1. 分支与分工

| 分支 | 用途 |
| --- | --- |
| `master` | 小组认可、验证过的共同版本 |
| `dev/orbislumen` | OrbisLumen 的个人开发分支 |
| `dev/booyean` | Booyean 的个人开发分支 |
| `dev/yangchunyu-1` | yangchunyu-1 的个人开发分支 |
| `dev/qiulin087` | Qiulin087 的个人开发分支 |
| `dev/第五位成员的用户名` | 第五位成员加入后，替换为实际用户名 |

`dev/用户名` 是分支命名方式，不是真正的父子关系。各分支不会自动同步，也不会因名称而成为私有分支。不要另外创建名为 `dev` 的分支，它会与 `dev/用户名` 的引用路径冲突。

协作路径：`个人分支 → PR → 负责人审查 → 负责人合并 → master`。

- 每个人只向自己的个人分支 push；不直接更新 `master`。
- OrbisLumen 的 PR 由 Booyean 审查；Booyean 的 PR 由 OrbisLumen 审查。
- 其他成员的 PR，由这两位负责人中的一位审查即可，不要求两人同时批准。
- 小组约定由这两位负责人执行最终合并。若需 GitHub 强制限制合并人，配置第 5 节的规则。
- 个人分支中的代码尚未进入共同版本。上传分支与合并 PR 是两个动作。

## 2. 首次加入与创建个人分支

先接受仓库协作者邀请。第一次下载仓库时执行：

```bash
git clone https://github.com/OrbisLumen/CNN-in-automatic-drive.git
cd CNN-in-automatic-drive
git config user.name "你的名字"
git config user.email "你的邮箱"
```

以下以 OrbisLumen 为例，其他成员替换分支名。后续命令默认在仓库根目录执行。

```bash
git status
git fetch origin
git switch --no-track -c dev/orbislumen origin/master
git push -u origin dev/orbislumen
```

- `status`：检查当前分支和未提交的文件改动。切换前先妥善提交已有工作。
- `fetch origin`：获取远程最新信息，不自动合并当前分支。
- `switch -c`：创建分支并切换过去；`origin/master` 是创建起点。
- `--no-track`：先不把个人分支与远程主分支建立跟踪关系。
- `push -u`：上传个人分支，并让它跟踪远程同名分支。之后通常直接 `git push`。

上述起点不会包含只存在于本地 `master` 的提交。如果要把当前已提交但未上传的工作保留到个人分支，改用 `git switch -c dev/orbislumen`，从当前 commit 创建，再执行同样的 `push -u`。

分支已经存在时，不再使用 `-c`：

```bash
git switch dev/orbislumen
```

如果远程已有个人分支、本地尚未创建，可以使用：

```bash
git fetch origin
git switch --track origin/dev/orbislumen
```

## 3. 日常修改、验证和上传

开始新任务前，在个人分支同步小组共同版本。先确保工作区干净：

```bash
git switch dev/orbislumen
git status
git fetch origin
git merge origin/master
```

修改文件后，检查差异并执行相关验证。第一学期 MyDeZero 项目的测试方式为：

```bash
cd projl/proj
python3 -m pytest tests
cd ../..
```

下面以修改根目录 `README.md` 为例；实际提交时，将文件名替换为本次任务涉及的文件：

```bash
git diff
git add README.md
git diff --cached
git diff --cached --check
git commit -m "docs(readme): 补充项目运行说明"
git push
```

| 命令 | 含义 |
| --- | --- |
| `git diff` | 查看尚未暂存的文件改动 |
| `git add 文件名` | 选择本次 commit 包含的改动 |
| `git diff --cached` | 检查即将提交的内容 |
| `git diff --cached --check` | 检查差异中的空白错误等问题，不能代替功能测试 |
| `git commit` | 将暂存区保存为一次本地版本 |
| `git push` | 上传已经 commit 的历史，不上传尚未提交的编辑内容 |

`add` 后再次编辑文件，需要再次 `add`。已提交的工作不需要重复提交才能上传。

一条 commit 表达一个完整改动，建议使用 `类型(范围): 具体改动`：

```text
feat(mydezero): 添加 ReLU 激活函数
fix(autograd): 修复广播输入的梯度形状
test(matmul): 补充矩阵乘法梯度测试
docs(readme): 补充项目运行说明
```

修改前阅读对应学期的 `request/` 和适用的 `AGENTS.md`；参考材料放在 `reference/`，作业放在对应 `work/hw*/`，学期项目放在对应 `proj/`。避免改动无关学期，不提交个人环境、数据集和无关训练输出。

## 4. 创建、审查和合并 PR

在 GitHub 打开 `Pull requests → New pull request`，选择：

```text
base:    master
compare: dev/orbislumen
```

也可以用 GitHub CLI 创建：

```bash
gh pr create --base master --head dev/orbislumen --title "docs(readme): 补充项目运行说明"
```

PR 说明写清楚改动目的、主要内容、实际验证命令和结果，以及需要审查者关注的地方。未运行测试时说明原因，不填写未经验证的“通过”。

提交前可以检查 PR 将包含的全部改动：

```bash
git fetch origin
git log --oneline origin/master..HEAD
git diff --stat origin/master...HEAD
```

PR 包含分支相对目标分支的整体差异，不只包含最后一条 commit。一个 PR 聚焦一项任务；前一项任务尚未合并时，避免继续在同一分支混入不相关任务。

负责人在 `Files changed` 中查看实现、任务范围和验证结果，针对具体问题留言，然后提交审查：

- `Comment`：讨论或一般反馈。
- `Request changes`：有需要修正的问题。
- `Approve`：认为可以合并。

点击 `Start a review` 后还需要提交最终 review。作者不能批准自己的 PR。

收到意见后，在原个人分支修改、commit、push，原 PR 会更新；不需要再开一个 PR。新增可审查改动后，按规则重新获得批准。

长期沿用个人分支时，建议负责人选择 `Create a merge commit`。合并后，每位成员用第 3 节的 `fetch` 和 `merge origin/master` 更新自己的分支，并 push 上传同步结果。

如果小组选择 `Squash and merge`，更适合每项任务从最新 `origin/master` 创建新分支，例如 `dev/orbislumen-relu`，完成后删除任务分支，不持续沿用已 squash 的分支。

## 5. 两位审查人，以及只有两人能更新 master

必须分别控制“谁的批准有效”和“谁能执行合并”。只设置 `Required approvals = 1`，不能指定批准人；只添加 CODEOWNERS，也不会限制谁能点击合并。

### 5.1 指定两位审查负责人

在默认分支添加 `.github/CODEOWNERS`：

```text
* @OrbisLumen @Booyean
```

两位负责人必须有仓库写权限。两个人写在同一行，代表任意一位批准即可。建议在 PR 审查规则中启用 `Require review from Code Owners`，并设置批准数为 **1**。不要把“负责人有两人”理解成必须设置两次批准。

### 5.2 分成两条独立 Ruleset

建议同时应用下面两条 `Active` 的 Branch Ruleset，目标都精确选择 `master`，或选择当前默认分支。

| Ruleset | 内容 | Bypass list |
| --- | --- | --- |
| `Master review` | 必须 PR；1 次批准；必须 Code Owner 批准；新提交取消旧批准；解决审查讨论；禁止强推和删除 | 空，不给两位负责人绕过审查的权限 |
| `Master maintainers` | 只启用 `Restrict updates` | 仅 OrbisLumen 和 Booyean，模式为 `For pull requests only` |

第二条规则中，用具体用户身份添加这两人，不要添加整个 `Write` 角色；否则所有写权限成员都会获得例外。也不需要扩大 Booyean 的仓库管理权限。

GitHub 当前官方 REST API 的 Ruleset 支持 `User` 类型的 bypass actor，可用用户 ID 指定例外。如果页面不能选择具体用户，应通过受支持的接口配置并验证，或在组织仓库使用只包含这两人的团队；不要把 `Write` 角色作为替代。

这里的例外只作用于第二条“限制更新”规则。两人在第一条“审查”规则中没有例外，因此仍必须先经过 PR 和负责人批准。其他成员即使有写权限、PR 已获批准，也不应能更新 `master`。

不要把 `Restrict updates` 和全部审查要求放进同一条规则后，再允许两人绕过整条规则；那样会连审查要求一起绕过。为实现上述配置，需移除现有审查规则中的管理员 `Always allow` 例外。

配置完成后，需要用不同身份验证：普通成员能 push 个人分支但不能合并到 `master`；两位负责人在未获批准时也不能合并，获批准后可以合并；两位负责人不能直接 push 到 `master`。

仓库所有者仍拥有修改规则的管理权。这里限制的是规则保持启用时的日常更新操作，不能剥夺所有者管理仓库的能力。

### 5.3 当前已核实的状态（2026-10-06）

- 仓库属于个人账号 OrbisLumen，默认分支是 `master`。
- 已有 `Master` 规则：必须 PR、1 次批准、解决讨论、新提交取消旧批准、禁止强推和删除。
- `Require review from Code Owners` 尚未启用，管理员仍有 `Always allow` 例外。
- 本次说明没有修改 Rules，也没有添加实际的 CODEOWNERS 文件。第 5.1 和 5.2 节是待管理员实施并验证的配置。

## 6. 常见情况

- **push 被主分支规则拒绝**：检查当前分支，向自己的 `dev/用户名` 上传，再开 PR；不要尝试用强推绕过审查。
- **push 提示 non-fast-forward**：远程同名分支已有本地没有的提交，先 `fetch`，查看并合并远程同名分支，再验证和 push。
- **merge 出现冲突**：用 `git status` 找出冲突文件，理解双方改动，编辑为最终内容并删除冲突标记，再 `git add`、`git commit`、验证和 push。不要一律选择保留某一方。
- **上传到了个人分支，master 却没有变化**：这是正常情况，只有 PR 合并后，共同版本才会更新。
- **同一个 PR 改动太多**：将无关任务放在从最新 `origin/master` 创建的其他任务分支，避免一起提交审查。

## 官方资料

- [Git push 命令](https://git-scm.com/docs/git-push)
- [GitHub PR 审查](https://docs.github.com/en/pull-requests/get-started/reviewing-pull-requests-quickstart)
- [Code Owners](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners)
- [Ruleset 支持的规则](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/available-rules-for-rulesets)
- [Ruleset 创建与绕过权限](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/creating-rulesets-for-a-repository)
- [Ruleset REST API：具体用户例外](https://docs.github.com/en/rest/repos/rules#create-a-repository-ruleset)
