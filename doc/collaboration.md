# Git and GitHub Collaboration

The five-member team uses one branch and one PR per task. Create a branch from
the latest `origin/master` when a task is ready to start.
This guide describes the workflow. GitHub settings determine actual permissions.

## Roles and Branches

- Push to the task branch and submit code and documentation to `master` through a PR.
- All five team members can review PRs. For each PR, at least three distinct members other than the author must approve before merge.
- Use lowercase names that describe the task, such as `feat/donkeycar-autopilot`,
  `fix/pooling-gradient`, or `docs/proj1-report`.
- Retire legacy `dev/<username>` branches from daily use. Check for unmerged work before removing any legacy branch.
- Define an owner, deliverables, and acceptance checks for each task. Only one person edits a binary document, such as Word, at a time.

## Daily Workflow

Run from the repository root after cloning.

First-time setup:

```bash
git clone https://github.com/OrbisLumen/CNN-in-automatic-drive.git
cd CNN-in-automatic-drive
git config user.name "Your Name"
git config user.email "you@example.com"
```

Before starting a new task, check that the working tree is clean, fetch
`origin`, and create the task branch. Use `git switch <branch>` to resume an
existing local task or `git switch --track origin/<branch>` to resume a remote task.

```bash
git status
git fetch origin
git switch --no-track -c feat/donkeycar-autopilot origin/master
```

Read the relevant `request/` and `AGENTS.md` before editing. Run relevant checks,
then commit selected files:

```bash
python -m pytest tests
git diff
git add README.md
git diff --cached
git diff --cached --check
git commit -m "docs(readme): simplify project overview"
git push -u origin HEAD
```

Adapt tests and files to the task. Run `git add` again after further edits;
uncommitted changes are not pushed. Each commit should cover one complete change.
Use `type(scope): description`, with types such as `feat`, `fix`, `test`, `docs`,
`refactor`, or `chore`. Avoid vague titles such as "update" or "final version".
Do not commit personal environments, data, weights, or unrelated changes.

## PRs and Merging

Check the full branch diff before creating a PR:

```bash
git fetch origin
git log --oneline origin/master..HEAD
git diff --stat origin/master...HEAD
```

On GitHub, choose `base: master` and select the task branch as the compare branch.
Follow these rules:

- Keep one task per PR and exclude unrelated changes, environments, data, weights, and temporary outputs.
- Write titles in English as `type(scope): description`, using `feat`, `fix`, `test`, `docs`, `refactor`, or `chore`. Example: `feat(data): add MNIST loader`.
- Use the [PR template](../.github/pull_request_template.md) and write concise English descriptions of the purpose and main changes.
- Record relevant test commands and results. If tests were not run, explain why. For cross-platform changes, list tested operating systems and Python versions.
- Update documentation when APIs, usage, or configuration change. List updated files, or explain why no update is needed.

Ask at least three teammates to review the changes for correctness, tests, scope
and maintainability. Reviewers submit **Approve** to record their approval.
Address change requests on the same branch. Pushing new reviewable changes
dismisses stale approvals, so request review again.

Once the PR has three current approvals, including a code owner, resolve all
review threads and complete the relevant checks. A member with write access can
then select **Create a merge commit**.

After the PR merges, sync local `master` and delete the completed local branch.
Use **Delete branch** on the merged PR if the remote branch still exists.

```bash
git switch master
git pull --ff-only origin master
git branch -d feat/donkeycar-autopilot
```

## Main Branch Rules

The active [Collaboration ruleset](https://github.com/OrbisLumen/CNN-in-automatic-drive/rules/24490741)
covers `master`. It requires a PR, three approvals including a code owner, and
resolution of review threads. It also blocks force pushes and branch deletion.
The ruleset gives repository administrators an always-on bypass. Team members,
including administrators, use the three-approval workflow.

All five members are listed in [CODEOWNERS](../.github/CODEOWNERS) for every
path. GitHub reads this file from the PR's base branch. A change to the owner
list takes effect for PRs targeting `master` once it lands there.

## Common Issues

- Main-branch push rejected: push to the task branch and open a PR.
- `non-fast-forward`: fetch and inspect the remote branch, merge, verify, then push.
- Merge conflicts: use `git status`, reconcile both changes, remove conflict markers, verify, and commit.
- Main branch unchanged after pushing a task branch: wait for the PR to merge.

References: [PR reviews](https://docs.github.com/en/pull-requests/get-started/reviewing-pull-requests-quickstart),
[CODEOWNERS](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners),
[Rulesets](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/creating-rulesets-for-a-repository),
[Rulesets API](https://docs.github.com/en/rest/repos/rules#create-a-repository-ruleset).
