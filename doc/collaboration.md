# Git and GitHub Collaboration

The five-member team uses one branch and one PR per task. Create a branch from
the latest `origin/master` when a task is ready to start.
This guide describes the workflow. GitHub settings determine actual permissions.

## Roles and Branches

- Push to the task branch and submit code and documentation to `master` through a PR.
- OrbisLumen and Booyean review and merge. Each reviews the other's PRs; authors cannot approve their own work.
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
python -m pytest projl/proj/tests
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

Maintainers review correctness, tests, scope, and maintainability.
`Comment` is not approval; address `Request changes`; `Approve` grants approval.
After `Start a review`, select `Submit review` to publish it.
Push fixes to the same branch to update the PR. New changes require fresh approval.

OrbisLumen or Booyean merges after approval, resolved discussions, relevant
verification, and configured automated checks pass. Use **Create a merge commit**.

After the PR merges, sync local `master` and delete the completed local branch.
Use **Delete branch** on the merged PR if the remote branch still exists.

```bash
git switch master
git pull --ff-only origin master
git branch -d feat/donkeycar-autopilot
```

## Main Branch Permissions: Target and History

Target: everyone can push task branches; only the two maintainers can update
`master` through approved PRs. Maintainers cannot push directly either.
Review and merge restrictions need separate configuration; CODEOWNERS alone does
not restrict who can merge.

The default branch's `.github/CODEOWNERS` should contain:

```text
* @OrbisLumen @Booyean
```

Both maintainers need write access. Use two rulesets targeting `master`:

| Rule | Settings | Bypass |
| --- | --- | --- |
| `Master review` | Require PR, one Code Owner approval, dismiss stale approvals, resolve discussions, block force pushes and deletion | None |
| `Master maintainers` | `Restrict updates` only | The two named maintainers, `For pull requests only` |

Do not bypass the entire Write role. Remove administrator `Always allow` from the
review rule. If individual users cannot be selected, check official API support
or use a maintainer team in an organization repository. Owners can still change rules.

**Historical check on 2026-10-06, not live status:** PRs, one approval, Code Owner
approval, resolved discussions, stale approval dismissal, and force-push/deletion
blocks were enabled. CODEOWNERS was submitted in
[PR #3](https://github.com/OrbisLumen/CNN-in-automatic-drive/pull/3), then pending merge.
CODEOWNERS takes effect only after entering the target branch. Administrator
`Always allow` remained; maintainer-only updates were not configured or verified.
Editing this guide does not change remote rules.

After configuration, verify that members can push task branches but cannot
merge; maintainers can merge approved PRs but cannot merge unapproved PRs or push
directly to the main branch.

## Common Issues

- Main-branch push rejected: push to the task branch and open a PR.
- `non-fast-forward`: fetch and inspect the remote branch, merge, verify, then push.
- Merge conflicts: use `git status`, reconcile both changes, remove conflict markers, verify, and commit.
- Main branch unchanged after pushing a task branch: wait for the PR to merge.

References: [PR reviews](https://docs.github.com/en/pull-requests/get-started/reviewing-pull-requests-quickstart),
[CODEOWNERS](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners),
[Rulesets](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/creating-rulesets-for-a-repository),
[Rulesets API](https://docs.github.com/en/rest/repos/rules#create-a-repository-ruleset).
