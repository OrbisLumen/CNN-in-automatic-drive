# Git and GitHub Collaboration

The five-member team uses **personal branch → PR → maintainer review → merge into master**.
This guide describes the workflow. GitHub settings determine actual permissions.

## Roles and Branches

| Member | Personal branch |
| --- | --- |
| OrbisLumen | `dev/orbislumen` |
| Booyean | `dev/booyean` |
| yangchunyu-1 | `dev/yangchunyu-1` |
| Qiulin087 | `dev/qiulin087` |
| lucky-money-account | `dev/lucky-money-account` |

- Push to your own branch; submit code and documentation to `master` through PRs.
- OrbisLumen and Booyean review and merge. Each reviews the other's PRs; authors cannot approve their own work.
- Reuse personal branches. Choose **Create a merge commit**, then sync with the main branch.
- Use lowercase branch names and no separate `dev` branch. Names alone do not enforce permissions.
- Define an owner, deliverables, and acceptance checks for each task. Only one person edits a binary document, such as Word, at a time.

## Daily Workflow

Run from the repository root. Replace OrbisLumen's branch with your own.

First-time setup:

```bash
git clone https://github.com/OrbisLumen/CNN-in-automatic-drive.git
cd CNN-in-automatic-drive
git config user.name "Your Name"
git config user.email "you@example.com"
git fetch origin
git switch --no-track -c dev/orbislumen origin/master
git push -u origin dev/orbislumen
```

For an existing local branch, use `git switch dev/orbislumen`.
For a remote-only branch, use `git switch --track origin/dev/orbislumen`.
To start from the current commit, use `git switch -c dev/orbislumen` instead.

Sync before starting a task and after a PR merges. Save your work and ensure a clean
working tree before switching or merging:

```bash
git status
git switch dev/orbislumen
git fetch origin
git merge origin/master
git push
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
git push
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

On GitHub, choose `base: master` and `compare: your personal branch`.
Keep one task per PR. Describe the purpose, changes, verification commands, and
results. Explain skipped tests. For cross-platform changes, list tested operating
systems and Python versions.

Maintainers review correctness, tests, scope, and maintainability.
`Comment` is not approval; address `Request changes`; `Approve` grants approval.
After `Start a review`, select `Submit review` to publish it.
Push fixes to the same branch to update the PR. New changes require fresh approval.

OrbisLumen or Booyean merges after approval, resolved discussions, relevant
verification, and configured automated checks pass. Keep unrelated tasks out of
an open PR.

## Main Branch Permissions: Target and History

Target: everyone can push personal branches; only the two maintainers can update
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

After configuration, verify that members can push personal branches but cannot
merge; maintainers can merge approved PRs but cannot merge unapproved PRs or push
directly to the main branch.

## Common Issues

- Main-branch push rejected: push to your personal branch and open a PR.
- `non-fast-forward`: fetch and inspect the remote branch, merge, verify, then push.
- Merge conflicts: use `git status`, reconcile both changes, remove conflict markers, verify, and commit.
- Main branch unchanged after a personal push: wait for the PR to merge.

References: [PR reviews](https://docs.github.com/en/pull-requests/get-started/reviewing-pull-requests-quickstart),
[CODEOWNERS](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners),
[Rulesets](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/creating-rulesets-for-a-repository),
[Rulesets API](https://docs.github.com/en/rest/repos/rules#create-a-repository-ruleset).
