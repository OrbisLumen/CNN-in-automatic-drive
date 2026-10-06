# MyDeZero Working Guidelines

Follow the repository-level `AGENTS.md` along with these project preferences.

1. Analyze requested changes and explain simple code suggestions before editing.
2. Keep implementations small, readable, and close to the local DeZero reference.
   Preserve necessary correctness fixes without adding unnecessary complexity.
3. Avoid extensive defensive input validation in teaching functions. Document
   expected inputs and use NumPy's normal behavior unless explicit checks are
   requested or needed to fix a concrete bug.
4. Keep `README.md` focused on the overview, project structure, setup, and tests.
   Put detailed examples, gradient conventions, and implementation comparisons
   in `notes/`, such as `notes/usage.md`, rather than extending the README.
5. Keep docstrings concise and useful; avoid repeating explanations in several
   places. Explain why substantial documentation is needed before adding it.
6. Test actual behavior and correctness. Avoid tests added solely to support
   unnecessary validation; update relevant tests when simplifying code.
7. When commits are requested, commit functionality first, tests second, and
   documentation third, as separate commits.
8. Keep changes narrowly scoped to this project.
