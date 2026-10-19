# Contributing

Read [CONTRIBUTING.md](https://github.com/Algorithmiq/tn-noise-learning/blob/main/CONTRIBUTING.md) first: it
covers the Contributor License Agreement and the Code of Conduct.

## Development environment

Pick one:

- **Devcontainer**: open the repository in VS Code with the Dev Containers extension. The container has uv and just,
  syncs the environment and installs the git hooks.
- **Nix**: `nix develop` (or `direnv allow`) gives a shell with Python, uv, just and lychee.
- **By hand**: install [uv](https://docs.astral.sh/uv/) and [just](https://github.com/casey/just), then
  `just build --all-groups && uv run prek install`.

## Everyday commands

| Command | What it does |
| --- | --- |
| `just build --all-groups` | `uv sync` with every extra and group |
| `just test` / `just test -m "not slow"` | the test suite |
| `just fmt` | format and apply safe lint fixes |
| `just lint` | every hook (ruff, ty, licence headers, …) on every file |
| `just typecheck` | ty |
| `just docs` / `just serve-docs` | build / serve the documentation |
| `just doctest` | run the code blocks of the README and the docs |
| `just example` | run `examples/minimal_example.py` |

CI runs the same recipes, so a failing check can be reproduced locally with the recipe it ran.

## Pull requests

- Commits and PR titles follow `<type>(<optional scope>): <gitmoji> <description>`, with types `feat`, `fix`,
  `docs`, `style`, `refactor`, `test`, `chore`, `build`, `ci`, and `!` for breaking changes.
- Fill in the pull request template, including the AI/LLM disclosure.
- `lint` and `All tests` must pass and one maintainer must approve; PRs are squash-merged.
