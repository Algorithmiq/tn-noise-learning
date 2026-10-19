set shell := ["bash", "-eu", "-o", "pipefail", "-c"]
set positional-arguments := true

root := justfile_directory()

# List the recipes.
default:
    @just --list

# Install the package with every extra; extra arguments go to `uv sync` (e.g. `just build --group docs`).
build *ARGS:
    uv sync --all-extras "$@"

# Format the code and apply the safe lint fixes.
fmt:
    uv run --no-sync ruff format .
    uv run --no-sync ruff check --fix .

# Run every hook on the whole tree, as the lint workflow does on a PR's commits.
lint:
    uv run --no-sync prek run --all-files

# Type-check with ty.
typecheck:
    uv run --no-sync ty check

# Run the test suite; extra arguments go to pytest (e.g. `just test -m "not slow"`).
test *ARGS:
    uv run --no-sync pytest "$@"

# CI leg: the test suite with a JUnit report named after LABEL.
test-ci LABEL:
    uv run --no-sync pytest -r aR --durations=20 --durations-min=5.0 \
        --junit-xml=pytest-{{ LABEL }}.xml -o junit_family=legacy

# The test suite with branch coverage, written to coverage.xml.
coverage:
    uv run --no-sync pytest --cov --cov-report=xml:coverage.xml --cov-report=term-missing

# Run the Python code blocks of README.md and docs/ in a scratch directory (they write files).
doctest:
    #!/usr/bin/env bash
    set -euo pipefail
    cd "$(mktemp -d)"
    # --markdown-docs imports every .py it is given; gen_ref_pages.py only runs inside a mkdocs build
    uv run --project "{{ root }}" --no-sync pytest --markdown-docs --markdown-docs-syntax=superfences \
        -p no:cacheprovider --rootdir "{{ root }}" -c "{{ root }}/pyproject.toml" \
        --ignore="{{ root }}/docs/gen_ref_pages.py" \
        "{{ root }}/README.md" "{{ root }}/docs"

# Run the examples in a scratch directory (they write files).
example:
    #!/usr/bin/env bash
    set -euo pipefail
    cd "$(mktemp -d)"
    uv run --project "{{ root }}" --no-sync python "{{ root }}/examples/minimal_example.py"

# Build the documentation into site/.
docs:
    uv run --no-sync mkdocs build --strict

# Serve the documentation with live reload.
serve-docs:
    uv run --no-sync mkdocs serve

# Check the links of the built site, external ones included. Needs `lychee` on PATH.
check-doc-links: docs
    lychee --config .lychee.toml 'site/**/*.html'

# Build the sdist and wheel into dist/.
dist:
    rm -rf dist
    uv build --out-dir dist

# Move a pinned astral tool (ruff or ty) and its prek hook to VERSION together. GNU sed.
bump-tool TOOL VERSION:
    sed -i -E 's/"{{ TOOL }}==[0-9.]+"/"{{ TOOL }}=={{ VERSION }}"/' pyproject.toml
    sed -i -E '/astral-sh\/{{ TOOL }}-pre-commit"/{n; s/rev = "v[0-9.]+"/rev = "v{{ VERSION }}"/}' prek.toml
    uv lock --upgrade-package {{ TOOL }}
