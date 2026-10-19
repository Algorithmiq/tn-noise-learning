#!/usr/bin/env python3

# Copyright 2026 Algorithmiq
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fail when a workflow step runs a command that belongs in the justfile.

Workflows choose matrices, environment and artifacts; the commands themselves live in
recipes, so that a failing lane can be reproduced locally by running what it ran. This
guards that split, which nothing else can: a copy pasted back into a `run:` block works
perfectly well until it drifts from the recipe it duplicates.

The exceptions in ``ALLOWED`` are keyed by step name, so renaming an exempt step fails
here and forces the exception to be reconsidered rather than inherited silently.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

# Substrings are matched case-sensitively against the whole `run:` block. Each maps to
# what to do instead.
FORBIDDEN: dict[str, str] = {
    "uv sync": "install with `just build [uv sync args…]`",
    "uv run": "call the recipe that owns the command",
    "uv build": "use `just dist`",
    "uv tool run": "add a recipe for the tool being run",
    "uvx ": "add a recipe for the tool being run",
    "pytest": "use `just test` / `just test-ci` / `just coverage` / `just doctest`",
    "mkdocs": "use `just docs`",
    "prek run": "use `just lint`",
    "ty check": "use `just typecheck`",
    "ruff ": "use `just fmt` / `just lint`",
    "lychee": "use `just check-doc-links` locally or lycheeverse/lychee-action in CI",
    "apt-get": "install tools in .github/actions/setup",
    "brew install": "install tools in .github/actions/setup",
}

# (workflow file name, step name) -> why the step may keep its commands.
ALLOWED: dict[tuple[str, str], str] = {}

# Version and help queries are not builds.
BENIGN = re.compile(r"\b(uv|ruff|ty|lychee|just)\s+--(version|help)\b")


def steps_of(document: object) -> list[tuple[str, str]]:
    """Return every (step name, run block) pair in a workflow document."""
    jobs = document.get("jobs") if isinstance(document, dict) else None
    if not isinstance(jobs, dict):
        return []

    found: list[tuple[str, str]] = []
    for job in jobs.values():
        if not isinstance(job, dict):
            continue
        for step in job.get("steps") or []:
            if not isinstance(step, dict):
                continue
            run = step.get("run")
            if isinstance(run, str):
                found.append((str(step.get("name", "<unnamed step>")), run))
    return found


def check(path: Path) -> list[str]:
    """Return one message per forbidden command found in ``path``."""
    document = yaml.safe_load(path.read_text())
    steps = steps_of(document)
    problems = []
    for name, run in steps:
        if (path.name, name) in ALLOWED:
            continue
        body = BENIGN.sub("", run)
        for command, remedy in FORBIDDEN.items():
            if command in body:
                problems.append(f"{path}: step '{name}' runs '{command.strip()}' — {remedy}")

    # An exception that no longer matches a step is stale: the step was renamed or the
    # commands were moved into a recipe after all.
    names = {name for name, _ in steps}
    problems.extend(
        f"{path}: no step named '{step}'; drop that entry from ALLOWED"
        for file_name, step in ALLOWED
        if file_name == path.name and step not in names
    )
    return problems


def main() -> int:
    """Check every workflow named on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", type=Path)
    args = parser.parse_args()

    problems = [message for path in args.files for message in check(path)]
    if not problems:
        return 0

    sys.stderr.write("\n".join(problems) + "\n")
    sys.stderr.write(
        "\nCommands belong in the justfile; workflows choose matrices, environment and "
        "artifacts. Add the exception to ALLOWED in tools/check-workflow-commands.py if a "
        "step genuinely cannot use a recipe.\n",
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
