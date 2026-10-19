#!/usr/bin/env bash
# Create or update the labels the workflows and templates use. Idempotent.
set -euo pipefail
repo="Algorithmiq/tn-noise-learning"

labels=(
  "bug|d73a4a|Something isn't working"
  "enhancement|a2eeef|New feature or request"
  "documentation|0075ca|Improvements or additions to documentation"
  "performance|917d6d|Memory and/or CPU usage improvements"
  "python|3572a5|Python sources or tests"
  "ci|6f42c1|GitHub Actions and the justfile"
  "dependencies|0366d6|Dependency changes"
  "tools|c5def5|Developer tooling"
  "nix|7ebae4|Nix flake"
  "devcontainer|1d76db|Devcontainer and editor settings"
  "refactor|feef03|Codebase refactoring"
  "lo-prio|8de821|Low priority"
  "ruff-update|d4c5f9|Automated ruff bump"
  "ty-update|d4c5f9|Automated ty bump"
  "automation|ededed|Opened by a workflow"
)

for entry in "${labels[@]}"; do
  IFS='|' read -r name color description <<< "$entry"
  gh label create "$name" --repo "$repo" --color "$color" --description "$description" --force
done
