#!/usr/bin/env bash
# Apply the standard repository settings. Idempotent; run as a repository admin.
set -euo pipefail
repo="Algorithmiq/tn-noise-learning"

gh repo edit "$repo" \
  --description "Learn quantum noise channels from measurement shots with tensor networks" \
  --homepage "https://algorithmiq.github.io/tn-noise-learning/" \
  --enable-issues --enable-discussions --enable-wiki=false --enable-projects=false \
  --enable-squash-merge --enable-merge-commit=false --enable-rebase-merge=false \
  --enable-auto-merge --delete-branch-on-merge --allow-update-branch \
  --add-topic quantum-computing --add-topic noise-learning --add-topic tensor-networks \
  --add-topic quantum-process-tomography --add-topic jax

# deployment environment for trusted publishing (Task 12)
gh api --method PUT "repos/$repo/environments/pypi" --silent

# GitHub Pages built by a workflow (Task 13)
if gh api "repos/$repo/pages" --silent 2>/dev/null; then
  gh api --method PUT "repos/$repo/pages" -f build_type=workflow --silent
else
  gh api --method POST "repos/$repo/pages" -f build_type=workflow --silent
fi
printf '\033[32m✓\033[0m settings applied\n'
