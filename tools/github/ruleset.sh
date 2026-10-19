#!/usr/bin/env bash
# Create or update the default-branch ruleset from ruleset.json. Run only after Task 9 is merged:
# it requires the `lint` and `All tests` checks.
set -euo pipefail
repo="Algorithmiq/tn-noise-learning"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
name="$(jq -r .name "$here/ruleset.json")"

id="$(gh api "repos/$repo/rulesets" --jq ".[] | select(.name == \"$name\") | .id")"
if [[ -n "$id" ]]; then
  gh api --method PUT "repos/$repo/rulesets/$id" --input "$here/ruleset.json" --silent
else
  gh api --method POST "repos/$repo/rulesets" --input "$here/ruleset.json" --silent
fi
printf '\033[32m✓\033[0m ruleset "%s" applied\n' "$name"
