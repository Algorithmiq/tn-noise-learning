#!/usr/bin/env bash
set -euo pipefail

just build --all-groups
uv run --no-sync prek install --install-hooks
