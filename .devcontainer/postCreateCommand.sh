#!/usr/bin/env bash
set -euxo pipefail

echo 'eval "$(uv generate-shell-completion zsh)"' >> /home/vscode/.zshrc
echo 'eval "$(uvx --generate-shell-completion zsh)"' >> /home/vscode/.zshrc

mkdir -p ~/.zsh/completions
just --completions zsh > ~/.zsh/completions/_just
cat <<'EOT' >> /home/vscode/.zshrc
fpath=(~/.zsh/completions $fpath)
autoload -U compinit
compinit
EOT
