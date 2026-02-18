#!/usr/bin/env bash
# Start ArXiv RAG Copilot with secrets injected from 1Password.
# Prerequisites: 1Password CLI (op) installed and signed in.
# Usage:
#   ./run.sh          # docker compose up (foreground)
#   ./run.sh -d       # docker compose up (detached)
#   ./run.sh down     # docker compose down

set -euo pipefail

if ! command -v op &>/dev/null; then
  echo "Error: 1Password CLI (op) not found. Install it from https://developer.1password.com/docs/cli/get-started/" >&2
  exit 1
fi

op run --env-file=.env.tpl -- docker compose up "$@"
