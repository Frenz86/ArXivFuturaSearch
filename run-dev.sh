#!/usr/bin/env bash
# Start ArXiv RAG Copilot in development mode with secrets from 1Password.
# Prerequisites: 1Password CLI (op) installed and signed in.
# Usage:
#   ./run-dev.sh                          # default: uvicorn with reload
#   ./run-dev.sh python -m pytest         # run tests with secrets

set -euo pipefail

if ! command -v op &>/dev/null; then
  echo "Error: 1Password CLI (op) not found. Install it from https://developer.1password.com/docs/cli/get-started/" >&2
  exit 1
fi

if [ $# -eq 0 ]; then
  op run --env-file=.env.tpl -- uvicorn app.main:app --reload
else
  op run --env-file=.env.tpl -- "$@"
fi
