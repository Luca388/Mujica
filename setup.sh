#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_VERSION="3.12"

if ! command -v uv > /dev/null; then
    echo "cannot find uv" >&2
    exit 1
fi

if ! command -v ffmpeg > /dev/null; then
    echo "cannot find ffmpeg" >&2
    exit 1
fi

uv python install "$PYTHON_VERSION"
uv venv --python "$PYTHON_VERSION" .venv
uv pip install --python .venv -r requirements.txt

echo "complete."
