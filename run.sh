#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -d .venv ]; then
    echo ".venv 가 없습니다. 먼저 ./setup.sh 를 실행하세요." >&2
    exit 1
fi

exec .venv/bin/python ui.py "$@"
