#!/bin/sh
set -eu
cd -- "$(dirname -- "$0")"
if [ -x .venv/bin/python ]; then
  exec .venv/bin/python serve.py "$@"
fi
exec python3 serve.py "$@"
