#!/usr/bin/env bash
# Offline gate planning/comparison. GPU execution remains a separately owned,
# serialized runner; this command never launches inference or changes a service.
set -euo pipefail
exec python3 "$(dirname "$0")/gate.py" "$@"
