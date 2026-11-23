#!/usr/bin/env bash
# Compile only; never installs, launches a GPU job, or modifies a service.
set -euo pipefail
exec python3 "$(dirname "$0")/build.py" "$@"
