"""Prints export statements for every non-empty key in .env.

Usage (bash/zsh):
    eval "$(uv run load-env)"

This lets the script set variables in the calling shell without a subshell,
since a Python process cannot mutate its parent's environment directly.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


def main() -> None:
    env_file = Path(".env")
    if not env_file.exists():
        print("# .env not found", file=sys.stderr)
        sys.exit(1)

    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)=(.*)$', line)
        if not m:
            continue
        key, value = m.group(1), m.group(2)
        # Strip surrounding quotes if present
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        if value:
            # Escape single quotes in value before wrapping
            safe = value.replace("'", "'\\''")
            print(f"export {key}='{safe}'")
