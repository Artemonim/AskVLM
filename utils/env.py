from __future__ import annotations

import os
from pathlib import Path


def load_env_file(path: str | Path = ".env") -> None:
    """Load simple KEY=VALUE pairs from a .env file into os.environ.

    - Ignores lines starting with '#'
    - Strips quotes around values
    - Does not overwrite existing environment variables
    """
    p = Path(path)
    if not p.exists():
        return
    try:
        for raw in p.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            key = k.strip()
            val = v.strip().strip('"').strip("'")
            os.environ.setdefault(key, val)
    except OSError:
        # Silent fallback; env loading is best-effort
        return
