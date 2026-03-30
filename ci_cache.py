"""Local CI cache: SHA256 hash guards (see AgentEnforcer2 CACHING.md)."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

CI_CACHE_DIR = Path(".ci_cache")

# * Stages safe to skip without stale coverage (pytest not cached).
CACHEABLE_TOOLS = frozenset({"ruff-format", "ruff", "compile", "mypy", "pyright"})

_EXTRA_PATHS_BY_TOOL: dict[str, tuple[str, ...]] = {
    "ruff-format": ("pyproject.toml", "askvlm.defaults.json"),
    "ruff": ("pyproject.toml", "askvlm.defaults.json"),
    "compile": ("pyproject.toml", "askvlm.defaults.json"),
    "mypy": ("pyproject.toml", "askvlm.defaults.json"),
    "pyright": ("pyproject.toml", "askvlm.defaults.json"),
}


def _hash_linting_tree(h: hashlib._Hash) -> None:
    """Include all files under ``.linting/`` for lint-related cache keys."""
    lint_dir = Path(".linting")
    if not lint_dir.is_dir():
        return
    cwd = Path.cwd()
    for path in sorted(lint_dir.rglob("*")):
        if not path.is_file():
            continue
        try:
            rel = path.resolve().relative_to(cwd)
        except ValueError:
            rel = path
        h.update(rel.as_posix().encode("utf-8", errors="replace"))
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\n")


def _is_skippable_path(path: Path) -> bool:
    """Return False for venv, cache dirs, and bytecode."""
    parts = set(path.parts)
    if "__pycache__" in parts:
        return False
    if ".venv" in parts or "venv" in parts:
        return False
    if ".ci_cache" in parts:
        return False
    return ".git" not in parts


def _iter_py_files_under(targets: list[str]) -> list[Path]:
    """List tracked ``.py`` files under target dirs (sorted)."""
    out: list[Path] = []
    cwd = Path.cwd()
    for t in targets:
        p = (cwd / t).resolve()
        if p.is_file() and p.suffix == ".py":
            if _is_skippable_path(p):
                out.append(p)
        elif p.is_dir():
            matched = [f for f in p.rglob("*.py") if _is_skippable_path(f)]
            out.extend(sorted(matched))
    return sorted(set(out), key=lambda x: str(x).lower())


def compute_stage_hash(tool: str, targets: list[str]) -> str:
    """Return a SHA256 hex digest of all inputs relevant to ``tool``."""
    h = hashlib.sha256()
    if tool in {"ruff-format", "ruff", "mypy", "pyright"}:
        _hash_linting_tree(h)
    cwd = Path.cwd()
    for path in _iter_py_files_under(targets):
        try:
            rel = path.resolve().relative_to(cwd)
        except ValueError:
            rel = path
        h.update(rel.as_posix().encode("utf-8", errors="replace"))
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\n")

    extras = _EXTRA_PATHS_BY_TOOL.get(tool, ())
    for extra_rel in extras:
        p = Path(extra_rel)
        if p.is_file():
            h.update(extra_rel.encode("utf-8"))
            h.update(b"\0")
            h.update(p.read_bytes())
            h.update(b"\n")

    return h.hexdigest()


def cache_hit(tool: str, digest: str) -> bool:
    """Return True if trust stamp exists and digest matches."""
    hf = CI_CACHE_DIR / f"{tool}.sha256"
    tf = CI_CACHE_DIR / f"{tool}.trusted"
    if not hf.is_file() or not tf.is_file():
        return False
    try:
        stored = hf.read_text(encoding="utf-8").strip()
    except OSError:
        return False
    return stored == digest


def write_stage_cache(tool: str, digest: str) -> None:
    """Write hash file and trust stamp for a successful stage."""
    CI_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    hf = CI_CACHE_DIR / f"{tool}.sha256"
    tf = CI_CACHE_DIR / f"{tool}.trusted"
    hf.write_text(digest + "\n", encoding="utf-8")
    tf.touch(exist_ok=True)


def clear_all_cache() -> None:
    """Remove the entire local CI cache directory."""
    if CI_CACHE_DIR.is_dir():
        shutil.rmtree(CI_CACHE_DIR, ignore_errors=True)


def clear_stage_cache(tool: str) -> None:
    """Remove hash and trust files for one stage."""
    for suffix in (".sha256", ".trusted"):
        p = CI_CACHE_DIR / f"{tool}{suffix}"
        p.unlink(missing_ok=True)
