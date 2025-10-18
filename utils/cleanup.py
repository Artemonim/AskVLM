#!/usr/bin/env python3
"""Cleanup utility for Artemonim's Speech Kit.

This script cleans up Python cache files, build artifacts, and temporary files
to free up disk space and ensure clean development environment.
"""

import argparse
import contextlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def get_size_mb(path: str) -> float:
    """Get size of file or directory in MB."""
    p = Path(path)
    if p.is_file():
        return p.stat().st_size / (1024 * 1024)
    if p.is_dir():
        total = 0
        for child in p.rglob("*"):
            with contextlib.suppress(OSError, FileNotFoundError):
                if child.is_file():
                    total += child.stat().st_size
        return total / (1024 * 1024)
    return 0.0


def clean_python_cache(*, verbose: bool = False) -> float:
    """Clean Python cache files and return size freed in MB."""
    freed_size = 0.0

    # * Clean .pyc files
    pyc_files = list(Path().rglob("*.pyc"))
    for file in pyc_files:
        try:
            freed_size += get_size_mb(str(file))
            file.unlink(missing_ok=True)
            if verbose:
                pass
        except OSError:
            pass

    # * Clean __pycache__ directories
    for directory in Path().rglob("__pycache__"):
        try:
            freed_size += get_size_mb(str(directory))
            shutil.rmtree(directory, ignore_errors=True)
            if verbose:
                pass
        except OSError:
            pass

    return freed_size


def clean_build_artifacts(*, verbose: bool = False) -> float:
    """Clean build artifacts and return size freed in MB."""
    freed_size = 0.0

    artifacts = [
        "**/*.egg-info",
        ".coverage",
        "htmlcov",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "build",
        "dist",
        ".tox",
        ".nox",
    ]

    for pattern in artifacts:
        if pattern.startswith(".") and not pattern.startswith("*"):
            # * Single directory
            d = Path(pattern)
            if d.exists():
                freed_size += get_size_mb(pattern)
                shutil.rmtree(d, ignore_errors=True)
                if verbose:
                    pass
        else:
            # * Glob pattern - use Path.rglob for all patterns
            for match in Path().rglob(
                pattern.replace("**/", "") if pattern.startswith("**/") else pattern
            ):
                try:
                    freed_size += get_size_mb(str(match))
                    m = Path(match)
                    if m.is_dir():
                        shutil.rmtree(m, ignore_errors=True)
                    else:
                        m.unlink(missing_ok=True)
                    if verbose:
                        pass
                except OSError:
                    pass

    return freed_size


def clean_pip_cache() -> float:
    """Clean pip cache and return size freed in MB."""
    try:
        # * Get pip cache size before cleaning
        result = subprocess.run(
            [sys.executable, "-m", "pip", "cache", "info"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            shell=False,
        )

        cache_size = 0.0
        if result.returncode == 0:
            # * Parse cache info to get size
            for line in result.stdout.split("\n"):
                if "Size of cache:" in line:
                    # * Extract size (format: "Size of cache: X.X MB")
                    try:
                        size_str = line.split(":")[1].strip().split()[0]
                        cache_size = float(size_str)
                    except (IndexError, ValueError):
                        pass

        # * Purge cache
        result = subprocess.run(
            [sys.executable, "-m", "pip", "cache", "purge"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            shell=False,
        )

        if result.returncode == 0:
            return cache_size

    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return 0.0


def clean_temp_files(*, verbose: bool = False) -> float:
    """Clean temporary files and return size freed in MB."""
    freed_size = 0.0
    temp_dir = Path(tempfile.gettempdir())

    temp_prefixes = ["pip-", "tmp", "pytest-", "mypy-", "ruff-", "bandit-"]

    try:
        for item in temp_dir.iterdir():
            if any(item.name.startswith(prefix) for prefix in temp_prefixes):
                try:
                    freed_size += get_size_mb(str(item))
                    if item.is_dir():
                        shutil.rmtree(item, ignore_errors=True)
                    else:
                        item.unlink(missing_ok=True)
                    if verbose:
                        pass
                except OSError:
                    pass
    except OSError:
        if verbose:
            pass

    return freed_size


def main() -> int:
    """Run cleanup operations."""
    parser = argparse.ArgumentParser(
        description="Clean up Python cache and temporary files"
    )
    parser.add_argument(
        "--deep", action="store_true", help="Deep clean including temp files"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed output"
    )

    args = parser.parse_args()

    total_freed = 0.0

    # * Clean Python cache
    if args.verbose:
        pass
    freed = clean_python_cache(verbose=args.verbose)
    total_freed += freed
    if not args.verbose and freed > 0:
        pass

    # * Clean build artifacts
    if args.verbose:
        pass
    freed = clean_build_artifacts(verbose=args.verbose)
    total_freed += freed
    if not args.verbose and freed > 0:
        pass

    # * Clean pip cache
    if args.verbose:
        pass
    freed = clean_pip_cache()
    total_freed += freed
    if not args.verbose and freed > 0:
        pass

    # * Deep clean temp files
    if args.deep:
        if args.verbose:
            pass
        freed = clean_temp_files(verbose=args.verbose)
        total_freed += freed
        if not args.verbose and freed > 0:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
