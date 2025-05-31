#!/usr/bin/env python3
"""
Cleanup utility for Artemonim's Speech Kit.

This script cleans up Python cache files, build artifacts, and temporary files
to free up disk space and ensure clean development environment.
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile


def get_size_mb(path: str) -> float:
    """Get size of file or directory in MB."""
    if os.path.isfile(path):
        return os.path.getsize(path) / (1024 * 1024)
    if os.path.isdir(path):
        total = 0
        for dirpath, _dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total += os.path.getsize(filepath)
                except (OSError, FileNotFoundError):
                    pass
        return total / (1024 * 1024)
    return 0.0


def clean_python_cache(verbose: bool = False) -> float:
    """Clean Python cache files and return size freed in MB."""
    freed_size = 0.0

    # * Clean .pyc files
    pyc_files = glob.glob("**/*.pyc", recursive=True)
    for file in pyc_files:
        try:
            freed_size += get_size_mb(file)
            os.remove(file)
            if verbose:
                print(f"  Removed: {file}")
        except OSError:
            pass

    # * Clean __pycache__ directories
    pycache_dirs = glob.glob("**/__pycache__", recursive=True)
    for directory in pycache_dirs:
        try:
            freed_size += get_size_mb(directory)
            shutil.rmtree(directory, ignore_errors=True)
            if verbose:
                print(f"  Removed: {directory}")
        except OSError:
            pass

    return freed_size


def clean_build_artifacts(verbose: bool = False) -> float:
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
            if os.path.exists(pattern):
                freed_size += get_size_mb(pattern)
                shutil.rmtree(pattern, ignore_errors=True)
                if verbose:
                    print(f"  Removed: {pattern}")
        else:
            # * Glob pattern
            matches = glob.glob(pattern, recursive=True)
            for match in matches:
                try:
                    freed_size += get_size_mb(match)
                    if os.path.isdir(match):
                        shutil.rmtree(match, ignore_errors=True)
                    else:
                        os.remove(match)
                    if verbose:
                        print(f"  Removed: {match}")
                except OSError:
                    pass

    return freed_size


def clean_pip_cache(verbose: bool = False) -> float:
    """Clean pip cache and return size freed in MB."""
    try:
        # * Get pip cache size before cleaning
        result = subprocess.run(
            [sys.executable, "-m", "pip", "cache", "info"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
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
        )

        if result.returncode == 0:
            if verbose:
                print(f"  Pip cache purged (freed ~{cache_size:.1f} MB)")
            return cache_size
        else:
            if verbose:
                print("  Pip cache purge not available")
            return 0.0

    except (subprocess.TimeoutExpired, FileNotFoundError):
        if verbose:
            print("  Could not access pip cache")
        return 0.0


def clean_temp_files(verbose: bool = False) -> float:
    """Clean temporary files and return size freed in MB."""
    freed_size = 0.0
    temp_dir = tempfile.gettempdir()

    temp_prefixes = ["pip-", "tmp", "pytest-", "mypy-", "ruff-", "bandit-"]

    try:
        for item in os.listdir(temp_dir):
            if any(item.startswith(prefix) for prefix in temp_prefixes):
                item_path = os.path.join(temp_dir, item)
                try:
                    freed_size += get_size_mb(item_path)
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path, ignore_errors=True)
                    else:
                        os.remove(item_path)
                    if verbose:
                        print(f"  Removed temp: {item_path}")
                except OSError:
                    pass
    except OSError:
        if verbose:
            print("  Could not access temp directory")

    return freed_size


def main() -> int:
    """Main entry point."""
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

    print("🧹 Cleaning up Python cache and build artifacts...")

    total_freed = 0.0

    # * Clean Python cache
    if args.verbose:
        print("\n📁 Cleaning Python cache files...")
    freed = clean_python_cache(args.verbose)
    total_freed += freed
    if not args.verbose and freed > 0:
        print(f"  Python cache: {freed:.1f} MB")

    # * Clean build artifacts
    if args.verbose:
        print("\n🔨 Cleaning build artifacts...")
    freed = clean_build_artifacts(args.verbose)
    total_freed += freed
    if not args.verbose and freed > 0:
        print(f"  Build artifacts: {freed:.1f} MB")

    # * Clean pip cache
    if args.verbose:
        print("\n📦 Cleaning pip cache...")
    freed = clean_pip_cache(args.verbose)
    total_freed += freed
    if not args.verbose and freed > 0:
        print(f"  Pip cache: {freed:.1f} MB")

    # * Deep clean temp files
    if args.deep:
        if args.verbose:
            print("\n🗑️  Deep cleaning temporary files...")
        freed = clean_temp_files(args.verbose)
        total_freed += freed
        if not args.verbose and freed > 0:
            print(f"  Temp files: {freed:.1f} MB")

    print(f"\n✅ Cleanup complete! Freed {total_freed:.1f} MB of disk space.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
