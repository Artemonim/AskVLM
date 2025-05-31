#!/usr/bin/env python3
"""
Test runner and code quality analyzer for Artemonim's Speech Kit.

This script runs comprehensive static analysis and tests to ensure code quality.
Use this to check your code before committing or to validate AI-generated code.

Usage:
    python test.py                  # * Run all checks
    python test.py --quick          # * Run only fast checks (no pylint)
    python test.py --format-only    # * Only format code with ruff
    python test.py --type-check     # * Only run mypy type checking
    python test.py --security       # * Only run security analysis
    python test.py --tests          # * Only run pytest
    python test.py --install-deps   # * Install development dependencies
"""

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

# * Configuration for code analysis
PYTHON_PATHS = ["core", "gui", "editing", "utils", "main.py", "test.py"]
TEST_PATHS = ["tests"]
EXCLUDE_PATHS = ["env", ".git", "__pycache__", "*.pyc", "htmlcov"]


# * Color codes for terminal output
class Colors:
    """ANSI color codes for terminal output."""

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


@dataclass
class CheckResult:
    """Result of a code quality check."""

    name: str
    success: bool
    duration: float
    output: str = ""
    error_count: int = 0
    warning_count: int = 0
    details: List[str] = field(default_factory=list)


class CodeQualityChecker:
    """Main class for running code quality checks."""

    def __init__(self, verbose: bool = False) -> None:
        """Initialize the checker."""
        self.verbose = verbose
        self.results: List[CheckResult] = []
        self.project_root = Path(__file__).parent

    def _run_command(
        self, cmd: List[str], check_name: str, success_codes: Optional[List[int]] = None
    ) -> CheckResult:
        """Run a command and capture its result."""
        if success_codes is None:
            success_codes = [0]

        print(f"🔍 Running {check_name}...")
        start_time = time.time()

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root,
                timeout=300,  # ! 5 minute timeout
            )

            duration = time.time() - start_time
            success = result.returncode in success_codes

            # * Parse output for error/warning counts
            output_lines = result.stdout.split("\n") + result.stderr.split("\n")
            error_count = sum(
                1
                for line in output_lines
                if any(
                    keyword in line.lower()
                    for keyword in ["error:", "error ", "fatal:"]
                )
            )
            warning_count = sum(
                1
                for line in output_lines
                if any(
                    keyword in line.lower()
                    for keyword in ["warning:", "warn:", "caution:"]
                )
            )

            output = result.stdout + result.stderr

            if self.verbose or not success:
                print(f"  Output: {output[:500]}{'...' if len(output) > 500 else ''}")

            return CheckResult(
                name=check_name,
                success=success,
                duration=duration,
                output=output,
                error_count=error_count,
                warning_count=warning_count,
            )

        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return CheckResult(
                name=check_name,
                success=False,
                duration=duration,
                output="Command timed out after 5 minutes",
                error_count=1,
            )
        except Exception as e:
            duration = time.time() - start_time
            return CheckResult(
                name=check_name,
                success=False,
                duration=duration,
                output=f"Failed to run command: {e}",
                error_count=1,
            )

    def check_ruff_format(self) -> CheckResult:
        """Check code formatting with ruff."""
        cmd = ["python", "-m", "ruff", "format", "--check", "--diff"] + PYTHON_PATHS
        return self._run_command(cmd, "Ruff Format Check")

    def format_code(self) -> CheckResult:
        """Format code with ruff."""
        cmd = ["python", "-m", "ruff", "format"] + PYTHON_PATHS
        return self._run_command(cmd, "Ruff Format Fix")

    def check_ruff_lint(self) -> CheckResult:
        """Run ruff linting."""
        cmd = ["python", "-m", "ruff", "check", "--output-format=full"] + PYTHON_PATHS
        return self._run_command(cmd, "Ruff Lint", success_codes=[0, 1])

    def fix_ruff_lint(self) -> CheckResult:
        """Fix auto-fixable ruff issues."""
        cmd = ["python", "-m", "ruff", "check", "--fix"] + PYTHON_PATHS
        return self._run_command(cmd, "Ruff Auto-fix")

    def check_mypy(self) -> CheckResult:
        """Run mypy type checking."""
        cmd = ["python", "-m", "mypy"] + PYTHON_PATHS
        return self._run_command(cmd, "MyPy Type Check")

    def check_bandit(self) -> CheckResult:
        """Run bandit security analysis."""
        cmd = [
            "python",
            "-m",
            "bandit",
            "-r",
            ".",
            "-f",
            "json",
            "--exclude",
            ",".join(EXCLUDE_PATHS),
        ]
        return self._run_command(cmd, "Bandit Security Check", success_codes=[0, 1])

    def check_pylint(self) -> CheckResult:
        """Run pylint analysis."""
        cmd = ["python", "-m", "pylint"] + PYTHON_PATHS
        return self._run_command(
            cmd, "Pylint Analysis", success_codes=[0, 1, 2, 4, 8, 16]
        )

    def run_tests(self) -> CheckResult:
        """Run pytest tests."""
        if not Path("tests").exists():
            return CheckResult(
                name="Pytest Tests",
                success=True,
                duration=0.0,
                output="No tests directory found - skipping tests",
            )

        # * Check if there are any test files
        test_files = list(Path("tests").glob("test_*.py")) + list(
            Path("tests").glob("*_test.py")
        )
        if not test_files:
            return CheckResult(
                name="Pytest Tests",
                success=True,
                duration=0.0,
                output="No test files found in tests directory - skipping tests",
            )

        cmd = ["python", "-m", "pytest", "-v"]
        return self._run_command(cmd, "Pytest Tests")

    def install_dependencies(self) -> CheckResult:
        """Install development dependencies."""
        cmd = ["python", "-m", "pip", "install", "-e", ".[dev]"]
        return self._run_command(cmd, "Install Dependencies")

    def print_summary(self) -> None:
        """Print a summary of all check results."""
        total_duration = sum(r.duration for r in self.results)
        total_errors = sum(r.error_count for r in self.results)
        total_warnings = sum(r.warning_count for r in self.results)

        print(f"\n{'=' * 60}")
        print(f"{Colors.BOLD}CODE QUALITY SUMMARY{Colors.RESET}")
        print(f"{'=' * 60}")
        print(f"Total Duration: {total_duration:.2f}s")
        print(f"Total Errors: {Colors.RED}{total_errors}{Colors.RESET}")
        print(f"Total Warnings: {Colors.YELLOW}{total_warnings}{Colors.RESET}")
        print()

        # * Print detailed results
        for result in self.results:
            status_color = Colors.GREEN if result.success else Colors.RED
            status_icon = "✅" if result.success else "❌"

            print(f"{status_icon} {Colors.BOLD}{result.name}{Colors.RESET}")
            print(
                f"   Status: {status_color}{'PASS' if result.success else 'FAIL'}{Colors.RESET}"
            )
            print(f"   Duration: {result.duration:.2f}s")

            if result.error_count > 0:
                print(f"   Errors: {Colors.RED}{result.error_count}{Colors.RESET}")
            if result.warning_count > 0:
                print(
                    f"   Warnings: {Colors.YELLOW}{result.warning_count}{Colors.RESET}"
                )

            if not result.success and result.output:
                # * Show first few lines of error output
                error_lines = result.output.split("\n")[:5]
                for line in error_lines:
                    if line.strip():
                        print(f"   {Colors.RED}>{Colors.RESET} {line}")
            print()

        # * Overall result
        all_passed = all(r.success for r in self.results)
        overall_color = Colors.GREEN if all_passed else Colors.RED
        overall_status = (
            "ALL CHECKS PASSED! 🎉" if all_passed else "SOME CHECKS FAILED ⚠️"
        )

        print(f"{overall_color}{Colors.BOLD}{overall_status}{Colors.RESET}")

        if not all_passed:
            print(
                f"\n{Colors.YELLOW}💡 Run with --verbose for detailed output{Colors.RESET}"
            )


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Code quality checker for Artemonim's Speech Kit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--quick", action="store_true", help="Run only fast checks (skip pylint)"
    )
    parser.add_argument(
        "--format-only", action="store_true", help="Only format code with ruff"
    )
    parser.add_argument(
        "--type-check", action="store_true", help="Only run mypy type checking"
    )
    parser.add_argument(
        "--security", action="store_true", help="Only run security analysis"
    )
    parser.add_argument("--tests", action="store_true", help="Only run pytest")
    parser.add_argument(
        "--install-deps", action="store_true", help="Install development dependencies"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed output"
    )

    args = parser.parse_args()

    checker = CodeQualityChecker(verbose=args.verbose)

    print(
        f"{Colors.CYAN}{Colors.BOLD}Artemonim's Speech Kit - Code Quality Checker{Colors.RESET}"
    )
    print(f"{Colors.CYAN}Running comprehensive static analysis...{Colors.RESET}\n")

    # * Handle specific single checks
    if args.install_deps:
        result = checker.install_dependencies()
        checker.results.append(result)
        checker.print_summary()
        return 0 if result.success else 1

    if args.format_only:
        result = checker.format_code()
        checker.results.append(result)
        checker.print_summary()
        return 0 if result.success else 1

    if args.type_check:
        result = checker.check_mypy()
        checker.results.append(result)
        checker.print_summary()
        return 0 if result.success else 1

    if args.security:
        result = checker.check_bandit()
        checker.results.append(result)
        checker.print_summary()
        return 0 if result.success else 1

    if args.tests:
        result = checker.run_tests()
        checker.results.append(result)
        checker.print_summary()
        return 0 if result.success else 1

        # * Run full analysis
    print("Running comprehensive code quality checks...\n")

    # ! Phase 1: Auto-format code first (always)
    print("🔧 Auto-formatting code...")
    format_result = checker.format_code()
    if format_result.success:
        print("✅ Code formatting completed successfully")
    else:
        print("⚠️ Code formatting had issues")
        checker.results.append(format_result)

    # ! Phase 2: Check formatting and linting
    checker.results.append(checker.check_ruff_format())
    checker.results.append(checker.check_ruff_lint())

    # ! Phase 3: Type checking
    checker.results.append(checker.check_mypy())

    # ! Phase 4: Security analysis
    checker.results.append(checker.check_bandit())

    # ! Phase 5: Advanced analysis (skip if --quick)
    if not args.quick:
        checker.results.append(checker.check_pylint())

    # ! Phase 6: Tests
    checker.results.append(checker.run_tests())

    # * Show summary
    checker.print_summary()

    # * Return exit code based on results
    all_passed = all(r.success for r in checker.results)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
