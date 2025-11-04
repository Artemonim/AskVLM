#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Artemonim's Speech Kit Local CI (build.py)

- Auto-fixes formatting with ruff before analysis.
- Uses configs from .linting/
- Runs: ruff-format → ruff → compile → mypy → pyright → pytest → bandit (optional) → pip-audit (optional)

Usage examples:
  python build.py
  python build.py --tool ruff
  python build.py --path core editing
  python build.py --json
  # * Launch control:
  #   --skip-launch  -> run checks/tests only (no app launch)
  #   --fast-launch  -> launch app only (skip checks/tests)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, TextIO
import xml.etree.ElementTree as ET


def linting_path() -> str:
    path = ".linting"
    return path


LINTING_PATH = linting_path()


TOOLS: Dict[str, Dict[str, Any]] = {
    "ruff-format": {
        "command": [sys.executable, "-m", "ruff"],
        "args": ["format", "--check"],
        "args_fix": ["format"],
        "can_fix": True,
        "critical": False,
        "desc": "Ruff formatter",
    },
    "cuda": {
        "command": [sys.executable, "-c"],
        # * Checks CUDA availability in the current venv; fails CI if not available
        "args": [
            (
                "import sys;\n"
                "try:\n"
                "    import torch\n"
                "    ok = bool(getattr(torch,'cuda',None) and torch.cuda.is_available())\n"
                "    dev = torch.cuda.get_device_name(0) if ok else 'N/A'\n"
                "    cver = getattr(torch.version, 'cuda', None)\n"
                "    ver = getattr(torch, '__version__', '?')\n"
                "    print('[CUDA CHECK] available={0} device={1} cuda={2} torch={3}'.format(ok, dev, cver, ver))\n"
                "    sys.exit(0 if ok else 1)\n"
                "except Exception as e:\n"
                "    print('[CUDA CHECK] error:', e)\n"
                "    sys.exit(1)\n"
            )
        ],
        "can_fix": False,
        "critical": True,
        "desc": "CUDA availability check",
    },
    "ruff": {
        "command": [sys.executable, "-m", "ruff"],
        "args": ["check", "--output-format=concise", "--ignore", "E501"],
        "args_fix": ["check", "--fix", "--unsafe-fixes", "--output-format=concise", "--ignore", "E501"],
        "can_fix": True,
        "critical": True,
        "desc": "Ruff linter",
    },
    "compile": {
        "command": [sys.executable, "-m", "compileall"],
        "args": ["-q"],
        "can_fix": False,
        "critical": True,
        "desc": "Syntax compilation",
    },
    "mypy": {
        "command": [sys.executable, "-m", "mypy"],
        "args": [f"--config-file={LINTING_PATH}/mypy.ini"],
        "can_fix": False,
        "critical": True,
        "desc": "Type checking",
    },
    "pyright": {
        "command": ["npx", "--yes", "pyright"],
        "args": ["--outputjson", "--project", f"{LINTING_PATH}/pyrightconfig.json"],
        "can_fix": False,
        "critical": True,
        "desc": "Static types (Pyright)",
    },
    "pytest": {
        "command": [sys.executable, "-m", "pytest"],
        # * Coverage collection is configured via pyproject.toml addopts
        # * Keep --cov-fail-under=0 to avoid pytest failing; gating is done below
        "args": ["-q", "--maxfail=1", "--cov-fail-under=0"],
        "can_fix": False,
        "critical": True,
        "desc": "Unit tests",
    },
    "bandit": {
        "command": [sys.executable, "-m", "bandit"],
        "args": ["-q", "-f", "json"],
        "can_fix": False,
        "critical": False,
        "desc": "Security linter",
    },
    "pip-audit": {
        "command": [sys.executable, "-m", "pip_audit"],
        "args": [".", "-f", "json", "--progress-spinner", "off"],
        "can_fix": False,
        "critical": False,
        "desc": "Dependency audit",
    },
}


DEFAULT_TARGETS = ["core", "editing", "utils", "gui", "tests"]


class LocalCI:
    def __init__(self, verbose: bool, json_output: bool, targets: Optional[List[str]]):
        self.verbose = verbose
        self.json_output = json_output
        self.targets = targets or DEFAULT_TARGETS
        self.results: Dict[str, Any] = {}
        self.start_time = time.time()
        # Initialize log file
        self.log_file: TextIO = open("build.log", "w", encoding="utf-8")
        self._log(f"Build started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self._log(f"Targets: {', '.join(self.targets)}")
        self._log("=" * 50)
        # * Coverage gates
        self.coverage_warn_threshold: float = 75.0
        self.coverage_fail_threshold: float = 65.0
        self.coverage_xml_path: str = "coverage.xml"

    def _print(self, msg: str) -> None:
        if not self.json_output:
            print(msg)

    def _log(self, msg: str) -> None:
        """Write message to build.log"""
        self.log_file.write(msg + "\n")
        self.log_file.flush()

    def __del__(self):
        """Close log file when object is destroyed"""
        if hasattr(self, 'log_file') and self.log_file:
            self._log(f"Build finished at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            self.log_file.close()

    def _run(self, cmd: List[str]) -> Tuple[int, str, str]:
        try:
            if self.verbose and not self.json_output:
                self._print(f"* Running: {' '.join(cmd)}")
            shell = os.name == "nt" and cmd[0] in {"npx", "npm"}
            p = subprocess.run(cmd, capture_output=True, text=True, shell=shell, timeout=300)
            return p.returncode, p.stdout, p.stderr
        except subprocess.TimeoutExpired:
            return 124, "", "Command timed out"
        except FileNotFoundError:
            return 127, "", f"Command not found: {cmd[0]}"
        except Exception as e:
            return 1, "", str(e)

    def _available(self, tool: str) -> bool:
        base = TOOLS[tool]["command"]
        # * For Python modules, many do not support --version (e.g., compileall). Assume available.
        if base and base[0] == sys.executable:
            return True
        probe = base[:1] + ["--help"]
        code, _, _ = self._run(probe)
        return code == 0

    def _should_run_pip_audit(self) -> bool:
        """Check if pip-audit should run (only once per day)."""
        last_run_file = ".pip_audit_last_run"
        try:
            if os.path.exists(last_run_file):
                with open(last_run_file, "r", encoding="utf-8") as f:
                    last_run_str = f.read().strip()
                    last_run = float(last_run_str)
                    # Check if 24 hours (86400 seconds) have passed
                    if time.time() - last_run < 86400:
                        return False
        except (ValueError, OSError):
            # If file is corrupted or can't be read, allow run
            pass
        return True

    def _update_pip_audit_timestamp(self) -> None:
        """Update the timestamp of last pip-audit run."""
        last_run_file = ".pip_audit_last_run"
        try:
            with open(last_run_file, "w", encoding="utf-8") as f:
                f.write(str(time.time()))
        except OSError:
            # If can't write, silently fail - next run will try again
            pass

    # * Dependency freshness check (runs at most once per day)
    def _should_run_deps_check(self) -> bool:
        last_run_file = ".deps_last_check"
        try:
            if os.path.exists(last_run_file):
                with open(last_run_file, "r", encoding="utf-8") as f:
                    last_run_str = f.read().strip()
                    last_run = float(last_run_str)
                    if time.time() - last_run < 86400:
                        return False
        except (ValueError, OSError):
            pass
        return True

    def _update_deps_check_timestamp(self) -> None:
        last_run_file = ".deps_last_check"
        try:
            with open(last_run_file, "w", encoding="utf-8") as f:
                f.write(str(time.time()))
        except OSError:
            pass

    def _build_command(self, tool: str, fix: bool) -> List[str]:
        cfg = TOOLS[tool]
        cmd = list(cfg["command"])  # copy
        if fix and cfg.get("can_fix"):
            cmd += cfg.get("args_fix", [])
            cmd += self.targets
        else:
            cmd += cfg.get("args", [])
            if tool in {"ruff", "ruff-format", "compile", "pytest", "bandit"}:
                cmd += self.targets
            elif tool == "pyright":
                # Keep default "." unless specific targets provided
                if self.targets != DEFAULT_TARGETS:
                    cmd = [c for c in cmd if c != "."] + self.targets
            elif tool == "mypy":
                add = [p for p in self.targets if p != "tests"]
                if add and add != DEFAULT_TARGETS:
                    cmd += add
        return cmd

    def run_tool(self, name: str, fix: bool) -> Dict[str, Any]:
        if not self._available(name):
            return {"tool": name, "available": False, "exit_code": 127, "error": f"{name} not available", "exec_time": 0.0}
        cmd = self._build_command(name, fix)
        start_time = time.time()
        code, out, err = self._run(cmd)
        exec_time = round(time.time() - start_time, 2)
        res: Dict[str, Any] = {
            "tool": name,
            "available": True,
            "exit_code": code,
            "stdout": out,
            "stderr": err,
            "critical": TOOLS[name]["critical"],
            "fixed": fix and TOOLS[name].get("can_fix", False),
            "exec_time": exec_time,
        }
        # Normalize pytest: treat "no tests ran" as success
        if name == "pytest":
            text = (out or "") + "\n" + (err or "")
            if res["exit_code"] != 0 and "no tests ran" in text.lower():
                res["exit_code"] = 0
                res["stdout"] = text
        if name == "pyright" and out:
            try:
                data = json.loads(out)
                res["summary"] = data.get("summary", {})
                res["diagnostics"] = data.get("generalDiagnostics", [])
            except json.JSONDecodeError:
                res["parse_error"] = "pyright JSON parse error"
        if name == "bandit" and out:
            try:
                b = json.loads(out)
                sev = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
                for it in b.get("results", []) or []:
                    k = str(it.get("issue_severity", "")).upper()
                    if k in sev:
                        sev[k] += 1
                res["bandit_summary"] = sev
                res["bandit_results"] = b.get("results", []) or []
                if sev.get("HIGH", 0) > 0:
                    res["exit_code"] = 1
                    res["critical"] = True
            except json.JSONDecodeError:
                res["parse_error"] = "bandit JSON parse error"
        return res

    def run(self, only: Optional[str], fix: bool) -> Dict[str, Any]:
        tools = [only] if only else [
            "ruff-format", "ruff", "compile", "mypy", "pyright", "pytest", "cuda", "bandit", "pip-audit"
        ]

        if not self.json_output:
            self._print("==================== Local CI ====================")
            self._print(f"Targets: {', '.join(self.targets)}")
            if fix:
                self._print("Auto-fix enabled")

        if fix and not only:
            for t in ["ruff-format", "ruff"]:
                if not self.json_output:
                    self._print(f"Auto-fixing with {t}...")
                result = self.run_tool(t, True)
                self.results[t] = result
                if not self.json_output:
                    exec_time = result.get("exec_time", 0)
                    self._print(f"✓ {t} completed in {exec_time}s")

                # Log auto-fix results
                self._log(f"\n--- {t.upper()} AUTO-FIX ---")
                self._log(f"Exit code: {result.get('exit_code', 'unknown')}")
                self._log(f"Execution time: {result.get('exec_time', 0)}s")
                stdout = result.get("stdout", "")
                stderr = result.get("stderr", "")
                if stdout:
                    self._log("STDOUT:")
                    self._log(stdout)
                if stderr:
                    self._log("STDERR:")
                    self._log(stderr)

        # * Optional: check for outdated dependencies once per day (non-critical)
        if self._should_run_deps_check():
            code, out, err = self._run([sys.executable, "-m", "pip", "list", "--outdated", "--format=json"])
            self._log("\n--- DEPENDENCY FRESHNESS CHECK ---")
            self._log(f"Exit code: {code}")
            if out:
                self._log("STDOUT:")
                self._log(out)
            if err:
                self._log("STDERR:")
                self._log(err)
            if code == 0:
                try:
                    data = json.loads(out or "[]")
                    outdated_count = len(data) if isinstance(data, list) else 0
                    self._log(f"Outdated packages: {outdated_count}")
                except json.JSONDecodeError:
                    self._log("Could not parse pip outdated JSON output")
            self._update_deps_check_timestamp()

        for t in tools:
            if t in {"ruff-format", "ruff"} and fix and not only:
                continue

            # Skip pip-audit if it was run less than 24 hours ago
            if t == "pip-audit" and not self._should_run_pip_audit():
                if not self.json_output:
                    self._print("-- pip-audit --")
                    self._print("⏭️  pip-audit skipped (run once per day)")
                self.results[t] = {
                    "tool": t,
                    "available": True,
                    "exit_code": 0,
                    "stdout": "",
                    "stderr": "",
                    "critical": TOOLS[t]["critical"],
                    "fixed": False,
                    "exec_time": 0.0,
                    "skipped": True,
                }
                # Log skipped pip-audit
                self._log(f"\n--- PIP-AUDIT ---")
                self._log("Skipped: run once per day")
                continue

            if not self.json_output:
                self._print(f"-- {t} --")
            result = self.run_tool(t, False)
            self.results[t] = result

            # Update pip-audit timestamp if it ran successfully
            if t == "pip-audit" and result.get("exit_code") == 0:
                self._update_pip_audit_timestamp()

            if not self.json_output:
                exec_time = result.get("exec_time", 0)
                status = "✓" if result.get("exit_code", 0) == 0 else "✗"
                self._print(f"{status} {t} completed in {exec_time}s")

            # Log detailed results
            self._log(f"\n--- {t.upper()} ---")
            self._log(f"Exit code: {result.get('exit_code', 'unknown')}")
            self._log(f"Execution time: {result.get('exec_time', 0)}s")

            if result.get("available") is False:
                self._log(f"Tool not available: {result.get('error', 'Unknown error')}")
            else:
                stdout = result.get("stdout", "")
                stderr = result.get("stderr", "")
                if stdout:
                    self._log("STDOUT:")
                    self._log(stdout)
                if stderr:
                    self._log("STDERR:")
                    self._log(stderr)

        # * Evaluate test coverage after pytest run
        try:
            cov_res = self._evaluate_coverage()
            if cov_res is not None:
                self.results["coverage"] = cov_res
                if not self.json_output:
                    status = "✓" if cov_res.get("exit_code", 0) == 0 else "✗"
                    pct = cov_res.get("percent")
                    pct_str = f"{pct:.1f}%" if isinstance(pct, (int, float)) else "n/a"
                    self._print(f"-- coverage --\n{status} coverage={pct_str}")
                # Log details
                self._log("\n--- COVERAGE ---")
                self._log(f"Exit code: {cov_res.get('exit_code', 'unknown')}")
                self._log(f"Percent: {cov_res.get('percent', 'n/a')}")
                if cov_res.get("stdout"):
                    self._log("STDOUT:")
                    self._log(str(cov_res.get("stdout")))
                if cov_res.get("stderr"):
                    self._log("STDERR:")
                    self._log(str(cov_res.get("stderr")))
        except Exception as _exc:  # noqa: BLE001
            # Coverage evaluation errors are non-fatal
            pass

        self.results["summary"] = self._summary()
        if not self.json_output:
            self._print_summary()
        return self.results

    def _summary(self) -> Dict[str, Any]:
        # * Compute counts for critical failures and non-critical issues
        crit = 0
        noncrit_issues = 0
        for name, r in self.results.items():
            if not isinstance(r, dict):
                continue
            if not r.get("available"):
                continue
            if name == "summary":
                continue
            failed = r.get("exit_code", 0) != 0
            if failed and r.get("critical"):
                crit += 1
            elif failed and not r.get("critical"):
                noncrit_issues += 1
        return {
            "critical_failures": crit,
            "issues": noncrit_issues,
            "overall_status": "PASS" if crit == 0 else "FAIL",
            "execution_time": round(time.time() - self.start_time, 2),
        }

    def _count_nonempty_lines(self, text: str) -> int:
        # * Counts non-empty lines in given text
        return sum(1 for ln in (text or "").splitlines() if ln.strip())

    def _brief_failure_line(self, name: str, res: Dict[str, Any]) -> str:
        # * Builds a concise, human-friendly failure/issue summary line per tool
        out = (res.get("stdout") or "")
        err = (res.get("stderr") or "")
        text = (out + "\n" + err).strip()
        # * pyright: prefer JSON summary if parsed
        if name == "pyright":
            s = res.get("summary") or {}
            if isinstance(s, dict) and s:
                ec = s.get("errorCount")
                wc = s.get("warningCount")
                fc = s.get("filesAnalyzed")
                parts = []
                if isinstance(ec, int):
                    parts.append(f"errors={ec}")
                if isinstance(wc, int):
                    parts.append(f"warnings={wc}")
                if isinstance(fc, int):
                    parts.append(f"files={fc}")
                joined = ", ".join(parts) if parts else "failed"
                return f"pyright: {joined}"
            # Fallback to line count
            return f"pyright: {self._count_nonempty_lines(text)} diagnostics"
        # * mypy: extract 'Found X error(s)' if present
        if name == "mypy":
            m = re.search(r"Found\s+(\d+)\s+error", text)
            if m:
                return f"mypy: errors={m.group(1)}"
            return f"mypy: {self._count_nonempty_lines(text)} problems"
        # * ruff: concise format -> count lines
        if name == "ruff":
            return f"ruff: {self._count_nonempty_lines(out)} violations"
        # * compile: generic message
        if name == "compile":
            return "compile: syntax errors detected"
        # * pytest: try to extract summary line
        if name == "pytest":
            tail = "\n".join([ln for ln in text.splitlines() if ln.strip()][-3:])
            m = re.search(r"(\d+)\s+failed.*", tail)
            if m:
                return f"pytest: failed={m.group(1)}"
            return "pytest: failures detected"
        # * bandit: use computed severity summary if available
        if name == "bandit":
            sev = res.get("bandit_summary") or {}
            if sev:
                low = int(sev.get("LOW", 0))
                med = int(sev.get("MEDIUM", 0))
                high = int(sev.get("HIGH", 0))
                return f"bandit: HIGH={high}, MEDIUM={med}, LOW={low}"
            return f"bandit: {self._count_nonempty_lines(text)} findings"
        # * pip-audit: try parse JSON list length
        if name == "pip-audit":
            try:
                data = json.loads(out or err or "[]")
                if isinstance(data, list):
                    return f"pip-audit: vulnerabilities={len(data)}"
            except json.JSONDecodeError:
                pass
            return f"pip-audit: {self._count_nonempty_lines(text)} findings"
        # * coverage: show percent and gate
        if name == "coverage":
            pct = res.get("percent")
            try:
                pctf = float(pct) if pct is not None else None
            except Exception:
                pctf = None
            if pctf is not None:
                if pctf < self.coverage_fail_threshold:
                    return f"coverage: {pctf:.1f}% < fail {self.coverage_fail_threshold:.0f}%"
                return f"coverage: {pctf:.1f}%"
            return "coverage: failed to compute"
        # * Default: show exit code and line count
        return f"{name}: exit={res.get('exit_code')}, lines={self._count_nonempty_lines(text)}"

    def _brief_warning_line(self, name: str, res: Dict[str, Any]) -> Optional[str]:
        # * Builds a concise warning-only line for tools that report warnings with exit code 0
        if name == "pyright":
            s = res.get("summary") or {}
            if isinstance(s, dict):
                wc = s.get("warningCount")
                ec = s.get("errorCount")
                fc = s.get("filesAnalyzed")
                if isinstance(wc, int) and wc > 0 and (not isinstance(ec, int) or ec == 0):
                    parts = [f"warnings={wc}"]
                    if isinstance(fc, int):
                        parts.append(f"files={fc}")
                    return f"pyright: {', '.join(parts)}"
        if name == "bandit":
            sev = res.get("bandit_summary") or {}
            if isinstance(sev, dict) and sev:
                high = int(sev.get("HIGH", 0))
                med = int(sev.get("MEDIUM", 0))
                low = int(sev.get("LOW", 0))
                if high == 0 and (med > 0 or low > 0):
                    return f"bandit: MEDIUM={med}, LOW={low}"
        if name == "coverage":
            pct = res.get("percent")
            try:
                pctf = float(pct) if pct is not None else None
            except Exception:
                pctf = None
            if pctf is not None and self.coverage_fail_threshold <= pctf < self.coverage_warn_threshold:
                return f"coverage: {pctf:.1f}% < warn {self.coverage_warn_threshold:.0f}%"
        return None

    def _limit(self, items: List[str], n: int = 20) -> List[str]:
        if len(items) <= n:
            return items
        else:
            limited = items[:n]
            remaining = len(items) - n
            limited.append(f"- and {remaining} more...")
            return limited

    def _pyright_messages(self, res: Dict[str, Any], severity: str) -> List[str]:
        out: List[str] = []
        diags = res.get("diagnostics") or []
        for d in diags:
            if str(d.get("severity", "")).lower() != severity.lower():
                continue
            file = d.get("file", "")
            rng = d.get("range") or {}
            start = rng.get("start") or {}
            line = start.get("line")
            col = start.get("character")
            msg = d.get("message", "")
            if isinstance(line, int) and isinstance(col, int):
                out.append(f"{file}:{line+1}:{col+1}: {msg}")
            else:
                out.append(f"{file}: {msg}")
        return out

    def _bandit_messages(self, res: Dict[str, Any], severities: List[str]) -> List[str]:
        results = res.get("bandit_results") or []
        out: List[str] = []
        sevset = {s.upper() for s in severities}
        for it in results:
            sev = str(it.get("issue_severity", "")).upper()
            if sev not in sevset:
                continue
            fn = it.get("filename", "")
            ln = it.get("line_number")
            txt = (it.get("issue_text", "") or "").strip()
            tid = it.get("test_id", "")
            if isinstance(ln, int) and ln > 0:
                out.append(f"{fn}:{ln}: [{sev}] {txt} ({tid})")
            else:
                out.append(f"{fn}: [{sev}] {txt} ({tid})")
        return out

    def _first_nonempty_lines(self, text: str, n: int = 20) -> List[str]:
        lines = [ln for ln in (text or "").splitlines() if ln.strip()]
        if len(lines) <= n:
            return lines
        else:
            limited = lines[:n]
            remaining = len(lines) - n
            limited.append(f"and {remaining} more...")
            return limited

    def _print_summary(self) -> None:
        s = self.results["summary"]
        self._print("==================== SUMMARY ====================")
        self._print(f"Critical failures: {s['critical_failures']}")
        self._print(f"Issues: {s.get('issues', 0)}")
        self._print(f"📋 Full log: build.log - use grep_tool to find specific issues")
        # * Print concise failure/issue lines for quick diagnostics
        brief_lines: List[str] = []
        warning_lines: List[str] = []
        detail_errors: Dict[str, List[str]] = {}
        detail_warnings: Dict[str, List[str]] = {}
        for name, r in self.results.items():
            if name == "summary" or not isinstance(r, dict):
                continue
            if not r.get("available"):
                continue
            exit_code = r.get("exit_code", 0)
            if exit_code != 0:
                brief_lines.append(self._brief_failure_line(name, r))
                # * Collect brief error details by tool
                if name == "pyright":
                    msgs = self._limit(self._pyright_messages(r, "error"))
                    if msgs:
                        detail_errors[name] = msgs
                elif name == "bandit":
                    sev = r.get("bandit_summary") or {}
                    if int(sev.get("HIGH", 0)) > 0:
                        msgs = self._limit(self._bandit_messages(r, ["HIGH"]))
                        if msgs:
                            detail_errors[name] = msgs
                elif name == "pytest":
                    text = ((r.get("stdout") or "") + "\n" + (r.get("stderr") or "")).strip()
                    lines = [ln for ln in text.splitlines() if "FAILED" in ln or "AssertionError" in ln]
                    if not lines:
                        lines = self._first_nonempty_lines(text)
                    if lines:
                        detail_errors[name] = lines
                else:
                    text = ((r.get("stdout") or "") + "\n" + (r.get("stderr") or "")).strip()
                    lines = self._first_nonempty_lines(text)
                    if lines:
                        detail_errors[name] = lines
            else:
                wl = self._brief_warning_line(name, r)
                if wl:
                    warning_lines.append(wl)
                    # * Collect brief warning details
                    if name == "pyright":
                        msgs = self._limit(self._pyright_messages(r, "warning"))
                        if msgs:
                            detail_warnings[name] = msgs
                    elif name == "bandit":
                        msgs = self._limit(self._bandit_messages(r, ["MEDIUM", "LOW"]))
                        if msgs:
                            detail_warnings[name] = msgs
        if brief_lines:
            self._print("Errors:")
            for line in brief_lines:
                self._print(f"  - {line}")
            # * Details per tool (limited)
            for tool, lines in detail_errors.items():
                self._print(f"    {tool}:")
                for ln in lines:
                    self._print(f"      - {ln}")
        if warning_lines:
            self._print("Warnings:")
            for line in warning_lines:
                self._print(f"  - {line}")
            for tool, lines in detail_warnings.items():
                self._print(f"    {tool}:")
                for ln in lines:
                    self._print(f"      - {ln}")
        self._print(f"Execution time: {s['execution_time']}s")
        self._print(f"Overall: {s['overall_status']}")

    # * Coverage evaluation helper
    def _evaluate_coverage(self) -> Optional[Dict[str, Any]]:
        """Read coverage.xml and apply gates: FAIL < fail_threshold; WARN < warn_threshold.

        Returns a result dict compatible with other tools, or None if no coverage file is found.
        """
        path = self.coverage_xml_path
        if not os.path.exists(path):
            return None
        start_time = time.time()
        percent: Optional[float] = None
        err: str = ""
        try:
            tree = ET.parse(path)
            root = tree.getroot()
            # coverage.py writes attributes on root: line-rate or totals
            line_rate = root.attrib.get("line-rate")
            if line_rate is not None:
                percent = float(line_rate) * 100.0
            else:
                # Fallback to totals
                lines_valid = root.attrib.get("lines-valid")
                lines_covered = root.attrib.get("lines-covered")
                if lines_valid and lines_covered:
                    lv = float(lines_valid)
                    lc = float(lines_covered)
                    percent = 0.0 if lv == 0 else (lc / lv) * 100.0
        except Exception as exc:  # noqa: BLE001
            err = str(exc)

        if percent is None:
            return {
                "tool": "coverage",
                "available": True,
                "exit_code": 1,
                "stdout": "",
                "stderr": err or "coverage percent not found",
                "critical": False,
                "fixed": False,
                "exec_time": round(time.time() - start_time, 2),
            }

        # Decide gate: critical fail if below fail threshold; warning if below warn threshold
        exit_code = 0
        critical = False
        if percent < self.coverage_fail_threshold:
            exit_code = 1
            critical = True

        return {
            "tool": "coverage",
            "available": True,
            "exit_code": exit_code,
            "stdout": f"coverage={percent:.2f}%",
            "stderr": "",
            "critical": critical,
            "fixed": False,
            "exec_time": round(time.time() - start_time, 2),
            "percent": percent,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Local CI runner for Artemonim's Speech Kit")
    parser.add_argument("--tool", choices=list(TOOLS.keys()), help="Run only a specific tool")
    parser.add_argument("--path", nargs="+", help="Limit analysis to given paths")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--no-fix", action="store_true", help="Disable auto-fix phase")
    parser.add_argument("--skip-launch", action="store_true", help="Run checks/tests only; do not launch app")
    parser.add_argument("--fast-launch", action="store_true", help="Launch the app only; skip checks/tests")
    args = parser.parse_args()

    # * FastLaunch: only launch the application (GUI), skip any checks
    if args.fast_launch:
        try:
            # * Launch GUI application in the foreground
            code = subprocess.call([sys.executable, "-m", "gui.main_window"])  # noqa: S603
            sys.exit(code)
        except KeyboardInterrupt:
            print("Interrupted by user (Ctrl+C)")
            sys.exit(130)
        except FileNotFoundError:
            print("Error: Python interpreter not found.", file=sys.stderr)
            sys.exit(127)
        except Exception as exc:  # noqa: BLE001
            print(f"Error launching application: {exc}", file=sys.stderr)
            sys.exit(1)

    targets = None
    if args.path:
        ok: List[str] = []
        for p in args.path:
            if os.path.exists(p):
                ok.append(p)
            else:
                print(f"Error: path not found: {p}", file=sys.stderr)
                sys.exit(1)
        targets = ok

    ci = LocalCI(verbose=args.verbose, json_output=args.json, targets=targets)
    res = ci.run(only=args.tool, fix=(not args.no_fix))
    if args.json:
        print(json.dumps(res, indent=2))

    # * Launch GUI by default after successful checks (unless explicitly skipped)
    if not args.skip_launch and res["summary"]["overall_status"] == "PASS":
        try:
            # * Launch GUI application in the foreground
            print("Launching GUI application...")
            code = subprocess.call([sys.executable, "-m", "gui.main_window"])  # noqa: S603
            sys.exit(code)
        except KeyboardInterrupt:
            print("Interrupted by user (Ctrl+C)")
            sys.exit(130)
        except FileNotFoundError:
            print("Error: Python interpreter not found.", file=sys.stderr)
            sys.exit(127)
        except Exception as exc:  # noqa: BLE001
            print(f"Error launching application: {exc}", file=sys.stderr)
            sys.exit(1)

    sys.exit(0 if res["summary"]["overall_status"] == "PASS" else 1)


if __name__ == "__main__":
    main()


