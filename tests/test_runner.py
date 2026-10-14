"""Regression contracts for test selection, worker limits, and headless Qt."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from _pytest.mark.expression import Expression

from build import TOOLS, LocalCI, parse_test_workers

_ROOT = Path(__file__).resolve().parents[1]

# * The fake interpreter is the external process boundary of the real wrappers.
# * Environment probes succeed without installing packages or importing ML stacks.
_FAKE_ACTIVATE = """
function global:python {
    if ($args.Count -eq 2 -and $args[0] -eq '-c') {
        if ($args[1] -like '*sys.version_info*') { '3.11' }
        $global:LASTEXITCODE = 0
        return
    }
    if ($args.Count -gt 0 -and $args[0] -eq 'build.py') {
        ConvertTo-Json -InputObject @($args) -Compress
        $global:LASTEXITCODE = 0
        return
    }
    throw ('Unexpected Python command: {0}' -f ($args -join ' '))
}
"""

_QT_PROBE = """
import json
import os
import runpy
import sys

sys.path.insert(0, sys.argv[1])
if sys.argv[2] == 'conftest':
    runpy.run_path(sys.argv[1] + '/tests/conftest.py')
from PySide6.QtWidgets import QApplication

app = QApplication([])
print(json.dumps({'platform': app.platformName(),
                  'configured': os.environ.get('QT_QPA_PLATFORM')}))
"""

_COLLECTION_PROBE = """
import json
import sys

import pytest

class Selection:
    def pytest_collection_finish(self, session):
        print(json.dumps({'selected': [item.nodeid for item in session.items],
                          'workers': session.config.option.numprocesses}))

raise SystemExit(pytest.main(sys.argv[1:], plugins=[Selection()]))
"""


@pytest.fixture
def wrapper_workspace(tmp_path: Path) -> Path:
    """Isolate the real wrappers from dependency setup and application launch."""
    for name in ("run.ps1", "build.ps1"):
        shutil.copy2(_ROOT / name, tmp_path / name)
    activate = tmp_path / ".venv" / "Scripts" / "Activate.ps1"
    activate.parent.mkdir(parents=True)
    activate.write_text(_FAKE_ACTIVATE, encoding="utf-8")
    return tmp_path


def _powershell(script: Path, arguments: list[str]) -> subprocess.CompletedProcess[str]:
    executable = shutil.which("pwsh")
    if executable is None:
        pytest.skip("PowerShell 7 is required for runner wrapper tests")
    return subprocess.run(  # noqa: S603 - trusted interpreter, fixed script and argv, no shell
        [executable, "-NoLogo", "-NoProfile", "-File", str(script), *arguments],
        cwd=script.parent,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=30,
    )


def test_default_worker_limit_stays_bounded() -> None:
    """Catch a default that can exhaust system commit by using every CPU."""
    with (_ROOT / "pyproject.toml").open("rb") as config_file:
        options = tomllib.load(config_file)["tool"]["pytest"]["ini_options"]["addopts"]
    worker_options = [
        options[index + 1] for index, value in enumerate(options) if value == "-n"
    ]
    assert worker_options == ["3"]


@pytest.mark.parametrize(
    ("fast", "cli_only", "expected"),
    [
        (False, False, [True, True, True, True, True]),
        (True, False, [True, True, False, False, False]),
        (False, True, [True, False, True, True, False]),
        (True, True, [True, False, False, False, False]),
    ],
)
def test_pytest_filters_preserve_both_selection_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    fast: bool,
    cli_only: bool,
    expected: list[bool],
) -> None:
    """Catch CLI selection overriding Fast and accidentally admitting real models."""
    monkeypatch.chdir(tmp_path)
    ci = LocalCI(
        verbose=False,
        json_output=False,
        targets=["tests"],
        fast=fast,
        cli_only=cli_only,
    )
    command = ci._build_command("pytest", fix=False)  # noqa: SLF001 - command builder contract
    # * The interpreter's -m selects pytest; marker options follow the module name.
    arguments = command[command.index("pytest") + 1 :]
    assert arguments.count("-m") == int(fast or cli_only)
    marker_sets = (set(), {"gui"}, {"slow"}, {"heavy_ml"}, {"slow", "gui"})
    if "-m" in arguments:
        expression = Expression.compile(arguments[arguments.index("-m") + 1])
        selected = [
            expression.evaluate(markers.__contains__) for markers in marker_sets
        ]
    else:
        selected = [True for _ in marker_sets]
    assert selected == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, None), ("0", "0"), ("002", "2"), ("auto", "auto"), (" AUTO ", "auto")],
)
def test_pytest_worker_override_includes_in_process_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw: str | None,
    expected: str | None,
) -> None:
    """Catch an override dropped for zero or corrupted before reaching xdist."""
    monkeypatch.chdir(tmp_path)
    workers = None if raw is None else parse_test_workers(raw)
    ci = LocalCI(
        verbose=False, json_output=False, targets=["tests"], test_workers=workers
    )
    command = ci._build_command("pytest", fix=False)  # noqa: SLF001 - command builder contract
    if expected is None:
        assert "-n" not in command
    else:
        assert command.count("-n") == 1
        assert command[command.index("-n") + 1] == expected


@pytest.mark.parametrize("tool", [name for name in TOOLS if name != "pytest"])
@pytest.mark.parametrize("fix", [False, True])
def test_pytest_flags_do_not_change_other_tool_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tool: str, *, fix: bool
) -> None:
    """Catch pytest-only options leaking into lint, type, or dependency commands."""
    monkeypatch.chdir(tmp_path)
    baseline = LocalCI(verbose=False, json_output=False, targets=["tests"])
    selected = LocalCI(
        verbose=False,
        json_output=False,
        targets=["tests"],
        fast=True,
        cli_only=True,
        test_workers="2",
    )
    assert selected._build_command(tool, fix) == baseline._build_command(tool, fix)  # noqa: SLF001 - command builder contract


@pytest.mark.integration
@pytest.mark.parametrize(
    "value",
    ["-1", "+3", "1_0", "2.5", "", "٢", "9" * 5000],
    ids=["negative", "plus", "underscore", "decimal", "empty", "unicode", "huge"],
)
def test_python_entry_rejects_invalid_workers_before_setup(
    tmp_path: Path, value: str
) -> None:
    """Catch invalid counts reaching setup instead of producing argparse exit 2."""
    # * Keep the huge-value case deterministic when the parent disables digit limits.
    environment = os.environ.copy()
    environment["PYTHONINTMAXSTRDIGITS"] = "4300"
    result = subprocess.run(  # noqa: S603 - trusted interpreter and fixed argv, no shell
        [
            sys.executable,
            "-B",
            str(_ROOT / "build.py"),
            "--skip-launch",
            "--fast",
            f"--test-workers={value}",
            # * Help prevents setup even if validation regresses and accepts the input.
            "--help",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=30,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "--test-workers" in result.stderr


@pytest.mark.integration
@pytest.mark.parametrize("script_name", ["run.ps1", "build.ps1"])
@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ([], []),
        (["-Cli", "-TestWorkers", "2"], ["--cli", "--test-workers", "2"]),
        (
            ["-Cli", "-Fast", "-TestWorkers", "0"],
            ["--cli", "--fast", "--test-workers", "0"],
        ),
        (["-TestWorkers", " auto "], ["--test-workers", "auto"]),
    ],
)
def test_powershell_entries_forward_test_options(
    wrapper_workspace: Path,
    script_name: str,
    arguments: list[str],
    expected: list[str],
) -> None:
    """Catch test options lost between a PowerShell entry and the Python process."""
    result = _powershell(
        wrapper_workspace / script_name,
        ["-SkipLaunch", "-SkipEnsureML", "-SkipEnsureCUDA", *arguments],
    )
    assert result.returncode == 0, result.stdout + result.stderr
    command = json.loads(result.stdout.splitlines()[-1])
    assert command[0] == "build.py"
    assert "--skip-launch" in command
    actual = [argument for argument in command[1:] if argument != "--skip-launch"]
    # * Argument order between independent switches is immaterial.
    assert sorted(actual) == sorted(expected)
    if "--test-workers" in command:
        assert command[command.index("--test-workers") + 1] == expected[-1]


@pytest.mark.integration
@pytest.mark.parametrize("script_name", ["run.ps1", "build.ps1"])
@pytest.mark.parametrize("value", ["-1", "+3", "1_0", "2.5", "", "٢"])
def test_powershell_entries_reject_invalid_workers(
    wrapper_workspace: Path, script_name: str, value: str
) -> None:
    """Catch wrapper validation forwarding malformed counts into environment setup."""
    result = _powershell(
        wrapper_workspace / script_name,
        ["-SkipLaunch", "-Fast", "-TestWorkers", value],
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "TestWorkers" in result.stderr


@pytest.mark.integration
@pytest.mark.parametrize("fast", [False, True])
def test_cli_collection_preserves_core_and_excludes_real_gui_tests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, fast: bool
) -> None:
    """Catch missing GUI markers or CLI filters deselecting mixed-module core tests."""
    monkeypatch.chdir(tmp_path)
    ci = LocalCI(
        verbose=False,
        json_output=False,
        targets=[str(_ROOT / "tests" / "test_metadata_and_icons.py")],
        fast=fast,
        cli_only=True,
        test_workers="2",
    )
    command = ci._build_command("pytest", fix=False)  # noqa: SLF001 - command builder contract
    result = subprocess.run(  # noqa: S603 - collect-only, fixed test file and argv, no shell
        [
            sys.executable,
            "-B",
            "-c",
            _COLLECTION_PROBE,
            *command[command.index("pytest") + 1 :],
            "-p",
            "no:cacheprovider",
            "-o",
            "addopts=",
            "--no-cov",
            "--collect-only",
        ],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    selection = next(
        json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")
    )
    assert selection["workers"] == 2
    selected = set(selection["selected"])
    assert "tests/test_metadata_and_icons.py::test_srt_metadata_round_trip" in selected
    gui_cases = {
        "tests/test_metadata_and_icons.py::test_input_status_scan_sets_icons",
        "tests/test_metadata_and_icons.py::test_pipeline_appends_ask_metadata_and_scan",
    }
    assert selected.isdisjoint(gui_cases)


def _qt_platform(
    environment: dict[str, str], *, load_conftest: bool
) -> dict[str, str | None]:
    result = subprocess.run(  # noqa: S603 - fixed QApplication probe, no shell or window.show
        [
            sys.executable,
            "-B",
            "-c",
            _QT_PROBE,
            str(_ROOT),
            "conftest" if load_conftest else "native",
        ],
        cwd=_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout.splitlines()[-1])


@pytest.mark.gui
@pytest.mark.integration
@pytest.mark.parametrize(
    ("show_gui", "preset", "expected"),
    [(None, None, "offscreen"), ("0", None, "offscreen"), (None, "minimal", "minimal")],
)
def test_conftest_configures_actual_qt_platform(
    show_gui: str | None, preset: str | None, expected: str
) -> None:
    """Catch native test windows or an explicit Qt platform silently overwritten."""
    environment = os.environ.copy()
    environment.pop("QT_QPA_PLATFORM", None)
    environment.pop("ASKVLM_TESTS_SHOW_GUI", None)
    if show_gui is not None:
        environment["ASKVLM_TESTS_SHOW_GUI"] = show_gui
    if preset is not None:
        environment["QT_QPA_PLATFORM"] = preset
    result = _qt_platform(environment, load_conftest=True)
    assert result["platform"] == expected


@pytest.mark.gui
@pytest.mark.integration
@pytest.mark.skipif(
    sys.platform != "win32", reason="Native Windows Qt platform contract"
)
def test_conftest_gui_debug_opt_out_preserves_native_platform() -> None:
    """Catch the debug opt-out leaving QApplication stuck on the headless plugin."""
    environment = os.environ.copy()
    environment.pop("QT_QPA_PLATFORM", None)
    environment.pop("ASKVLM_TESTS_SHOW_GUI", None)
    native = _qt_platform(environment, load_conftest=False)
    environment["ASKVLM_TESTS_SHOW_GUI"] = "1"
    debug = _qt_platform(environment, load_conftest=True)
    assert debug["configured"] is None
    assert debug["platform"] == native["platform"]
