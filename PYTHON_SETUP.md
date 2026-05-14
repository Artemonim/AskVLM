# Python 3.11 setup

AskVLM is developed and tested on **Python 3.11.x** only. Package metadata (`requires-python`), the PowerShell `build.ps1` / `run.ps1` flow, and optional ML wheels are aligned with that version.

## Check your version

```bash
python --version
```

You should see `Python 3.11.x`. If not, install Python 3.11 or point your environment at it (see below).

## Windows (recommended)

1. Install **Python 3.11** from [python.org](https://www.python.org/downloads/) or the Microsoft Store, and enable the **Python Launcher** (`py.exe`) when the installer offers it.
2. Create the project venv with the launcher:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   Alternatively, run `pwsh -File build.ps1` (or `run.ps1`) from the repo root: it prefers `py -3.11` when creating `.venv`.

## Why not 3.12 or newer?

The repository intentionally pins **3.11** for reproducible installs and CI. Broader version support would require relaxing `requires-python`, updating automation, and validating the full ML stack on additional interpreters.
