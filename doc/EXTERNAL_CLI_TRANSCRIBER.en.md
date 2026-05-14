# External CLI Transcriber

## Purpose

`python cli.py external-transcribe ...` — a stable CLI entry point "one call — one file" for local applications that need to pass AskVLM a single media file and receive the transcript text.

The flow is designed for machine interaction:

- Default Whisper model is `small`.
- By default the command writes only the transcript text to `stdout`.
- Speaker diarization and LLM dialog formatting are disabled by default.
- Whisper is loaded on demand and unloaded after the command finishes.
- If CUDA is available but VRAM allocation for Whisper fails, AskVLM automatically retries Whisper on CPU.
- On Windows with `--device` other than `cpu`, the command runs in an isolated child process to reduce the impact of upstream crash-on-exit bugs (`faster-whisper`/`ctranslate2`) on the calling process.

## Installation

First install the project and ML dependencies:

```powershell
pip install -e .
pip install -e .[ml]
```

If using a virtual environment on Windows:

```powershell
. .\.venv\Scripts\Activate.ps1
```

## Basic Usage

Return transcript text to `stdout`:

```powershell
python cli.py external-transcribe "C:\media\call.wav"
```

Write transcript to a file while also outputting to `stdout`:

```powershell
python cli.py external-transcribe "C:\media\call.wav" `
  --output-file "C:\media\call.txt"
```

Write transcript only to a file:

```powershell
python cli.py external-transcribe "C:\media\call.wav" `
  --output-file "C:\media\call.txt" `
  --no-stdout
```

Force CPU:

```powershell
python cli.py external-transcribe "C:\media\call.wav" --device cpu
```

Explicitly specify language:

```powershell
python cli.py external-transcribe "C:\media\call.wav" --language ru
```

## Command Contract

Command:

```text
python cli.py external-transcribe INPUT_PATH [options]
```

Input:

- `INPUT_PATH` — a single audio or video file.

Output:

- By default the command prints the final transcript text to `stdout`.
- If `--output-file` is specified, the same plain text is additionally written to that file.
- With `--no-stdout`, you must also pass `--output-file`.

Exit codes:

- `0` — transcription completed successfully, including the case of an empty or whitespace-only result after `str.strip()`.
- `1` — execution error with no valid successful result.
- Any other non-zero code — AskVLM failed to process the file or execution-stage failure.

Empty transcript: if `get_full_text()` returns a string that is empty after `strip()`, this is treated as a normal successful completion. The command:

- does not write an empty line to `stdout` (i.e. does not add an extra newline),
- creates `--output-file` as an empty file if the path was provided,
- does not print a diagnostic message to `stderr`.

## Default Behavior

The external CLI uses these defaults unless overridden:

- `--whisper-model small`
- `--device auto`
- `--compute-type auto`
- `--no-diarization`
- `--no-dialog-blocks`
- `--stdout`

Behavior of `--device auto`:

1. AskVLM first tries CUDA if it is available.
2. If model loading or GPU inference fails due to insufficient GPU memory, AskVLM unloads Whisper and retries on CPU.
3. The CPU fallback automatically picks a safe compute type for CPU.

This way an overloaded GPU does not block local integration as long as there is enough RAM in the system.

The fallback guarantee applies to the standard single-pass Whisper path. Options like diarization may require additional GPU capacity.

## Reliability on Windows (subprocess isolation)

For `external-transcribe` on Windows with `--device != cpu`, AskVLM runs transcription in a child process:

1. The parent launches the same CLI entrypoint with internal hidden child-mode flags.
2. The child writes a service JSON result before `pipeline.close(...)`.
3. The parent treats this JSON as the source of truth:
   - if the JSON is valid and contains `status=ok`, this is success even with a crash-like child exit code after the result was written;
   - if no valid successful JSON is found and the exit code looks like a native crash, the parent makes exactly one CPU retry;
   - if no valid successful JSON is found and the error is not crash-like, the parent surfaces the error without a CPU retry.

## JIT Loading and Unloading

The command is designed for single-shot use:

1. AskVLM parses CLI arguments.
2. AskVLM prepares audio in a temporary working directory unless `--work-dir` is specified.
3. Whisper is loaded into memory only when transcription begins.
4. AskVLM returns the transcript.
5. Whisper is unloaded before the process exits.

If `--work-dir` is not specified, AskVLM creates a temporary directory and removes it after completion.

## Recommended Integration Pattern

### PowerShell

```powershell
$transcript = python cli.py external-transcribe "C:\media\call.wav"
if ($LASTEXITCODE -ne 0) {
    throw "AskVLM transcription error."
}
Write-Host $transcript
```

### Python Subprocess

```python
import subprocess

result = subprocess.run(
    ["python", "cli.py", "external-transcribe", r"C:\media\call.wav"],
    capture_output=True,
    text=True,
    check=False,
)

if result.returncode != 0:
    raise RuntimeError(result.stderr or "AskVLM transcription error")

transcript_text = result.stdout
```

## Additional Options

Enable speaker diarization:

```powershell
python cli.py external-transcribe "C:\media\meeting.mp3" --diarization
```

Enable LLM text cleanup:

```powershell
python cli.py external-transcribe "C:\media\draft.wav" --dialog-blocks
```

These options are disabled by default: they increase startup cost and may load additional ML backends.

## Notes

- AskVLM stores model caches in the project's `.cache/` directory.
- `external-transcribe` is designed for one file per process invocation.
- For batch export to `txt`, `srt`, `vtt`, or `json`, use `python cli.py transcribe ...`.
