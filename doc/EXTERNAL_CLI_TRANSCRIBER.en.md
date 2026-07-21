# External CLI Transcriber

## Purpose

`python cli.py external-transcribe ...` — a stable CLI entry point "one call — one file" for local applications that need to pass AskVLM a single media file and receive the transcript text.

The flow is designed for machine interaction:

- Default Whisper model is `small`.
- Default STT provider is `whisper` (`--stt-provider whisper`). Optional CPU-only `gigaam-ctc` (GigaAM Multilingual, revision `ctc`) is available.
- By default the command writes only the transcript text to `stdout`.
- Speaker diarization and LLM dialog formatting are disabled by default.
- The model is loaded on demand and unloaded after the command finishes (legacy `--no-daemon`; resident in daemon mode).
- If CUDA is available but VRAM allocation for Whisper fails, AskVLM automatically retries Whisper on CPU.
- GigaAM CTC does not use CUDA, Whisper CUDA fallbacks, or Windows GPU child isolation: CPU only.
- On Windows with `--device` other than `cpu` (Whisper only), the command runs in an isolated child process to reduce the impact of upstream crash-on-exit bugs (`faster-whisper`/`ctranslate2`) on the calling process (legacy `--no-daemon` mode only).

## Daemon orchestrator and file queue (default mode)

Starting with this version, `external-transcribe` defaults to acting as a **thin client** of a single daemon orchestrator rather than loading the model in its own process:

1. The client checks whether the daemon is alive (via heartbeat). If not, it starts one **detached** `external-transcribe-daemon` process and waits for readiness.
2. The client enqueues a job in the file queue (default `<project>/.cache/external_queue`), waits for the result up to `--client-timeout`, prints the transcript, and exits.
3. If `--client-timeout` expires, the client **signals the daemon to drop** the job (cancel marker) so the daemon does not waste resources on an abandoned task and the queue does not accumulate stale entries.
4. **Bounded CPU recovery.** If the request targeted GPU (`--device` ≠ `cpu`) and the daemon was degraded (client timeout expired or daemon unavailable), the client performs one built-in (in-process) CPU transcription pass before exiting. On success it prints the transcript and exits with code `0`; only if the CPU pass also fails does the command return a degraded exit code (`10` for timeout, `11` for unavailable daemon). This prevents a stuck/unavailable GPU daemon from silently losing the transcript. For explicit `--device cpu`, no extra pass is performed (nothing to recover from).

Why this matters:

- **The model loads once** in the resident daemon, not per file — no repeated weight reads from disk and no "cold load per message".
- **One process per machine** serves all invocations (`external-transcribe-daemon` is a singleton via lock + heartbeat). Parallelism is capped by `--workers` (default `1`, per the "one active neural network" doctrine).
- **No orphaned workers**: heavy work lives in the daemon; the client is short-lived, and its crash does not leave background GPU processes behind.
- The daemon shuts down after idle time (`--idle-shutdown`, default 600 s), freeing VRAM, and wakes on the next job.

Start the daemon manually (optional — the client starts it automatically):

```powershell
python cli.py external-transcribe-daemon --workers 1 --whisper-model small --device cuda
```

GigaAM CTC (CPU; deps already in `.[ml]`):

```powershell
python cli.py external-transcribe-daemon --stt-provider gigaam-ctc --device cpu
python cli.py external-transcribe "C:\media\call.wav" --stt-provider gigaam-ctc
```

If a daemon is already live with a different `--stt-provider`, the client will **not** submit the job to the wrong resident model: it returns unavailable / mismatch. The singleton is preserved — restart the daemon with the desired provider.

Legacy one-shot run without the daemon (old "model in this same process" behavior):

```powershell
python cli.py external-transcribe "C:\media\call.wav" --no-daemon
```

## Installation

First install the project and ML dependencies (Whisper + GigaAM CTC share one extra):

```powershell
pip install -e .
pip install -e .[ml]
# * run.ps1/build.ps1 repair CUDA torch 2.10 by default (bare pip often leaves a CPU wheel)
.\run.ps1 -SkipLaunch -Fast
```

The `[ml]` stack includes torch/torchaudio **2.10** (CUDA wheels repaired by `run.ps1`/`build.ps1` by default: cu128 → cu126), transformers 5, hydra-core, omegaconf, **sentencepiece**, and **pyannote.audio**. The last two are required by remote-code modeling even for short-form `.transcribe` (no longform/VAD). On CPU, GigaAM uses roughly ~2.5 GB RAM (vs compact Whisper Small CPU) but no VRAM, with speed comparable to Whisper Small on GPU.

Model load uses Hugging Face Transformers with `trust_remote_code=True` **only** for repo `ai-sage/GigaAM-Multilingual` at revision `ctc` (the card's official remote-code API). That trust is scoped to this model/revision, not to arbitrary HF repos. If remote code raises `ImportError`/`ModuleNotFoundError` because `[ml]` is incomplete, the wrapper surfaces `pip install -e ".[ml]"` and keeps the missing module name.

Do not install `torch` from PyPI on top of a CUDA build without a follow-up `run.ps1`/`build.ps1`: bare `pip install -e .[ml]` often leaves CPU `torch 2.10.0`. Opt out with `-SkipEnsureCUDA`. See `doc/CUDA_SETUP.md`.

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

Select GigaAM CTC (CPU):

```powershell
python cli.py external-transcribe "C:\media\call.wav" --stt-provider gigaam-ctc --device cpu
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

- `--stt-provider whisper`
- `--whisper-model small`
- `--device auto`
- `--compute-type auto`
- `--no-diarization`
- `--no-dialog-blocks`
- `--stdout`

Behavior of `--device auto` for Whisper:

1. AskVLM first tries CUDA if it is available.
2. If model loading or GPU inference fails due to insufficient GPU memory, AskVLM unloads Whisper and retries on CPU.
3. The CPU fallback automatically picks a safe compute type for CPU.

For `--stt-provider gigaam-ctc` the device is always CPU (`auto` → `cpu`); `cuda` is rejected before model load. Whisper-only parameters (`--compute-type`, beam/VAD) are not sent to GigaAM.

This way an overloaded GPU does not block local integration as long as there is enough RAM in the system.

The fallback guarantee applies to the standard single-pass Whisper path. Options like diarization may require additional GPU capacity. GigaAM does not participate in Whisper CUDA→CPU recovery.

In addition to OOM fallback inside the Whisper path, daemon mode adds an external bounded CPU fallback at the client level: on timeout or GPU daemon unavailability, one in-process CPU pass is performed (see step 4 above). This covers cases where the GPU daemon hung or crashed and can no longer unload or switch to CPU on its own.

## Reliability on Windows (subprocess isolation)

For `external-transcribe` on Windows with `--device != cpu`, AskVLM runs transcription in a child process:

1. The parent launches the same CLI entrypoint with internal hidden child-mode flags.
2. The child writes a service JSON result before `pipeline.close(...)`.
3. The parent treats this JSON as the source of truth:
   - if the JSON is valid and contains `status=ok`, this is success even with a crash-like child exit code after the result was written;
   - if no valid successful JSON is found and the exit code looks like a native crash, the parent makes exactly one CPU retry;
   - if no valid successful JSON is found, the error is not crash-like, and stderr does not contain an internal IPC error marker — the parent surfaces the error without a CPU retry;
   - if no valid successful JSON is found and stderr contains an internal IPC error marker (child-process initialization failure), the parent makes exactly one CPU retry as a safety measure.

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

## Video Frame Extraction (`external-extract-frames`)

`python cli.py external-extract-frames ...` — a stable external CLI entry point "one call — one video file" for applications that need to extract frames with adaptive FPS.

### Contract

```text
python cli.py external-extract-frames INPUT_PATH --output-dir PATH [options]
```

Arguments:

- `INPUT_PATH` — a single video file.
- `--output-dir PATH` — directory where extracted frame images are written (created automatically).

Sampling options:

- `--fps FLOAT` (default `0.5`) — target sampling rate in frames per second.
- `--fps-fallback FLOAT` (default `0.2`) — FPS used when frame budget would be exceeded.
- `--frame-budget INT` (default `20`) — hard cap on the number of extracted frames. `0` disables the cap.

Adaptive FPS behavior:

1. Estimates `ceil(duration_s × fps)`.
2. If estimate ≤ `--frame-budget`, uses `--fps`.
3. If estimate > `--frame-budget`, uses `--fps-fallback`.
4. If the fallback estimate still exceeds `--frame-budget`, uses `frame_budget / duration_s` so the budgeted number of frames samples the full duration uniformly.
5. Any extra frame produced by ffmpeg rounding is truncated: the output never exceeds `--frame-budget`.

Output:

- By default prints extracted frame file paths to `stdout`, one per line.
- With `--json` prints a JSON object: `{"frames": [...], "fps_used": N, "duration_s": N}`.
- Exit code `0` on success (including zero-duration video). Exit code `1` on failure.

Colorspace resilience:

- Some containers tag frames with a color matrix that libswscale cannot convert to RGB for the image encoder (ffmpeg fails with `Invalid color space`).
- Frames are extracted as-is first; on such failure, extraction is retried with progressively stronger colorspace-normalizing `-vf` strategies (reset tag to BT.709, force swscale matrices, pixel format normalization).
- Total frame loss (exit code `1`) occurs only if all strategies yield zero frames.

### Examples

```powershell
# Extract frames to C:\media\frames with adaptive FPS
python cli.py external-extract-frames "C:\media\clip.mp4" `
  --output-dir "C:\media\frames"

# Get a JSON manifest
python cli.py external-extract-frames "C:\media\clip.mp4" `
  --output-dir "C:\media\frames" --json

# Force 1 FPS, no frame cap
python cli.py external-extract-frames "C:\media\clip.mp4" `
  --output-dir "C:\media\frames" --fps 1.0 --frame-budget 0
```

### Recommended Integration Pattern (Python)

```python
import json
import subprocess

result = subprocess.run(
    [
        "python", "cli.py", "external-extract-frames",
        r"C:\media\clip.mp4",
        "--output-dir", r"C:\media\frames",
        "--json",
    ],
    capture_output=True,
    text=True,
    check=False,
)

if result.returncode != 0:
    raise RuntimeError(result.stderr or "AskVLM frame extraction error")

manifest = json.loads(result.stdout)
frame_paths = manifest["frames"]
fps_used    = manifest["fps_used"]
```

## Notes

- AskVLM stores model caches in the project's `.cache/` directory.
- `external-transcribe` is designed for one file per process invocation.
- `external-extract-frames` is designed for one video file per process invocation; frames persist in `--output-dir` after the process exits.
- For batch export to `txt`, `srt`, `vtt`, or `json`, use `python cli.py transcribe ...`.
