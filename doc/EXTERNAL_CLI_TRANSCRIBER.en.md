# External CLI Transcriber

## Purpose

`python cli.py external-transcribe ...` — a stable CLI entry point "one call — one file" for local applications that need to pass AskVLM a single media file and receive the transcript text.

The flow is designed for machine interaction:

- Default Whisper model is `small`.
- Default STT provider is `whisper` (`--stt-provider whisper`). Optional GigaAM providers:
  - `gigaam-ctc` — `ai-sage/GigaAM-Multilingual` revision `ctc`, no punctuation or capitalization; CPU only;
  - `gigaam-e2e-ctc` — `ai-sage/GigaAM-v3` revision `e2e_ctc`, **with** punctuation and capitalization (BPE + ITN); CPU or CUDA.
- By default the command writes only the transcript text to `stdout`.
- Speaker diarization and LLM dialog formatting are disabled by default.
- The model is loaded on demand and unloaded after the command finishes (legacy `--no-daemon`; resident in daemon mode).
- If CUDA is available but VRAM allocation for Whisper or `gigaam-e2e-ctc` fails, AskVLM automatically continues on CPU (details in the "Default Behavior" section).
- `gigaam-ctc` does not use CUDA, CUDA→CPU recovery, or Windows GPU child isolation: CPU only.
- On Windows with `--device` other than `cpu` (Whisper and `gigaam-e2e-ctc`), the command runs in an isolated child process to reduce the impact of upstream crash-on-exit bugs (`faster-whisper`/`ctranslate2`, the torch CUDA runtime) on the calling process (legacy `--no-daemon` mode only).

## Daemon orchestrator and file queue (default mode)

Starting with this version, `external-transcribe` defaults to acting as a **thin client** of a single daemon orchestrator rather than loading the model in its own process:

1. The client checks whether the daemon is alive (via heartbeat). If not, it starts one **detached** `external-transcribe-daemon` process and waits for readiness.
2. The client enqueues a job in the file queue (default `<project>/.cache/external_queue`), waits for the result up to `--client-timeout`, prints the transcript, and exits.
3. If `--client-timeout` expires, the client **signals the daemon to drop** the job (cancel marker) so the daemon does not waste resources on an abandoned task and the queue does not accumulate stale entries.
4. **Bounded CPU recovery.** If the request targeted GPU (`--device` ≠ `cpu`) for a CUDA-capable provider (Whisper or `gigaam-e2e-ctc`) and the daemon was degraded (client timeout expired or daemon unavailable), the client performs one built-in (in-process) CPU transcription pass with the same provider before exiting. On success it prints the transcript and exits with code `0`; only if the CPU pass also fails does the command return a degraded exit code (`10` for timeout, `11` for unavailable daemon). This prevents a stuck/unavailable GPU daemon from silently losing the transcript. For explicit `--device cpu` and for `gigaam-ctc`, no extra pass is performed (nothing to recover from). The `gigaam-e2e-ctc` CPU pass loads the model into the client process (~2.9 GB RAM for the duration of the pass); callers with a strict RAM budget should pass an explicit `--device cpu`.

Why this matters:

- **The model loads once** in the resident daemon, not per file — no repeated weight reads from disk and no "cold load per message".
- **One process per machine** serves all invocations (`external-transcribe-daemon` is a singleton via lock + heartbeat). Parallelism is capped by `--workers` (default `1`, per the "one active neural network" doctrine).
- **No orphaned workers**: heavy work lives in the daemon; the client is short-lived, and its crash does not leave background GPU processes behind.
- The daemon shuts down after idle time (`--idle-shutdown`, default 600 s), freeing VRAM, and wakes on the next job.

Start the daemon manually (optional — the client starts it automatically):

```powershell
python cli.py external-transcribe-daemon --workers 1 --whisper-model small --device cuda
```

GigaAM (deps already in `.[ml]`; `gigaam-ctc` — CPU, `gigaam-e2e-ctc` — CPU or CUDA):

```powershell
python cli.py external-transcribe-daemon --stt-provider gigaam-ctc --device cpu
python cli.py external-transcribe "C:\media\call.wav" --stt-provider gigaam-ctc
python cli.py external-transcribe-daemon --stt-provider gigaam-e2e-ctc --device cpu
python cli.py external-transcribe "C:\media\call.wav" --stt-provider gigaam-e2e-ctc --device cpu
python cli.py external-transcribe-daemon --stt-provider gigaam-e2e-ctc --device cuda
python cli.py external-transcribe "C:\media\call.wav" --stt-provider gigaam-e2e-ctc --device cuda
```

If a daemon is already live with a different `--stt-provider`, the client will **not** submit the job to the wrong resident model: it returns unavailable / mismatch. The singleton is preserved — restart the daemon with the desired provider. For a GPU request of a CUDA-capable provider (Whisper, `gigaam-e2e-ctc`), such an unavailable result triggers the bounded CPU recovery (step 4), and the built-in CPU pass produces the transcript; with `--device cpu` and for `gigaam-ctc`, the command exits with code `11`.

The live daemon's device is compared with the request for diagnostics only: if the daemon heartbeat and the client's `--device` differ by class (`cpu` versus `auto`/`cuda`), the client writes a warning with the `ASKVLM_DAEMON_DEVICE_MISMATCH` marker to `stderr`, and the resident daemon still serves the job on its own device. To change the device, restart the daemon with the desired `--device`. The heartbeat carries the token requested when the daemon started (`auto`/`cpu`/`cuda`), not the actual device after selection and fallback.

Legacy one-shot run without the daemon (old "model in this same process" behavior):

```powershell
python cli.py external-transcribe "C:\media\call.wav" --no-daemon
```

## Installation

First install the project and ML dependencies (Whisper + both GigaAM providers share one extra):

```powershell
pip install -e .
pip install -e .[ml]
# * run.ps1/build.ps1 repair CUDA torch 2.10 by default (bare pip often leaves a CPU wheel)
.\run.ps1 -SkipLaunch -Fast
```

The `[ml]` stack includes torch/torchaudio **2.10** (CUDA wheels repaired by `run.ps1`/`build.ps1` by default: cu128 → cu126), transformers 5, hydra-core, omegaconf, **sentencepiece**, and **pyannote.audio**. The last two are required by remote-code modeling even for short-form `.transcribe` (no longform/VAD). On CPU, `gigaam-ctc` uses roughly ~2.5 GB RAM and `gigaam-e2e-ctc` ~2.9 GB RAM (vs compact Whisper Small CPU); on CPU neither uses VRAM. `gigaam-e2e-ctc` on CUDA uses about 1.5–1.65 GB VRAM per process (CUDA context, fp32 weights ~0.85 GB, 20 s window activations under fp16 autocast; nvidia-smi measurement on an RTX 3070, torch 2.10 + cu128).

Model load uses Hugging Face Transformers with `trust_remote_code=True` **only** for two pinned repos: `ai-sage/GigaAM-Multilingual` at revision `ctc` and `ai-sage/GigaAM-v3` at revision `e2e_ctc` (the cards' official remote-code API). That trust is scoped to these models/revisions, not to arbitrary HF repos. For v3 the wrapper patches `hydra.utils.instantiate` (CPU device) and `PreTrainedModel._finalize_model_loading` (transformers 5.x compat) around `from_pretrained` and **always** restores the originals in `finally`, including when load raises. The v3 weights are always loaded on CPU and cast to fp32; for `--device auto|cuda` the finished model is then moved to CUDA as a whole, after the patches are removed. The model is not converted to fp16 as a whole: the mel frontend (n_fft=320) stays fp32, and on CUDA the encoder runs under the remote code's own `torch.autocast(float16)`. If remote code raises `ImportError`/`ModuleNotFoundError` because `[ml]` is incomplete, the wrapper surfaces `pip install -e ".[ml]"` and keeps the missing module name.

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

Select GigaAM:

```powershell
python cli.py external-transcribe "C:\media\call.wav" --stt-provider gigaam-ctc --device cpu
python cli.py external-transcribe "C:\media\call.wav" --stt-provider gigaam-e2e-ctc --device cpu
python cli.py external-transcribe "C:\media\call.wav" --stt-provider gigaam-e2e-ctc --device cuda
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

For `--stt-provider gigaam-ctc` the device is always CPU (`auto` → `cpu`); `cuda` is rejected before model load.

For `--stt-provider gigaam-e2e-ctc` the `--device` token (`auto`/`cpu`/`cuda`) is validated without importing torch and resolved at model load:

1. `cpu` — the model stays on CPU, CUDA is not touched.
2. `cuda` — the model is moved to CUDA if CUDA is available; otherwise it runs on CPU with a warning in the log (the command does not fail).
3. `auto` — CUDA is chosen only if CUDA is available and `torch.cuda.mem_get_info` reports at least 2.5 GiB of free VRAM (`GIGAAM_AUTO_CUDA_MIN_FREE_VRAM_BYTES` in `core/gigaam_device.py`: the process peak of ~1.6 GiB plus headroom for a GPU neighbor, such as a resident Whisper daemon); otherwise CPU with a warning. The free-VRAM query creates the process's CUDA context; if `auto` then stays on CPU, the context (a few hundred MiB) is held until the process exits.
4. A GPU out-of-memory error while moving the model returns the whole model to CPU (no mixed weight placement) and releases the CUDA cache.
5. A GPU out-of-memory error on a given 20-second window moves the model to CPU, releases the CUDA cache, and retries that same window on CPU; subsequent windows and jobs of the resident daemon stay on CPU until the process restarts.
6. Unloading the model (`close`/`unload`) after CUDA work releases the CUDA cache (`torch.cuda.empty_cache`).

On CUDA the encoder runs under fp16 autocast, so the text may differ slightly from a CPU run (on the 5-minute test file — 2 words out of 453). The resident daemon on CUDA processes a 5-minute file about 3.5× faster than on CPU (RTX 3070 measurement: about 2.9 s versus 10 s per job, including client startup); in one-shot `--no-daemon` mode on short files CUDA is slower than CPU because of CUDA initialization and the child process.

Whisper-only parameters (`--compute-type`, beam/VAD) are not sent to GigaAM.

This way an overloaded GPU does not block local integration as long as there is enough RAM in the system.

The fallback guarantee applies to the standard single-pass Whisper path and to `gigaam-e2e-ctc`. Options like diarization may require additional GPU capacity. `gigaam-ctc` does not participate in CUDA→CPU recovery.

In addition to OOM fallback inside the model path, daemon mode adds an external bounded CPU fallback at the client level: on timeout or GPU daemon unavailability, one in-process CPU pass is performed (see step 4 above). This covers cases where the GPU daemon hung or crashed and can no longer unload or switch to CPU on its own.

## Reliability on Windows (subprocess isolation)

For `external-transcribe` on Windows with `--device != cpu` (Whisper and `gigaam-e2e-ctc`; `gigaam-ctc` always runs in-process on CPU), AskVLM runs transcription in a child process:

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

## Long media (GigaAM CTC)

Both GigaAM providers (`gigaam-ctc` = Multilingual revision `ctc` and `gigaam-e2e-ctc` = v3 revision `e2e_ctc`) reject files strictly longer than 400000 samples (~25.0 s at 16 kHz) with `ValueError("Too long wav file, use 'transcribe_longform' method.")`. AskVLM does not use native `transcribe_longform` (pyannote VAD, needs `HF_TOKEN`). The same 20 s windowing applies to v3.

Instead of a duration gate and a short-form branch, the wrapper **always** streams the WAV in 20-second windows (in samples of the source file):

- the first window shorter than 20 s (EOF) — a single `model.transcribe` call on the original path;
- otherwise each window is written to a temporary `chunk_NNN.wav`, transcribed separately, texts joined with a space; segments carry absolute `start`/`end` (frame offset / sample rate);
- a file of exactly 20.0 s may go through the chunked path (one temp chunk);
- a non-standard WAV header (`wave.Error` / `EOFError`, including WAVE_FORMAT_EXTENSIBLE on Python 3.11) — one ffmpeg re-encode to PCM 16 kHz mono, then the same window loop.

Cancel of a long job is cooperative: the pipeline raises `CancelledError` from `on_segment` between chunks, not inside a single model call.

For `gigaam-e2e-ctc` on CUDA, a window that fails with a VRAM out-of-memory error is retried on CPU, so the stitched result has no skipped windows.

### `--client-timeout` for long files

In daemon mode the job deadline is `submitted_at + client_timeout` (default 300 s). When it expires, the client sets a cancel marker; the daemon stops at a 20-second window boundary.

Callers need to scale `--client-timeout` with media duration. For files longer than a few minutes the default may be insufficient: hours of audio on CPU GigaAM will not fit in 300 s. After timeout, cancel is cooperative between windows, not instantaneous inside `model.transcribe`.

## Notes

- AskVLM stores model caches in the project's `.cache/` directory.
- `external-transcribe` is designed for one file per process invocation.
- `external-extract-frames` is designed for one video file per process invocation; frames persist in `--output-dir` after the process exits.
- For batch export to `txt`, `srt`, `vtt`, or `json`, use `python cli.py transcribe ...`.
