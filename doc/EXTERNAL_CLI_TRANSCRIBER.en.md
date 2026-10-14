# External CLI Transcriber

## Purpose

`python cli.py external-transcribe ...` — a stable CLI entry point for local applications that pass AskVLM media files and receive transcript text. One input keeps the "one call — one file" contract (plain text on `stdout`); several inputs in one call return one JSON line per input.

The flow is designed for machine interaction:

- Default Whisper model is `small`.
- Default STT provider is `whisper` (`--stt-provider whisper`). Optional GigaAM providers:
  - `gigaam-ctc` — `ai-sage/GigaAM-Multilingual` revision `ctc`, no punctuation or capitalization; CPU only;
  - `gigaam-e2e-ctc` — `ai-sage/GigaAM-v3` revision `e2e_ctc`, **with** punctuation and capitalization (BPE + ITN); CPU or CUDA.
- By default the command writes only the transcript text to `stdout`.
- Speaker diarization and LLM dialog formatting are disabled by default.
- By default the files go through the shared STT supervisor: worker processes keep the models resident, one process per configuration (provider, model, compute type, CPU or GPU). With legacy `--no-daemon` the model is loaded in the command's own process and unloaded before it exits.
- Through the supervisor the device is strict: `--device cpu` never uses the GPU and `--device cuda` stays on the GPU (CPU only when no CUDA device exists, with a warning); `auto` prefers the GPU and runs on CPU only when that is expected to finish sooner. The in-process CUDA→CPU fallbacks described in "Default Behavior" apply to legacy `--no-daemon` only.
- `gigaam-ctc` does not use CUDA, CUDA→CPU recovery, or Windows GPU child isolation: CPU only.
- On Windows with `--device` other than `cpu` (Whisper and `gigaam-e2e-ctc`), the command runs in an isolated child process to reduce the impact of upstream crash-on-exit bugs (`faster-whisper`/`ctranslate2`, the torch CUDA runtime) on the calling process (legacy `--no-daemon` mode only).

## STT supervisor and worker pool (default mode)

By default `external-transcribe` is a **thin client** of one STT supervisor (`external-transcribe-daemon`) that runs a pool of worker processes. The client never loads a model.

### Call flow

1. The client starts one wall-clock deadline (`--client-timeout`, default 300 s) for the whole call, probes each input's duration with `ffprobe` (a failed probe only means "duration unknown"), and writes one job per input into the file queue `<queue root>/v2/` (queue root: `ASKVLM_EXTERNAL_QUEUE_DIR` or `<project>/.cache/external_queue`). All inputs of one call share a call id, `--priority` and `--max-parallel`.
2. If no supervisor heartbeat is fresh, the client starts a **detached** supervisor (with `CREATE_BREAKAWAY_FROM_JOB` when the caller's job object allows it, so the supervisor survives the caller) and keeps re-starting it at most every 10 s while none answers. If no supervisor becomes reachable within 30 s, the unfinished inputs end as `unavailable` (exit `11`); the message names the supervisor log `<queue root>/v2/control/supervisor.out.log`.
3. The client waits for the results until the deadline. When the deadline passes, it writes cancel markers for the unfinished jobs (the supervisor drops or stops them) and reports `timeout` (exit `10`). `--client-timeout inf` waits without a deadline.
4. There is **no in-process CPU recovery** in the client: a GPU job that cannot finish is not re-run on CPU inside the client process.

### Supervisor and workers

- **One supervisor per queue root**, guarded by an OS file lock (`v2/control/supervisor.lock`); a second supervisor exits at once with code `0`. A heartbeat (`v2/control/supervisor.heartbeat.json`) carries the limits, the workers and the queue size.
- **A worker is a process** that keeps one model resident and serves one configuration (a worker key): provider, model, compute type normalized per backend (Whisper: `int8` on CPU, `float16` on GPU for `auto`; GigaAM: `fp32`), and backend `cpu` or `gpu`. Language is per job and never part of the key.
  - CPU worker: one model shared by several concurrent jobs (CPU slots); each slot runs on `threads_per_slot` threads (default 2).
  - GPU worker: one process that batches across jobs — up to 8 GigaAM 20-second windows or up to 4 Whisper clips of at most 30 s with an explicit language (other Whisper jobs run sequentially). GigaAM never runs CUDA inference from several threads.
- **Workers start on demand** for the configurations that are queued and stop after an idle TTL (default 600 s, `--idle-shutdown`). When a new configuration does not fit, idle workers of other configurations are stopped first (least recently used). Busy workers are never stopped to free resources. The supervisor exits shortly after the last worker stops with an empty queue.
- **Every worker lives in its own Windows Job Object** with a memory limit equal to its commit reservation; the whole process tree (including `ffmpeg`) dies with the job, so no orphaned GPU processes remain. A worker that exceeds its reservation fails alone; the next reservation for that configuration grows by 25%.
- **Admission control:** before a worker starts, the supervisor reserves CPU slots and threads, commit (Windows private bytes) and, for GPU workers, VRAM. It starts at most one cold load at a time and checks the system free commit and the physically free VRAM (`nvidia-smi`) against guards. A request that can never fit the configured limits gets `unavailable` with the numbers.
- **Device policy:** `cpu` runs on CPU; `cuda` runs on the GPU and waits for GPU capacity (it falls back to CPU only when no CUDA device exists, with the warning `ASKVLM_CUDA_UNAVAILABLE`); `auto` prefers the GPU and moves a job to CPU only when its estimated time to result on CPU is below 0.8 × the estimate for waiting for the GPU, so a long `auto` queue can fill both GPU and CPU slots. `gigaam-ctc` is CPU-only. CPU and GPU results may differ slightly.
- **Priority and fair share:** `--priority high|normal|low` (default `normal`). A higher class is admitted first; within a class the scheduler takes turns between calls, so a single-file call does not wait behind a whole batch. Jobs that already started are never preempted. A waiting job is promoted one class every `priority_aging_s` (default 300 s), so `low` work is eventually served.
- **Failures:** a worker crash retries its job once (`max_attempts` 2, persisted across supervisor restarts); a configuration whose model fails to load cools down and its jobs end as `unavailable` with the load error; the worker's error text (for example `Too long wav file`) reaches the client's `stderr`.
- Learned timings and memory peaks are kept in `v2/state/estimates.json`; logs are `v2/control/supervisor.log` and `v2/logs/<worker>.log`.
- A still-running old (v1) daemon only scans `<queue root>/incoming` and never takes v2 jobs; it exits after its own idle time.

### Limits and configuration

The supervisor reads its limits from (first match): `external-transcribe-daemon --config PATH`, the environment variable `ASKVLM_STT_POOL_CONFIG`, `<project>/stt_pool.json` (git-ignored, per machine), or conservative built-in defaults (2 CPU slots, 1 GPU worker, 12 GiB commit, 4 GiB VRAM). `stt_pool.example.json` in the project root shows every key with the values for an 8-CPU-slot / 24 GiB commit, 2-GPU-worker / 7 GiB VRAM machine; copy it to `stt_pool.json` and adjust. Unknown keys or invalid values stop the supervisor with an error.

| Key | Meaning |
| --- | --- |
| `cpu.max_slots`, `cpu.threads_per_slot`, `cpu.max_slots_per_worker` | Concurrent CPU jobs, threads per job, slots per CPU worker process |
| `gpu.max_workers`, `gpu.batch_lanes`, `gpu.host_threads`, `gpu.scale_out_backlog_s` | GPU worker processes, batch lanes (GigaAM/Whisper), CPU threads reserved per GPU worker, backlog that justifies another GPU worker |
| `threads.budget` | One CPU-thread budget for CPU slots and the host threads of GPU workers |
| `memory.commit_budget_gib`, `memory.free_commit_guard_gib`, `memory.critical_free_commit_gib` | Commit budget for all workers together, free-commit guard for a new worker, free commit below which idle workers are stopped |
| `vram.budget_gib`, `vram.free_guard_gib` | VRAM budget for all GPU workers, free-VRAM guard for a new GPU worker |
| `lifecycle.*` | Worker idle TTL, supervisor linger, stop and cancel grace periods |
| `scheduling.*` | CPU spill ratio, starvation guard, attempts, GPU/CPU contention factor, priority aging, cooldowns |

Start the supervisor manually (optional — the client starts it automatically):

```powershell
python cli.py external-transcribe-daemon
python cli.py external-transcribe-daemon --config C:\path\stt_pool.json --idle-shutdown 300
python cli.py external-transcribe-daemon --cpu-slots 4 --gpu-workers 1 --commit-budget-gb 16 --vram-budget-gb 6
```

The old daemon flags `--workers`, `--stt-provider`, `--whisper-model`, `--device`, `--compute-type` and `--language` are accepted and ignored (a note with `ASKVLM_DEPRECATED` goes to `stderr`): the pool serves every configuration.

### Batch calls

```powershell
python cli.py external-transcribe a.wav b.mp3 c.ogg --max-parallel 2 --priority low --output-dir C:\out
```

- With several inputs `stdout` carries one JSON object per input, in input order: `{"index", "input", "status", "text", "error"}` (plus `"output"` with `--output-dir`); a line is written as soon as all earlier inputs are finished. `status` is `ok`, `empty`, `error`, `timeout` or `unavailable`.
- `--output-dir DIR` writes `<stem>.txt` for every successful input; repeated stems get `-2`, `-3`, … suffixes, decided before the run.
- `--output-file` with several inputs is written once at the end: the transcripts joined by newlines in input order (a failed input contributes an empty line).
- `--max-parallel N` caps how many jobs of this call run at the same time; it does not affect other callers. `--daemon-workers N` is a deprecated alias (note on `stderr`).
- Exit code: `10` if any input timed out, else `11` if any was unavailable, else `1` if any failed, else `0`.

`--diarization`, `--dialog-blocks` and `--work-dir` are not supported through the supervisor: the command exits with code `2` and suggests `--no-daemon`.

GigaAM examples (deps in `.[ml]`; `gigaam-ctc` — CPU, `gigaam-e2e-ctc` — CPU or CUDA):

```powershell
python cli.py external-transcribe "C:\media\call.wav" --stt-provider gigaam-ctc
python cli.py external-transcribe "C:\media\call.wav" --stt-provider gigaam-e2e-ctc --device cpu
python cli.py external-transcribe "C:\media\call.wav" --stt-provider gigaam-e2e-ctc --device cuda
```

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
python cli.py external-transcribe INPUT_PATH [INPUT_PATH ...] [options]
```

Input:

- `INPUT_PATH` — an audio or video file; several files are allowed through the supervisor (`--no-daemon` accepts exactly one).

Output (one input):

- By default the command prints the final transcript text to `stdout`.
- If `--output-file` is specified, the same plain text is additionally written to that file; `--output-dir` additionally writes `<stem>.txt`.
- With `--no-stdout`, you must also pass `--output-file` or `--output-dir`.
- Several inputs: see "Batch calls" above.

Exit codes:

- `0` — transcription completed successfully, including the case of an empty or whitespace-only result after `str.strip()`.
- `1` — execution error with no valid successful result; through the supervisor `stderr` carries the worker's error text.
- `2` — invalid arguments (for example `--work-dir` without `--no-daemon`, an unknown `--priority`).
- `10` — the call deadline (`--client-timeout`) passed before the transcript was ready; `stderr` starts with `ASKVLM_CLIENT_TIMEOUT:`.
- `11` — the supervisor was unavailable, or the request can never fit the configured limits, or its model failed to load; `stderr` starts with `ASKVLM_DAEMON_UNAVAILABLE:`.
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
- `--priority normal`

Through the supervisor the device policy of "STT supervisor and worker pool" applies, and models are loaded in strict device mode: no CUDA→CPU fallback happens inside a worker (a GPU out-of-memory error fails the attempt; the supervisor retries an `auto` job on CPU and a `cuda` job on the GPU). The rules below describe the in-process model path used by legacy `--no-daemon` and by the other AskVLM commands.

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
5. A GPU out-of-memory error on a given 20-second window moves the model to CPU, releases the CUDA cache, and retries that same window on CPU; subsequent windows stay on CPU until the process restarts.
6. Unloading the model (`close`/`unload`) after CUDA work releases the CUDA cache (`torch.cuda.empty_cache`).

On CUDA the encoder runs under fp16 autocast, so the text may differ slightly from a CPU run (on the 5-minute test file — 2 words out of 453). A resident CUDA model processes a 5-minute file about 3.5× faster than on CPU (RTX 3070 measurement with the earlier resident daemon: about 2.9 s versus 10 s per job, including client startup); in one-shot `--no-daemon` mode on short files CUDA is slower than CPU because of CUDA initialization and the child process. Throughput of the pool modes is measured in `doc/STT_CONCURRENCY_BENCHMARK.en.md`.

Whisper-only parameters (`--compute-type`, beam/VAD) are not sent to GigaAM.

This way an overloaded GPU does not block local integration as long as there is enough RAM in the system.

The fallback guarantee applies to the standard single-pass Whisper path and to `gigaam-e2e-ctc` in legacy `--no-daemon` mode. Options like diarization may require additional GPU capacity. `gigaam-ctc` does not participate in CUDA→CPU recovery. The client performs no CPU recovery of its own.

## Reliability on Windows (subprocess isolation)

For `external-transcribe --no-daemon` on Windows with `--device != cpu` (Whisper and `gigaam-e2e-ctc`; `gigaam-ctc` always runs in-process on CPU), AskVLM runs transcription in a child process (through the supervisor, worker processes in Job Objects provide the isolation):

1. The parent launches the same CLI entrypoint with internal hidden child-mode flags.
2. The child writes a service JSON result before `pipeline.close(...)`.
3. The parent treats this JSON as the source of truth:
   - if the JSON is valid and contains `status=ok`, this is success even with a crash-like child exit code after the result was written;
   - if no valid successful JSON is found and the exit code looks like a native crash, the parent makes exactly one CPU retry;
   - if no valid successful JSON is found, the error is not crash-like, and stderr does not contain an internal IPC error marker — the parent surfaces the error without a CPU retry;
   - if no valid successful JSON is found and stderr contains an internal IPC error marker (child-process initialization failure), the parent makes exactly one CPU retry as a safety measure.

## JIT Loading and Unloading

In legacy `--no-daemon` mode the command is designed for single-shot use (through the supervisor, workers keep models resident until their idle TTL):

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
python cli.py external-transcribe "C:\media\meeting.mp3" --no-daemon --diarization
```

Enable LLM text cleanup:

```powershell
python cli.py external-transcribe "C:\media\draft.wav" --no-daemon --dialog-blocks
```

These options are disabled by default: they increase startup cost and may load additional ML backends. They work only in legacy `--no-daemon` mode; through the supervisor the command exits with code `2`.

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

Through the supervisor the deadline is the call start + `client_timeout` (default 300 s) and covers queue time, model loading and transcription. When it expires, the client sets a cancel marker; the worker stops at a 20-second window boundary (a worker that does not stop within the cancel grace is killed).

Callers need to scale `--client-timeout` with media duration. For files longer than a few minutes the default may be insufficient: hours of audio on CPU GigaAM will not fit in 300 s. After timeout, cancel is cooperative between windows, not instantaneous inside `model.transcribe`.

## Notes

- AskVLM stores model caches in the project's `.cache/` directory.
- `external-transcribe` takes one file per call or a batch of files through the supervisor (`--no-daemon`: one file).
- `external-extract-frames` is designed for one video file per process invocation; frames persist in `--output-dir` after the process exits.
- For batch export to `txt`, `srt`, `vtt`, or `json`, use `python cli.py transcribe ...`.
