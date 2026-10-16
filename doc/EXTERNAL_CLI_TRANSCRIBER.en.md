# External CLI Transcriber

## Purpose

`python cli.py external-transcribe ...` — a stable CLI entry point for local applications that pass AskVLM media files and receive transcript text. One input keeps the "one call — one file" contract (plain text on `stdout`); several inputs in one call return one JSON line per input. With `--format json` the command returns a structured document (segments with words and scores in the format of the hardsub skill's `transcribe.py`), with `--diarization` also the speaker turns; the separate command `python cli.py external-diarize ...` returns only the turns (sections "Structured output", "Speaker diarization", "The `external-diarize` command").

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

1. The client starts one wall-clock deadline (`--client-timeout`, default 300 s) for the whole call, probes each input's duration with `ffprobe` (a failed probe only means "duration unknown"), and writes one job per input (two with `--diarization`: transcription and diarization) into the file queue `<queue root>/v2/` (queue root: `ASKVLM_EXTERNAL_QUEUE_DIR` or `<project>/.cache/external_queue`). All inputs of one call share a call id, `--priority` and `--max-parallel`.
2. If no supervisor heartbeat is fresh, the client starts a **detached** supervisor (with `CREATE_BREAKAWAY_FROM_JOB` when the caller's job object allows it, so the supervisor survives the caller) and keeps re-starting it at most every 10 s while none answers. If no supervisor becomes reachable within 30 s, the unfinished inputs end as `unavailable` (exit `11`); the message names the supervisor log `<queue root>/v2/control/supervisor.out.log`.
3. The client waits for the results until the deadline. When the deadline passes, it writes cancel markers for the unfinished jobs (the supervisor drops or stops them) and reports `timeout` (exit `10`). `--client-timeout inf` waits without a deadline.
4. There is **no in-process CPU recovery** in the client: a GPU job that cannot finish is not re-run on CPU inside the client process.

### Supervisor and workers

- **One supervisor per queue root**, guarded by an OS file lock (`v2/control/supervisor.lock`); a second supervisor exits at once with code `0`. A heartbeat (`v2/control/supervisor.heartbeat.json`) carries the limits, the workers and the queue size.
- **A worker is a process** that keeps one model resident and serves one configuration (a worker key): provider, model, compute type normalized per backend (Whisper: `int8` on CPU, `float16` on GPU for `auto`; GigaAM: `fp32`), and backend `cpu` or `gpu`. Language is per job and never part of the key.
  - CPU worker: one model shared by several concurrent jobs (CPU slots); each slot runs on `threads_per_slot` threads (default 2).
  - Diarization worker (key `sherpa-diarize/pyannote-seg-3.0+wespeaker-resnet34-lm/fp32/cpu`): CPU only and exactly one slot per process, because sherpa-onnx holds the GIL for its whole run; parallelism comes from several processes. The slot runs on `diarization.threads_per_slot` threads (default 4) and shares the CPU slots, the thread budget and the commit budget with STT.
  - GPU worker: one process that batches across jobs — up to 8 GigaAM 20-second windows or up to 4 Whisper clips of at most 30 s with an explicit language (other Whisper jobs run sequentially). GigaAM never runs CUDA inference from several threads.
- **Workers start on demand** for the configurations that are queued and stop after an idle TTL (default 600 s, `--idle-shutdown`). When a new configuration does not fit, idle workers of other configurations are stopped first (least recently used). Busy workers are never stopped to free resources. The supervisor exits shortly after the last worker stops with an empty queue.
- **Every worker lives in its own Windows Job Object** with a memory limit equal to its commit reservation plus `memory.child_headroom_gib` (default 1 GiB). The limit covers the whole process tree, so child processes such as `ffmpeg` count against it; the headroom is not reserved and not counted in the commit budget. The whole tree dies with the job, so no orphaned GPU processes remain. When the job reaches its limit (in the worker or in a child), the next reservation for that configuration grows by 25% (once per worker); a ready worker takes no new jobs and stops once its running jobs end (a worker that hit the limit while loading still serves), and a job that failed on that worker without a more specific error kind is retried on a fresh worker as `memory_limit` while attempts remain. `ffmpeg` exit codes that mean "out of memory" (`0xC000012D`, `0xC0000017`, `-12`) count as a memory failure as well. Workers run with the Windows error dialogs turned off, so a child that cannot start (for example with `0xC000012D`) exits with its code instead of waiting for a click on an "Application Error" box.
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
| `diarization.threads_per_slot` | Threads of a diarization worker slot (default 4; the default is clamped to `threads.budget`, an explicit value above the budget is a configuration error). Pool commands have no `--threads` flag |
| `memory.commit_budget_gib`, `memory.free_commit_guard_gib`, `memory.critical_free_commit_gib` | Commit budget for all workers together, free-commit guard for a new worker, free commit below which idle workers are stopped |
| `memory.child_headroom_gib` | Commit a worker's Job Object allows above the worker's reservation for its child processes (`ffmpeg`), default 1; not reserved and not counted in the commit budget; `0` caps the job at the reservation |
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

- With several inputs `stdout` carries one JSON object per input, in input order: `{"index", "input", "status", "text", "error"}` (plus `"output"` with `--output-dir`; with `--format json` also `"result"`: the input's document or `null`); a line is written as soon as all earlier inputs are finished. `status` is `ok`, `empty`, `error`, `timeout` or `unavailable`.
- `--output-dir DIR` writes `<stem>.txt` for every successful input; repeated stems get `-2`, `-3`, … suffixes, decided before the run.
- `--output-file` with several inputs is written once at the end: the transcripts joined by newlines in input order (a failed input contributes an empty line); with `--format json` JSON Lines of the documents in input order (`null` for a failed input). `--output-dir` with `--format json` writes `<stem>.json`.
- `--max-parallel N` caps how many jobs of this call run at the same time; it does not affect other callers. `--daemon-workers N` is a deprecated alias (note on `stderr`).
- Exit code: `10` if any input timed out, else `11` if any was unavailable, else `1` if any failed, else `0`.

`--dialog-blocks` and `--work-dir` are not supported through the supervisor: the command exits with code `2` and suggests `--no-daemon`. `--diarization` works through the supervisor (section "Speaker diarization").

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

The `[ml]` stack includes torch/torchaudio **2.10** (CUDA wheels repaired by `run.ps1`/`build.ps1` by default: cu128 → cu126), transformers 5, hydra-core, omegaconf, **sentencepiece**, **pyannote.audio**, and **sherpa-onnx** 1.13.8 (CPU speaker diarization, section "Speaker diarization"). `sentencepiece` and `pyannote.audio` are required by remote-code modeling even for short-form `.transcribe` (no longform/VAD). On CPU, `gigaam-ctc` uses roughly ~2.5 GB RAM and `gigaam-e2e-ctc` ~2.9 GB RAM (vs compact Whisper Small CPU); on CPU neither uses VRAM. `gigaam-e2e-ctc` on CUDA uses about 1.5–1.65 GB VRAM per process (CUDA context, fp32 weights ~0.85 GB, 20 s window activations under fp16 autocast; nvidia-smi measurement on an RTX 3070, torch 2.10 + cu128).

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

- By default the command prints the final transcript text to `stdout`; with `--format json` a JSON document (section "Structured output").
- If `--output-file` is specified, the same plain text is additionally written to that file; `--output-dir` additionally writes `<stem>.txt` (with `--format json`: `<stem>.json`).
- With `--no-stdout`, you must also pass `--output-file` or `--output-dir`.
- Several inputs: see "Batch calls" above.

Exit codes:

- `0` — transcription completed successfully, including the case of an empty or whitespace-only result after `str.strip()`.
- `1` — execution error with no valid successful result; through the supervisor `stderr` carries the worker's error text.
- `2` — invalid arguments (for example `--work-dir` without `--no-daemon`, an unknown `--priority`, a Whisper decoding option with GigaAM, an invalid `--start`/`--duration` window, `--format json` together with `--dialog-blocks`).
- `10` — the call deadline (`--client-timeout`) passed before the transcript was ready; `stderr` starts with `ASKVLM_CLIENT_TIMEOUT:`.
- `11` — the supervisor was unavailable, or the request can never fit the configured limits, or its model failed to load; `stderr` starts with `ASKVLM_DAEMON_UNAVAILABLE:`.
- Any other non-zero code — AskVLM failed to process the file or execution-stage failure.

Empty transcript: if `get_full_text()` returns a string that is empty after `strip()`, this is treated as a normal successful completion. With `--format json` the document is printed in this case too. In text mode the command:

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
- `--format text`
- no window (`--start 0`, to the end) and no explicit Whisper decoding options (pipeline values: beam 10, VAD on, word timestamps on, condition_on_previous_text on)

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

`--compute-type` is ignored for GigaAM. Explicit Whisper decoding options (`--beam-size`, `--vad/--no-vad`, `--condition-on-previous-text/--no-condition-on-previous-text`, `--word-timestamps/--no-word-timestamps`) together with GigaAM are an error (exit `2`) with an explanation, even when the value equals the default.

This way an overloaded GPU does not block local integration as long as there is enough RAM in the system.

The fallback guarantee applies to the standard single-pass Whisper path and to `gigaam-e2e-ctc` in legacy `--no-daemon` mode. Extra options (for example `--dialog-blocks`) may require additional memory; diarization runs on the CPU only. `gigaam-ctc` does not participate in CUDA→CPU recovery. The client performs no CPU recovery of its own.

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

Enable speaker diarization (through the supervisor and with `--no-daemon`, section "Speaker diarization"):

```powershell
python cli.py external-transcribe "C:\media\meeting.mp3" --diarization
python cli.py external-transcribe "C:\media\meeting.mp3" --diarization --format json --output-file "C:\media\meeting.json"
```

Enable LLM text cleanup (legacy `--no-daemon` only; through the supervisor the command exits with code `2`):

```powershell
python cli.py external-transcribe "C:\media\draft.wav" --no-daemon --dialog-blocks
```

These options are disabled by default: they increase the cost and load additional models.

## Structured output (`--format json`)

`--format json` (default `text`) returns one JSON document per input in the format of the hardsub skill's `transcribe.py`:

```json
{
  "media": "C:\\media\\call.wav",
  "sample_rate": 16000,
  "segments": [
    {
      "start": 0.5,
      "end": 2.58,
      "text": "Hello, my name is Zira.",
      "no_speech_prob": 0.012,
      "avg_logprob": -0.21,
      "words": [{"word": "Hello,", "start": 0.5, "end": 0.92, "probability": 0.97}]
    }
  ]
}
```

- `media` is the input path as it was passed to the command; `sample_rate` is always `16000`.
- Segment and word times are seconds from the media start (including `--start`), rounded to milliseconds; `no_speech_prob`, `avg_logprob` and `probability` are rounded to three decimals.
- Whisper: every segment has `no_speech_prob`, `avg_logprob` and `words` (`[]` with `--no-word-timestamps`). GigaAM: the segments are its 20-second windows with `start`, `end`, `text`; they carry no Whisper fields (the keys are absent, not `null`).
- With `--diarization` the document gains `turns` (section "Speaker diarization") and every segment gains, as its last key, `speaker`: the label with the largest total overlap, or `null` when the segment overlaps no turn.
- One input: the indented document goes to `stdout` (also for an empty transcript, exit `0`); `--output-file` receives the same document, `--output-dir` writes `<stem>.json`.
- Several inputs: the JSON Lines rows on `stdout` gain `"result"` with the document (`null` for a failed input); `--output-file` receives JSON Lines of the documents in input order (`null` for a failed input).
- Encoding: JSON on `stdout` escapes non-ASCII characters (`\uXXXX`), so it can be read with any console or pipe encoding; files are written in UTF-8 without escaping, with platform line endings (CRLF on Windows, like the hardsub skill's files). Text mode prints text as before: callers should run Python in UTF-8 mode (`python -X utf8 cli.py ...` or `PYTHONUTF8=1`) when the transcript may contain characters outside the console code page.
- Through the supervisor Whisper `--format json` jobs on the GPU take the worker's sequential path, not the clip batch; GigaAM on the GPU builds the segments from the windows of its batch.
- `--format json` together with `--dialog-blocks` is an error (exit `2`): the JSON is built from the raw segments, LLM formatting does not affect it.

## Whisper decoding options and media window

- `--beam-size N` (1–64), `--vad/--no-vad`, `--condition-on-previous-text/--no-condition-on-previous-text` and `--word-timestamps/--no-word-timestamps` are Whisper-only. Without a flag the AskVLM pipeline value applies: beam 10, VAD on, word timestamps on, condition_on_previous_text on (the faster-whisper default). The hardsub skill's settings (`transcribe.py`): `--beam-size 5 --no-vad --no-condition-on-previous-text --word-timestamps`.
- Any of these options given explicitly with `--stt-provider gigaam-ctc|gigaam-e2e-ctc` is an error (exit `2`) with an explanation.
- `--start S` (≥ 0, default 0) and `--duration D` (≥ 0.001 s, default: to the end) limit processing to a media window (`ffmpeg -ss` before `-i`, `-t` after); JSON times stay relative to the media start. A window that starts beyond the end of the file is an error of that input (exit `1`): when the file duration is known (`ffprobe`) no job is queued, otherwise the worker fails the job with "no audio decoded".
- Through the supervisor Whisper jobs on the GPU with explicit options or a window take the worker's sequential path, not the clip batch; GigaAM on the GPU applies the window inside its batch; plain jobs are routed as before. The new job fields are written as job file version 2; jobs without them are still written as version 1, so an old supervisor accepts them and explicitly rejects a version 2 job (`unsupported_version`) instead of losing its options.

## Speaker diarization

`--diarization` finds who speaks when, on the CPU: sherpa-onnx 1.13.8 with pyannote segmentation 3.0 (ONNX) and WeSpeaker ResNet34-LM — the same models and settings as the hardsub skill's `diarize.py`. No Hugging Face token and no GPU are needed.

- Works through the supervisor and with `--no-daemon`. Through the supervisor the client queues two jobs per input (transcription and diarization) that share the call id, priority, deadline and `--max-parallel` accounting (both count), and merges the results itself, without ML packages. A failure of either part fails the whole input (a diarization failure is prefixed `speaker diarization:`), and the other part is cancelled.
- Speaker labels are `speaker_00`, `speaker_01`, … in order of each voice's first appearance in time.
- Text (`--format text`): if fewer than two distinct voices are assigned to the transcript segments, the text is exactly the text without diarization, unprefixed. Otherwise every non-empty segment line starts with `speaker_xx: `; a segment that overlaps no turn stays unprefixed.
- JSON: `turns` is the list of turns `{"start", "end", "speaker"}` in start order; a segment's `speaker` is the label with the largest total overlap, `null` without overlap.
- CPU only: in `external-transcribe` diarization runs on the CPU whatever `--device` says (it applies to the transcription); in `external-diarize` `--device cuda` is an error (exit `2`).
- Pool resources: separate diarization workers with one slot per process (section "Supervisor and workers"), threads from `diarization.threads_per_slot`. Measured: about 0.1 s of processing per second of audio on 4 threads; a process peak of about 1 GB on 6 minutes of audio (sherpa keeps a copy of the samples, about 0.5 GB per hour of audio). The default commit reservation (0.8 + 1.5 GiB) covers about 2 hours of audio; a worker that exceeds its reservation fails and the next reservation grows by 25%.
- Cancellation: the first stage (segmentation, about 20–25% of the run) cannot be interrupted; a cancel takes effect at the next embedding chunk, and a worker that does not answer within `lifecycle.cancel_grace_s` is stopped (not counted as a crash).
- `--no-daemon --diarization`: diarization runs in the command's own process (4 threads) before the transcription model loads, so two models are never loaded at the same time. With `--dialog-blocks` consecutive lines of one speaker are joined. The GUI keeps its own pyannote diarization.
- Diarization separates voices, not characters: short sounds (under ~1 s) may form a cluster of their own, and one voice may split in two. Tune `--num-speakers`/`--threshold` of `external-diarize` per material.

## The `external-diarize` command

```text
python cli.py external-diarize INPUT_PATH [INPUT_PATH ...] [options]
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--num-speakers N` | — | Exact number of speakers (≥ 1); without it voices are clustered by `--threshold` |
| `--threshold T` | `0.5` | Clustering threshold (> 0): larger merges voices, smaller splits them |
| `--min-on S` | `0.3` | Shortest kept turn, seconds |
| `--min-off S` | `0.5` | Pause that splits two turns of one speaker, seconds |
| `--start`, `--duration` | `0`, to the end | Media window, as in `external-transcribe` |
| `--device auto\|cpu\|cuda` | `auto` | CPU only: `auto` means `cpu`, `cuda` is an error (exit `2`) |
| `--priority`, `--max-parallel`, `--client-timeout` | as in `external-transcribe` | Supervisor mode |
| `--output-file`, `--output-dir`, `--stdout/--no-stdout` | | As in `external-transcribe --format json` (`<stem>.json`) |
| `--no-daemon` | off | One input in the command's own process (4 threads) |

The pool's diarization threads come from `diarization.threads_per_slot` in the pool configuration; there is no `--threads` flag.

Output:

- One input: `{"media": ..., "turns": [{"start", "end", "speaker"}, ...]}`, indented (the hardsub skill's `diarize.py` format; times from the media start, rounded to milliseconds). Empty `turns` (for example silence) is a success, exit `0`. A file without decodable audio is an error, exit `1`.
- Several inputs: JSON Lines `{"index", "input", "status", "error", "result"}` (plus `"output"` with `--output-dir`), `result` is the input's document or `null`; `status` is `ok`, `error`, `timeout` or `unavailable`. `--output-file` receives JSON Lines of the documents (`null` for a failed input).
- Exit codes and `stderr` markers as in `external-transcribe`: `0`, `1`, `2`, `10` (`ASKVLM_CLIENT_TIMEOUT:`), `11` (`ASKVLM_DAEMON_UNAVAILABLE:`). JSON on `stdout` is ASCII-safe, files are UTF-8.

```powershell
python cli.py external-diarize "C:\media\dialog.mp4" --num-speakers 2
python cli.py external-diarize a.wav b.wav --threshold 0.7 --output-dir C:\out
python cli.py external-diarize "C:\media\dialog.mp4" --start 60 --duration 30 --no-daemon
```

## Diarization models: download, cache and security

- The models are downloaded on first use (about 33 MB) into `<project>/.cache/models/sherpa-diarization/` (`.cache` is git-ignored); later runs use the cache without network access.
- The sources are pinned in `core/diarization_models.py`: the archive `sherpa-onnx-pyannote-segmentation-3-0.tar.bz2` and the file `wespeaker_en_voxceleb_resnet34_LM.onnx` from the `k2-fsa/sherpa-onnx` GitHub releases, with the SHA-256 and sizes of the archive and both models.
- Downloads use HTTPS only (a redirect to non-HTTPS is refused). The stream goes into a temporary `.part` in the same folder with a size cap and a streaming SHA-256 and is renamed atomically only after verification; on any error the `.part` is deleted. The archive hash is verified before the archive is opened; exactly one file with the exact name is read from it (a regular file, no links, size-capped) — no full extraction, no shell, no external processes. A cached model is used only when its SHA-256 matches, otherwise it is replaced. A cross-process lock (`diarization-models.lock`) keeps two processes from downloading at the same time.
- A download or verification failure is an explicit error: through the supervisor a worker load failure (`unavailable` with the text), with `--no-daemon` exit `1`.
- The dependency `sherpa-onnx==1.13.8` is part of `.[ml]`; `run.ps1`/`build.ps1` install it.

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
