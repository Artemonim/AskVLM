# STT concurrency benchmark (2026-10-14)

## Why

Before moving `external-transcribe` from a singleton daemon to a supervisor with a worker pool, the worker internals had to be chosen:

- **P**: N separate processes, each with its own copy of the model;
- **S**: one process, one model, N threads over shared weights;
- **B**: batching, several inputs in one model pass.

The measurement ran on the target PC, raising the load until speed or quality dropped.

## Tool

`tools/benchmark_stt_concurrency.py` (package `tools/stt_concurrency/`):

- `run`: fixed configuration matrix;
- `sweep`: saturation search per group (engine × device × mode × threads per stream);
- `combo`: two configurations at once (CPU + GPU), with each side's slowdown against its solo measurement;
- `memory-growth`: memory of one long-lived GPU pool worker, job by job (section "Memory growth of a long-lived GPU worker");
- `summarize`, `list`, and `--dry-run` (writes nothing).

Safety:

- free commit and VRAM are checked before each step;
- during a step, a Job Object puts a hard commit cap on the whole process tree (the `memory_cap` outcome is a saturation point);
- a system free-commit watchdog (`system_commit_low`);
- children die with the parent;
- resume works with a workload fingerprint (results with a different fingerprint are re-measured).

Example (data paths are passed as arguments):

```powershell
python tools/benchmark_stt_concurrency.py sweep --clips-dir <16 kHz WAV dir> --manifest <reference JSONL> --out-dir <outside the repo> --dry-run
python tools/benchmark_stt_concurrency.py sweep --clips-dir <...> --manifest <...> --out-dir <...>
python tools/benchmark_stt_concurrency.py combo --auto-combos --clips-dir <...> --manifest <...> --out-dir <...>
```

## Conditions

- Ryzen 9 5950X (16 cores / 32 threads), 64 GB RAM, RTX 3070 8 GB, Windows 11, Python 3.11.9, torch 2.10 + cu128, faster-whisper 1.2.1.
- The production bot and its dependencies were stopped; background load was minimal.
- Data: FLEURS ru_ru test, first 120 clips (~11 s average), WER against `ref_fleurs_norm`.
- Each step ran at least 45 s and at least 12 clips per stream.
- Whisper decoding was identical for all modes: beam 5, no VAD, no conditioning, temperature 0.
- Single run, no repeats; xRT = audio seconds / wall seconds.

## Results: group maxima

"Maximum" means the best step of a group. Some CPU groups ended at the machine limit (N×T ≤ 32 logical threads, still gaining 8–10% on the last step) rather than at saturation. The GigaAM P×1 scan was stopped by the memory guard at P12×1 (+26% on the last step), so P16×1 was not measured.

| Engine, device | Chosen mode | xRT | Memory (commit / VRAM) | Alternatives |
|---|---|---|---|---|
| GigaAM-v3 e2e_ctc, CPU | **S8×4** (1 process, 8 threads × 4) | 68.7 | 4.1 GB | P8×4: 68.6× at ~30 GB; P12×1: 65.1× at 42 GB; frugal S12×1: 64.1× at 3.7 GB |
| GigaAM-Multilingual ctc, CPU | **S8×4** | 69.1 | ~3.5 GB | P8×4: 67.6× |
| Whisper small int8, CPU | **S6×2** | 11.0 | 2.6 GB | P6×2: 11.2× at 8.9 GB; batch B2×16: 7.1× |
| GigaAM-v3 e2e_ctc, CUDA | **trade-off**: B8 (batch of 8 windows in 1 process) uses less memory; P3 (3 processes) is faster | B8: 137.5; P3: 179.3 | B8: 6.1 GB / +1.7 GB; P3: 17.4 GB / +4.7 GB | B16: 138×; P2: 130× at 12 GB / +3.1 GB |
| Whisper small fp16, CUDA | **B4** (batch of 4 clips) | 35.4 | 3.1 GB / +1.3 GB | P2–P3: ~25× (GPU 95% busy); B8 rejected on quality |

Under a shared 16-thread budget (N×T = 16, closer to real work next to other workers):

| Engine, CPU | S4×4 | P4×4 |
|---|---|---|
| GigaAM-v3 e2e_ctc | 57.9× at 4.1 GB | 58.8× at 15.4 GB |
| GigaAM-Multilingual ctc | 59.5× | 58.4× |
| Whisper small int8 | 10.7× at ~2.2 GB | 11.1× at ~6.1 GB |

Observations:

- **CPU.** At the same N×T, threads over shared weights match processes in speed and transcripts, with 3–12× less memory. GigaAM produces identical transcripts in every configuration. Whisper int8 output depends on the threads per stream: 64–88% identity against P1×16, WER varies by up to 1.5 pp.
- **GigaAM on CUDA in S mode (threads) is broken.** At S2 the step's corpus WER was 13.15% against 4.24% for the first step; on common clips it rose by 0.53 pp; transcript identity 88.5%, mean similarity 0.902. This mode is not used.
- **GigaAM on CUDA appears to be limited not by the GPU** (16–38% utilization) but by the CPU side of each process (~2.7 cores per process). Both batching and extra processes raise throughput.
- **Whisper on CUDA with batch 8** gives WER +0.61 pp, so the limit is 4. Batching applies only to clips ≤30 s. On long files, fixed 30 s chunks lost some text: similarity 0.945 against the sequential reference in an early smoke run.

## Results: CPU and GPU at the same time

| Pair | GPU side: xRT (solo) | CPU side: xRT (solo) | Total | CPU load |
|---|---|---|---|---|
| GigaAM CUDA P3 + GigaAM CPU S8×4 | 79 (179), −56% | 58 (68.7), −16% | 137 | 100% |
| GigaAM CUDA P3 + Whisper CPU P6×2 | 127 (179), −29% | 8.5 (11.2), −24% | 136 | 53% |
| Whisper CUDA B4 + GigaAM CPU S8×4 | 17.8 (35.4), −50% | 65 (68.7), −5% | 83 | 98% |
| Whisper CUDA B4 + Whisper CPU P6×2 | 22 (35.4), −37% | 10.5 (11.2), −7% | 33 | 44% |

Concurrent CPU load slows GPU workers noticeably, even when the CPU is not fully loaded. The cause was not isolated: it may be core contention, memory bandwidth, PCIe or driver overhead. For GigaAM the pair total (137×) is below the GPU alone (179×).

## Decisions for the worker pool

1. **CPU worker:** one process per configuration; slots are threads over shared weights (`num_workers` for Whisper, a thread-safe inference path for GigaAM).
2. **GPU worker (working hypothesis).** A process with batching: up to 8 windows for GigaAM, up to 4 clips of ≤30 s for Whisper; long Whisper files run sequentially. Several such processes are allowed within the GPU-slot limit. For GigaAM this is a trade-off: one batching process (137×) is more frugal than three plain processes (179×), and "several processes, each batching" has not been measured yet. It must be measured when the pool is implemented. CUDA threads are not used for GigaAM.
3. **One CPU-core budget** that includes the CPU side of GPU workers (~2.7 cores per GigaAM process, ~1–2 for Whisper). CPU spill must not take those cores. For CPU configurations under a shared budget, use the "16 threads" rows above.
4. **Scheduler seed coefficients** come from the tables above; slowdown factors come from the combo runs.

## Limitations

- A single run with no repeats on short clips (FLEURS ~11 s). Long files were only measured in early smoke runs.
- "Several processes, each batching" was not measured.
- Some CPU maxima are bounded by the machine's logical thread count rather than by saturation.
- Raw results are kept outside the repo: `%LOCALAPPDATA%\AskVLM\benchmarks\stt-concurrency-20261014` (fingerprint `ef0abe20d151472e`). They were recorded before the fingerprint gained an audio content hash, so the current tool re-measures everything in that folder; existing summaries can be rebuilt with `summarize`.

## Memory growth of a long-lived GPU worker (2026-10-16)

### Why

In production the long-lived GPU worker `gigaam-e2e-ctc/ai-sage/GigaAM-v3@e2e_ctc/fp32/gpu` grew by about 1 MB of commit per job (~120 MB/h, correlation with uptime 0.96) until it reached its Job Object cap. The goals were to find the growth mechanism, to check whether the pool's memory bounds stop it (`gpu.cufft_plan_cache_max_size`, `gpu.cudnn_v8_lru_cache_limit`, `gpu.idle_cache_release_s`, `gpu.vram_cap`), and to compare torch 2.10.0 with 2.14.1.

### The `memory-growth` tool

The command starts one real pool worker through the production path: `default_worker_launch` and `WorkerProcess` (its own Job Object with a commit cap of the reservation plus `memory.child_headroom_gib`), the `GigaamGpuBatchEngine` engine factory, and a `spec` built from a validated pool config, the seed estimates and the limits, the way the supervisor builds it. It then feeds N jobs strictly one at a time: the next job is sent after the previous result, optionally after a pause. The supervisor is left out on purpose, because its worker replacement (the `memory.recycle_commit_fraction` threshold, the live cap raise, the uptime cap) would hide the growth. Instead, the summary shows the job at which the replacement threshold would have fired.

Inputs are synthetic. The source is Russian speech by the SAPI voice "Microsoft Pavel" through `pwsh`, or a `--source` file such as `tests/fixtures/test_video_short.mp4`. The source is repeated into one long WAV, and FFmpeg cuts the jobs from it at seeded offsets as 16 kHz mono Ogg Opus, like Telegram voice notes. `--durations varied` draws each duration uniformly from `[--duration-min-s, --duration-max-s]` (1–60 s by default, millisecond resolution, so each job has its own input shape). `--durations fixed` gives every job the same duration. No user recordings are used.

For each job the tool records:

- the memory snapshot of the result (private bytes, CUDA reserved/allocated/max reserved, the number of cuFFT plans);
- the Job Object peak;
- the private bytes before the job;
- the whole-GPU used VRAM from nvidia-smi (`--no-gpu-sample` turns it off);
- timings.

The summary reports:

- the baseline after load;
- the least-squares slope per job and per hour over the last 80% of the jobs;
- the shift of the median of the last 20% of the jobs against the 20% before them (a "plateau" means a shift within the tolerance, 16 MiB by default);
- maxima, the cuFFT and CUDA trajectories, throughput, and the pool's replacement threshold.

Safety: a run starts only with at least 20 GiB of free commit and 3 GiB of free VRAM (`--min-free-commit-gib`, `--min-free-vram-gib`) and only when the worker's Job Object cap fits into free commit minus `--commit-reserve-gib` (12 by default). This check comes before anything else, the torch version probe included. The commit rules are checked again after the inputs are cut, right before the worker spawn, because cutting can take minutes while other processes take memory. A refusal at either point ends the run with code 3 before any worker starts and is recorded in `run.json` (`guard` or `spawn_guard`). A running measurement stops when free commit drops below `--commit-reserve-gib`.

Exit codes:

- 0 — every job ran and the summary was written;
- 1 — the run stopped early;
- 2 — invalid input, or the inputs could not be built (status `inputs_failed`; the run folder keeps `run.json` and the summary);
- 3 — the resource check refused the run;
- 4 — every job ran but the summary could not be written;
- 130 — Ctrl+C.

A run that stopped early exits with code 1 even when its summary could not be written either. The pool config comes from the code defaults (`stt_pool.json` and `ASKVLM_STT_POOL_CONFIG` are never read) or from `--pool-config`. `--pool-set PATH=VALUE` overrides single keys. `--gpu-memory-bounds library` turns all four GPU bounds off, which is the behavior from before the bounds existed. Results go to `.cache/stt_memory_growth/<time>-<label>/` (`run.json`, `results.jsonl`, `results.csv`, `summary.json`, `summary.md`, `worker.log`). `summarize --out-dir <run dir>` recomputes the summary.

```powershell
python tools/benchmark_stt_concurrency.py memory-growth --jobs 2000 --label A --idle-gap-s 16 --idle-gap-every 50
python tools/benchmark_stt_concurrency.py memory-growth --jobs 2000 --label B --gpu-memory-bounds library
python tools/benchmark_stt_concurrency.py memory-growth --jobs 2000 --label C --gpu-memory-bounds library --durations fixed --fixed-duration-s 13
python tools/benchmark_stt_concurrency.py memory-growth --jobs 1000 --label V2 --gpu-memory-bounds library --pool-set gpu.cudnn_v8_lru_cache_limit=256
```

### Conditions

- **Machine and stacks.** Same machine, driver 610.88 (CUDA 13.3). The 2.10 stack is the working `.venv`: torch 2.10.0+cu128 (CUDA 12.8, cuDNN 9.10.2), transformers 5.14.1, numpy 1.26.4. The 2.14 stack is a temporary venv: torch 2.14.1+cu130 (CUDA 13.0, cuDNN 9.24.0) and torchaudio 2.11.0 (the last torchaudio release; there is no newer one), with all other packages at the same versions.
- **Pool config.** The code defaults: 8 lanes and a seed reservation (commit 7.0 GiB, Job Object cap 8.0 GiB, replacement threshold 7.0 GiB, VRAM 2.2 GiB, allocator cap 3.2 GiB).
- **Background load.** The production bot (its GPU worker on the old code, ~7.0 GiB of private bytes) and another team's tests ran at the same time, so speed is comparable only within the back-to-back trio (T1–T3).
- **Stopped runs.** The commit guard stopped A at job 1377 and two first attempts (B at job 65, A′ at job 40). Later runs used `--commit-reserve-gib 8`.
- **Job counts.** N = 1000–2000 instead of 300: a job takes ~0.3 s, and a long run shows the slope more precisely.
- **Idle pauses.** In A and A′ every 50th job is followed by a 16 s pause, longer than `gpu.idle_cache_release_s` = 15 s. Back-to-back jobs would never trigger the cache release, while pauses between voice notes are common in production.
- **Shapes.** The `varied` durations (seed 1) are the same in every run. About 32% of the jobs are shorter than the 20 s window, and only those create a new input shape: the windows of longer jobs are zero-padded to 20 s.

### Results

Private bytes are the worker process's own commit, in MiB:

- "Baseline" is the value right after load and warm-up; it drops by about 0.9 GiB after the first job.
- "Level" is the median of the last 20% of the jobs.
- The slope is computed over the last 80% of the jobs.

| Run | Stack | Durations | GPU bounds | Jobs | Baseline | 1st job → level | Slope, MiB/job | 20% shift, MiB | CUDA reserved | cuFFT plans |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 2.10 | varied | pool (32 / 256 / 15 s / cap) | 1377 | 6095 | 5207 → 5471 | +0.006 | +10.5 (plateau) | 1428 | 32 |
| B | 2.10 | varied | library | 2000 | 6152 | 5264 → 7252 (max 7381) | **+0.79** | **+275** | 1460 | 559, growing |
| C | 2.10 | fixed 13 s | library | 2000 | 6128 | 5223 → 5222 | 0.000 | 0.0 | 1460 | 3 |
| V1 | 2.10 | varied | library + cuFFT 32 | 1000 | 6100 | 5211 → 6505 | **+1.15** | **+160** | 1460 | 32 |
| V2 | 2.10 | varied | library + cuDNN 256 | 1000 | 6130 | 5242 → 5539 | +0.020 | +4.1 (plateau) | 1460 | 291 |
| A′ | 2.14 | varied | pool | 1000 | 5819 | 4926 → 4971 | −0.007 | −1.2 (plateau) | 1006 | 32 |
| B′ | 2.14 | varied | library | 1000 | 5845 | 4954 → 5136 | +0.056 | +11.1 | 1098 | 291 |

- **CUDA allocator.** CUDA allocated is constant in every run (864 MiB on 2.10, 862 MiB on 2.14). CUDA reserved does not grow in any run.
- **Job Object peak.** The peak (ffmpeg included) equals the load peak: 6.0–6.1 GiB on 2.10, 5.7–5.8 GiB on 2.14. Without bounds (B) the private bytes catch up with it: by job 2000 the Job Object peak reaches 7428 MiB, and the private-byte maximum is 7381 MiB. B reached the 7.0 GiB replacement threshold at job 1709.
- **Growth by job kind.** In B a job shorter than 20 s (a new shape) added +2.1 MiB on average (median +2.6) and 0.87 cuFFT plans. A longer job added +0.57 MiB on average (median +0.09, because the growth comes in steps).
- **Idle release.** The idle cache release works (27 releases in A). After a release, the first job lifts the private bytes about 200 MiB above the level, and the next job brings them back. The level is not affected.
- **Production rate.** At the production rate (~120 jobs/h), B means ~95 MiB/h and ~1.1 GiB per 12 h, and V1 ~140 MiB/h. This matches the growth seen in production; production likely has a higher share of short voice notes than the 32% here.

torch 2.14.1 against 2.10.0, back-to-back runs on a quiet machine (fixed 30 s, pool bounds, 300 jobs each: 2.10 → 2.14 → 2.10):

| | 2.10 (T1 / T3) | 2.14 (T2) |
|---|---|---|
| Private bytes after load | 6129 / 6098 MiB | 5788 MiB |
| Private-byte level | 5209 / 5102 MiB | 4772 MiB |
| CUDA reserved | 1456 / 1376 MiB | 992 MiB |
| Whole-GPU VRAM after load (increase) | 1640 / 1645 MiB | 1154 MiB |
| Jobs per minute (xRT) | 203 (102) / 220 (110) | 187 (93) |
| Median job | 0.25 / 0.24 s | 0.28 s |

On the 1000 inputs shared by A and A′ every transcript had exactly the same length; the texts themselves were not stored.

### Conclusions

1. **Mechanism (high confidence).** The growth is host memory of the process (private bytes), not the CUDA cache, and it is driven by the variety of input shapes:
   - With one duration (C), 2000 jobs added not a single byte, so there is no per-job leak.
   - With varied durations and no bounds (B), the worker grows by +0.79 MiB/job with no plateau, while reserved and allocated stay constant.
   - Almost all of the growth comes from the cuDNN v8 plan cache (`TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT`, 10000 by default). Bounding cuFFT alone (V1) does not change the growth: over jobs 201–1000, V1 grows +1.15 MiB/job and B +1.17. Bounding cuDNN alone (V2) stops it: +0.02 MiB/job, although there are 291 cuFFT plans and their number keeps growing.
   - cuFFT plans cost ~0.1 MiB each and matter only with thousands of shapes.
   - The number of cuDNN plans is not visible from outside. At torch's estimate of ~200 KiB per plan, the 10000 limit amounts to ~2 GiB, which is about what B had accumulated by job 2000; whether B was saturating is not verified. Either way, the default bound lies above the worker's budget (replacement threshold 7 GiB, cap 8 GiB).
2. **The pool bounds stop the growth (high confidence).**
   - A grows by +0.006 MiB/job, with a plateau at 5471 MiB (5.34 GiB) from about job 140 to job 1377, and CUDA reserved stays at ~1.4 GiB under the 3.2 GiB cap. A′ (2.14) is flat as well.
   - The headroom to the replacement threshold (~1.7 GiB above the level) and to the VRAM cap (~1.8 GiB) covers the short spike after a cache release.
   - No unbounded growth remains on these inputs. The 12 h uptime cap stays as the backstop for what this measurement does not cover: other durations and languages, multi-hour files, growth of the CUDA context over a day.
3. **torch 2.14.1 (medium confidence).**
   - **Memory.** It uses less: −0.3 GiB of commit after load, −0.3…−0.5 GiB at the level, −0.38…−0.45 GiB of CUDA reserved, and −0.4…−0.5 GiB of whole-GPU VRAM.
   - **Growth.** Without bounds it grows about 20 times slower (+0.056 MiB/job against +1.17 over jobs 201–1000). The pool bounds are still needed there, because the cuFFT plan count grows the same way.
   - **Speed.** It is not faster. In the back-to-back trio 2.14 is 8–15% slower, but that is a single run, and 2.10 itself varied by 8% between T1 and T3. The varied runs under foreign load looked better for 2.14, but they ran at different times and are not comparable.
   - **Compatibility.** GigaAM works with the transformers 5.14.1 patches. torchaudio is frozen at 2.11.0. The model's remote code needs an importable `pyannote` (transformers `check_imports`), as in the working venv.
   - **Switching.** It would require changing the `torch>=2.10,<2.11` pin in `pyproject.toml` and the torch 2.10 + cu128 CUDA repair in `run.ps1`/`build.ps1`.

### Measurement limitations

- One run per configuration. The 2.10 vs 2.14 speed comparison is a single T1–T3 trio.
- The duration distribution is uniform over 1–60 s rather than the production one, with a single language and voice.
- The Job Object peak over the process lifetime is a running maximum, not the peak of one job.
- Whole-GPU VRAM includes other processes. CUDA reserved describes the worker's memory more precisely.
- Raw data is kept out of Git, in `.cache/stt_memory_growth/` of the working copy that ran the measurement, runs `20261016T…-<label>`: A, B, C, V1, V2, A′ = `A-varied-pool-t214`, B′ = `B-varied-library-t214`, T1–T3. The load of run `20261016T184215Z-A-varied-pool-t214` failed because the temporary venv lacked `pyannote`. Its `summary.json` was rebuilt with status `unknown`.
