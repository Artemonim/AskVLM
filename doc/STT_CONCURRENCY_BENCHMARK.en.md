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
