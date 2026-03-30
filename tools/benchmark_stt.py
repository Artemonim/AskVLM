from __future__ import annotations

import concurrent.futures
import csv
import json
import queue
import re
import threading
import time
from dataclasses import dataclass
import contextlib
import sys
from pathlib import Path
from typing import Callable, Iterable, Literal
import os
import subprocess as _subprocess
import json as _json

import typer

# * Make direct script execution work without a preconfigured PYTHONPATH.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.audio_io import prepare_audio, cleanup_intermediate_audio
from core.ffmpeg import get_media_duration_seconds
from core.whisperx_wrapper import WhisperXWrapper


# * Benchmark variants configuration -----------------------------------------------------------


@dataclass(frozen=True)
class WorkerSpec:
    model_name: Literal["small", "large-v3"]
    device: Literal["cuda", "cpu"]
    count: int


@dataclass(frozen=True)
class Variant:
    name: str
    workers: list[WorkerSpec]


VARIANTS: list[Variant] = [
    Variant("1x large-v3 GPU", [WorkerSpec("large-v3", "cuda", 1)]),
    Variant("1x large-v3 CPU", [WorkerSpec("large-v3", "cpu", 1)]),
    Variant(
        "1x large-v3 GPU + 1x large-v3 CPU",
        [WorkerSpec("large-v3", "cuda", 1), WorkerSpec("large-v3", "cpu", 1)],
    ),
    Variant("1x small GPU", [WorkerSpec("small", "cuda", 1)]),
    Variant("2x small GPU", [WorkerSpec("small", "cuda", 2)]),
    Variant(
        "1x small GPU + 1x small CPU",
        [WorkerSpec("small", "cuda", 1), WorkerSpec("small", "cpu", 1)],
    ),
    Variant(
        "2x small GPU + 1x small CPU",
        [WorkerSpec("small", "cuda", 2), WorkerSpec("small", "cpu", 1)],
    ),
    Variant(
        "2x small GPU + 2x small CPU",
        [WorkerSpec("small", "cuda", 2), WorkerSpec("small", "cpu", 2)],
    ),
]


# * Utilities ----------------------------------------------------------------------------------


def _list_media_files(directory: Path, exts: Iterable[str]) -> list[Path]:
    return [p for p in sorted(directory.iterdir()) if p.is_file() and p.suffix.lower() in exts]


def _normalize_text_for_wer(text: str) -> list[str]:
    # * Lowercase, strip ASS/formatting artifacts, keep letters/digits, normalize spaces
    t = text.lower()
    t = re.sub(r"\\n|\\N", " ", t)
    t = re.sub(r"\{[^}]*\}", " ", t)  # ASS overrides
    # Python's stdlib 're' does not support Unicode \p classes; filter manually
    t = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in t)
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return []
    return t.split()


def _edit_distance(ref: list[str], hyp: list[str]) -> int:
    # * Classic DP Levenshtein distance for token lists
    n, m = len(ref), len(hyp)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        ri = ref[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ri == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )
    return dp[n][m]


def wer_score(reference_text: str, hypothesis_text: str) -> tuple[float, int, int]:
    """Compute WER; returns (wer, edit_distance, ref_token_count)."""
    ref_tokens = _normalize_text_for_wer(reference_text)
    hyp_tokens = _normalize_text_for_wer(hypothesis_text)
    if not ref_tokens:
        return (0.0, 0, 0)
    dist = _edit_distance(ref_tokens, hyp_tokens)
    return (float(dist) / float(len(ref_tokens)), dist, len(ref_tokens))


def parse_ass_dialogue_texts(ass_file: Path) -> list[str]:
    # * Parse ASS: extract Dialogue lines' Text field (after 9 commas)
    out: list[str] = []
    try:
        text = ass_file.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return out
    for line in text.splitlines():
        if not line.startswith("Dialogue:"):
            continue
        body = line[len("Dialogue:"):].lstrip()
        parts = body.split(",", 9)  # 10 parts; last is Text
        if len(parts) < 10:
            continue
        text_field = parts[9]
        if text_field:
            out.append(text_field)
    return out


def ass_reference_text_with_filter(
    ass_file: Path,
    recognizer_texts: list[str],
    *,
    min_overlap_ratio: float = 0.15,
) -> str:
    # * Filter "visual-only" lines: keep a line if it overlaps any recognizer text
    # * Overlap = |tokens(line) ∩ tokens(rec)| / |tokens(line)|
    lines = parse_ass_dialogue_texts(ass_file)
    if not lines:
        return ""
    if not recognizer_texts:
        # No recognizer texts – keep all
        return " \n".join(lines)
    # Build token bags for recognizers once
    rec_token_sets = [set(_normalize_text_for_wer(t)) for t in recognizer_texts]
    filtered: list[str] = []
    for line in lines:
        ltoks = _normalize_text_for_wer(line)
        if not ltoks:
            continue
        ltok_set = set(ltoks)
        best = 0.0
        for rset in rec_token_sets:
            inter = len(ltok_set & rset)
            if inter == 0:
                continue
            best = max(best, float(inter) / float(len(ltoks)))
            if best >= min_overlap_ratio:
                break
        if best >= min_overlap_ratio:
            filtered.append(line)
    return " \n".join(filtered)


# * Execution ----------------------------------------------------------------------------------


@dataclass
class FileOutcome:
    media: Path
    transcript: str
    elapsed_s: float


def _make_worker(model_name: str, device: str, compute_type: str) -> WhisperXWrapper:
    # * Test hook: allow fake lightweight worker via env var to avoid heavy model loads
    if os.getenv("ASKVLM_BENCH_FAKE", "0") == "1":
        class _Fake:
            def transcribe(self, wav_path: Path) -> dict:
                # * Return deterministic short text
                time.sleep(0.05)
                return {"text": "hello world"}
        return _Fake()  # type: ignore[return-value]
    return WhisperXWrapper(
        model_name=model_name, device=device, compute_type=compute_type, model_root=None
    )


def _transcribe_one(
    worker: WhisperXWrapper,
    media: Path,
    work_dir: Path,
) -> FileOutcome:
    t0 = time.perf_counter()
    # * If running in fake mode, skip actual audio prep
    if os.getenv("ASKVLM_BENCH_FAKE", "0") == "1":
        res = worker.transcribe(media)
    else:
        wav = prepare_audio(media, work_dir)
        res = worker.transcribe(wav)
    txt = str(res.get("text", "")).strip()
    return FileOutcome(media=media, transcript=txt, elapsed_s=max(0.0, time.perf_counter() - t0))


def run_variant_on_files(
    variant: Variant,
    files: list[Path],
    out_dir: Path,
    *,
    save_transcripts: bool = True,
    gpu_compute: str | None = None,
    cpu_compute: str | None = None,
    variant_label: str | None = None,
    verbose: bool = False,
    silent_progress: bool = False,
) -> tuple[list[FileOutcome], float]:
    # * Scheduler: create a shared queue and launch N workers as specified
    start = time.perf_counter()
    q: "queue.Queue[Path]" = queue.Queue()
    for f in files:
        q.put(f)

    out_dir.mkdir(parents=True, exist_ok=True)
    results_lock = threading.Lock()
    print_lock = threading.Lock()
    wrappers_lock = threading.Lock()
    owned_wrappers: list[WhisperXWrapper] = []
    outcomes: list[FileOutcome] = []
    total = len(files)
    processed_count = 0
    stop_event = threading.Event()

    def _worker_thread(spec: WorkerSpec, idx: int) -> None:
        def _dbg(msg: str) -> None:
            if verbose:
                with print_lock:
                    typer.echo(msg)
                    try:
                        sys.stdout.flush()
                    except Exception:
                        pass
        ct = gpu_compute if spec.device == "cuda" else cpu_compute
        # Fallback to sensible defaults if not provided
        if ct is None:
            ct = "float16" if spec.device == "cuda" else "int8"
        wrapper = _make_worker(spec.model_name, spec.device, ct)
        # * Keep a strong reference in the parent to release on the main thread later
        with wrappers_lock:
            owned_wrappers.append(wrapper)
        while True:
            _dbg(f"  DEBUG: [{spec.device}:{idx}] Attempting to get next item from queue...")
            try:
                media = q.get_nowait()
            except queue.Empty:
                _dbg(f"  DEBUG: [{spec.device}:{idx}] Queue empty, exiting worker")
                break
            _dbg(f"  DEBUG: [{spec.device}:{idx}] Got item: {media.name}")
            try:
                _dbg(f"  -> [{spec.device}:{idx}] Start {media.name}")
                r = _transcribe_one(wrapper, media, out_dir)
                _dbg(f"  DEBUG: [{spec.device}:{idx}] Transcription done, saving...")
                if save_transcripts:
                    (out_dir / f"{media.stem}.txt").write_text(r.transcript, encoding="utf-8")
                with results_lock:
                    outcomes.append(r)
                with results_lock:
                    nonlocal processed_count
                    processed_count += 1
                _dbg(
                    f"  <- [{spec.device}:{idx}] Done {media.name} in {r.elapsed_s:.1f}s  ({processed_count}/{total})"
                )
            finally:
                q.task_done()
        _dbg(f"  DEBUG: [{spec.device}:{idx}] Worker thread returning")

    def _heartbeat() -> None:
        if silent_progress:
            return
        last_line = ""
        last_len = 0
        while not stop_event.wait(5.0):
            with results_lock:
                pc = processed_count
            elapsed = max(0.0, time.perf_counter() - start)
            # ETA estimation (simple average per completed item)
            if pc > 0:
                avg_per_item = elapsed / float(pc)
                remaining = max(0, total - pc)
                eta_s = avg_per_item * float(remaining)
            else:
                eta_s = 0.0
            # Format ETA as hhh:mm:ss instead of minutes-only
            eta_str = _format_hms(eta_s) if eta_s > 0 else "--:--:--"
            label = f"{variant_label} | " if variant_label else ""
            line = f"    ... {label}progress {pc}/{total}, elapsed {elapsed:.1f}s, ETA {eta_str}"
            if line != last_line:
                with print_lock:
                    # Live update on a single console line regardless of verbosity
                    padded = line
                    if len(padded) < last_len:
                        padded = padded + (" " * (last_len - len(padded)))
                    try:
                        sys.stdout.write("\r" + padded)
                        sys.stdout.flush()
                    except Exception:
                        pass
                    last_len = max(last_len, len(line))
            last_line = line

    threads: list[threading.Thread] = []
    # * Use non-daemon threads to ensure clean shutdown and avoid abrupt finalizers
    hb = threading.Thread(target=_heartbeat, daemon=False)
    hb.start()
    for spec in variant.workers:
        for i in range(spec.count):
            t = threading.Thread(target=_worker_thread, args=(spec, i), daemon=False)
            threads.append(t)
            t.start()

    if verbose:
        typer.echo(f"  DEBUG: Joining {len(threads)} worker threads...")
        sys.stdout.flush()
    for idx, t in enumerate(threads, 1):
        if verbose:
            typer.echo(f"  DEBUG: Joining thread {idx}/{len(threads)}...")
            sys.stdout.flush()
        t.join(timeout=3600.0)  # 1 hour timeout
        if t.is_alive():
            if verbose:
                typer.echo(f"  DEBUG: WARNING: Thread {idx} still alive after timeout!")
                sys.stdout.flush()

    if verbose:
        typer.echo(f"  DEBUG: All worker threads joined, stopping heartbeat")
        sys.stdout.flush()
    stop_event.set()
    with contextlib.suppress(Exception):
        hb.join(timeout=1.0)
    # * Release heavy models on the main thread to avoid crashes in worker destructor
    try:
        if verbose:
            typer.echo("  DEBUG: Unloading worker models on main thread...")
            sys.stdout.flush()
        with wrappers_lock:
            for w in owned_wrappers:
                with contextlib.suppress(Exception):
                    # Prefer explicit unload if available
                    unload = getattr(w, "unload", None)
                    if callable(unload):
                        unload(safe=True)
            owned_wrappers.clear()
        with contextlib.suppress(Exception):
            import gc as _gc
            _gc.collect()
    finally:
        elapsed = max(0.0, time.perf_counter() - start)
        # Clear live line to move cursor to fresh line before returning
        if not verbose:
            with print_lock:
                try:
                    sys.stdout.write("\r\n")
                    sys.stdout.flush()
                except Exception:
                    pass
        if verbose:
            typer.echo(f"  DEBUG: Returning from run_variant_on_files, elapsed={elapsed:.1f}s")
            sys.stdout.flush()
        # * Best-effort cleanup of temporary work folder created by prepare_audio
        try:
            work_subdir = out_dir / "_work"
            if work_subdir.exists():
                for p in list(work_subdir.iterdir()):
                    try:
                        if p.is_file():
                            p.unlink()
                        elif p.is_dir():
                            # remove nested files if any
                            for sub in list(p.rglob("*")):
                                try:
                                    if sub.is_file():
                                        sub.unlink()
                                except Exception:
                                    pass
                            with contextlib.suppress(Exception):
                                p.rmdir()
                    except Exception:
                        pass
                with contextlib.suppress(Exception):
                    work_subdir.rmdir()
        except Exception:
            # Best-effort only; ignore failures
            pass
        return outcomes, elapsed


def _sum_duration(files: Iterable[Path]) -> float:
    total = 0.0
    for f in files:
        total += max(0.0, get_media_duration_seconds(f))
    return total


# Helper: format seconds into HHH:MM:SS string (hours zero-padded to 3 digits)
def _format_hms(total_seconds: float | None, hours_pad: int = 3) -> str:
    """
    * Convert seconds to a string "hhh:mm:ss".
    Returns placeholder "--:--:--" for non-positive or None input.
    """
    try:
        if total_seconds is None or total_seconds <= 0:
            return "--:--:--"
        secs = int(round(total_seconds))
        hours = secs // 3600
        minutes = (secs % 3600) // 60
        seconds = secs % 60
        return f"{hours:0{hours_pad}d}:{minutes:02d}:{seconds:02d}"
    except Exception:
        return "--:--:--"


def _collect_all_texts(outcomes: list[FileOutcome]) -> list[str]:
    return [o.transcript for o in outcomes if o.transcript]


def _concat_texts(outcomes: list[FileOutcome]) -> str:
    return "\n".join(_collect_all_texts(outcomes))


def _signal_system_sound() -> None:
    """Emit a short system sound to signal a phase boundary."""
    try:
        if sys.platform.startswith("win"):
            import winsound as _winsound

            _winsound.MessageBeep(_winsound.MB_ICONASTERISK)
        else:
            # ASCII bell as a cross-platform fallback
            print("\a", end="", flush=True)
    except Exception:  # noqa: BLE001
        # Best-effort signal; ignore failures
        pass


def _cleanup_work_subdir(out_dir: Path) -> None:
    """Best-effort remove the hidden `_work` subdirectory under `out_dir`."""
    try:
        work_subdir = out_dir / "_work"
        if not work_subdir.exists():
            return
        for p in list(work_subdir.iterdir()):
            try:
                if p.is_file():
                    p.unlink()
                elif p.is_dir():
                    for sub in list(p.rglob("*")):
                        try:
                            if sub.is_file():
                                sub.unlink()
                        except Exception:
                            pass
                    with contextlib.suppress(Exception):
                        p.rmdir()
            except Exception:
                pass
        with contextlib.suppress(Exception):
            work_subdir.rmdir()
    except Exception:
        # Best-effort; ignore failures
        pass


def _collect_outcomes_from_dir(out_dir: Path, originals: list[Path]) -> list[FileOutcome]:
    outcomes: list[FileOutcome] = []
    for f in originals:
        t = out_dir / f"{f.stem}.txt"
        try:
            txt = t.read_text(encoding="utf-8") if t.exists() else ""
        except OSError:
            txt = ""
        outcomes.append(FileOutcome(media=f, transcript=txt, elapsed_s=0.0))
    return outcomes


def _duration_map(files: list[Path]) -> dict[str, float]:
    out: dict[str, float] = {}
    for f in files:
        try:
            out[f.stem] = max(0.0, get_media_duration_seconds(f))
        except Exception:  # noqa: BLE001
            out[f.stem] = 0.0
    return out


def _run_phase_subprocess(
    *,
    dataset: Literal["uv", "ver"],
    variant_base: str,
    gpu_compute: str | None,
    cpu_compute: str | None,
    filelist_path: Path,
    out_dir: Path,
    variant_idx: int | None = None,
    variants_total: int | None = None,
    verbose: bool = False,
    durations_by_stem: dict[str, float] | None = None,
    phase_total_work_s: float | None = None,
    global_total_work_s: float | None = None,
    global_base_work_s: float | None = None,
    global_rate_hint_wps: float | None = None,
) -> float:
    args = [
        sys.executable,
        str(Path(__file__).resolve()),
        "phase",
        "--dataset",
        dataset,
        "--variant-base",
        variant_base,
        "--filelist",
        str(filelist_path),
        "--out",
        str(out_dir),
    ]
    if gpu_compute:
        args += ["--gpu-compute", gpu_compute]
    if cpu_compute:
        args += ["--cpu-compute", cpu_compute]
    if variant_idx is not None and variants_total is not None:
        args += ["--variant-idx", str(variant_idx), "--variants-total", str(variants_total)]
    # Do not forward verbose to child to avoid interleaving output with parent live line
    # Also ensure the child disables its own progress output
    args += ["--silent-progress"]
    # Stream output live and render a parent-side live progress line
    try:
        files_list = _json.loads(filelist_path.read_text(encoding="utf-8"))
        total_files = len(files_list)
    except Exception:
        total_files = 0

    label = (
        f"Var {variant_idx}/{variants_total}: {variant_base} ({dataset})"
        if (variant_idx is not None and variants_total is not None)
        else f"{variant_base} ({dataset})"
    )

    processed = 0
    processed_work_s = 0.0
    last_len = 0
    start_ts = time.perf_counter()
    stop_event = threading.Event()

    # * ETA smoothing state (periodic recompute + countdown between recalculations)
    last_recalc_ts = start_ts
    last_recalc_eta_phase_s: float | None = None
    last_recalc_eta_global_s: float | None = None
    last_proc_snapshot = 0
    last_work_snapshot = 0.0
    recalc_period_s = 3.0

    def _eta_pair() -> tuple[str, str]:
        nonlocal last_recalc_ts, last_recalc_eta_phase_s, last_recalc_eta_global_s, last_proc_snapshot, last_work_snapshot
        now = time.perf_counter()
        elapsed = max(0.0, now - start_ts)

        # Decide whether to recompute raw ETAs
        progressed = (processed != last_proc_snapshot) or (abs(processed_work_s - last_work_snapshot) > 1e-6)
        need_recalc = progressed or ((now - last_recalc_ts) >= recalc_period_s)

        eta_phase_s: float | None = None
        eta_global_s: float | None = None

        if need_recalc:
            # Phase ETA by durations or counts
            if durations_by_stem and phase_total_work_s and processed_work_s > 0.0:
                phase_rem = max(0.0, phase_total_work_s - processed_work_s)
                local_rate = processed_work_s / max(1e-6, elapsed)
                eta_phase_s = phase_rem / max(1e-6, local_rate)
            elif processed > 0:
                rem = max(0, total_files - processed)
                avg = elapsed / float(processed)
                eta_phase_s = float(rem) * avg

            # Global ETA with hint-aware blended rate when durations are present
            if (
                durations_by_stem is not None
                and global_total_work_s is not None
                and global_base_work_s is not None
            ):
                global_done = (global_base_work_s or 0.0) + (processed_work_s or 0.0)
                global_rem = max(0.0, (global_total_work_s or 0.0) - global_done)
                local_rate = (processed_work_s / max(1e-6, elapsed)) if processed_work_s > 0.0 else 0.0
                rate = local_rate
                if (global_rate_hint_wps or 0.0) > 0.0:
                    # Weight grows with phase progress; near start use historical hint more
                    if phase_total_work_s and phase_total_work_s > 0.0:
                        phase_progress = max(0.0, min(1.0, processed_work_s / max(1e-6, phase_total_work_s)))
                    elif total_files > 0:
                        phase_progress = max(0.0, min(1.0, processed / float(total_files)))
                    else:
                        phase_progress = 0.0
                    alpha = phase_progress
                    rate = (alpha * local_rate) + ((1.0 - alpha) * float(global_rate_hint_wps))
                if rate > 0.0:
                    eta_global_s = global_rem / rate

            elif (
                global_total_work_s is not None
                and global_base_work_s is not None
                and processed > 0
                and total_files > 0
            ):
                # Treat each file as equal work unit when durations are missing
                frac = processed / float(total_files)
                global_done_units = (global_base_work_s or 0.0) + frac
                global_rem_units = max(0.0, (global_total_work_s or 0.0) - global_done_units)
                avg = elapsed / float(processed)
                eta_global_s = avg * global_rem_units

            # Commit recalculation checkpoint
            last_recalc_ts = now
            last_proc_snapshot = processed
            last_work_snapshot = processed_work_s
            if eta_phase_s is not None:
                last_recalc_eta_phase_s = eta_phase_s
            if eta_global_s is not None:
                last_recalc_eta_global_s = eta_global_s
        else:
            # Countdown between recalculations
            dt = max(0.0, now - last_recalc_ts)
            if last_recalc_eta_phase_s is not None:
                eta_phase_s = max(0.0, last_recalc_eta_phase_s - dt)
            if last_recalc_eta_global_s is not None:
                eta_global_s = max(0.0, last_recalc_eta_global_s - dt)

        # Fallbacks
        eta_phase = _format_hms(eta_phase_s) if (eta_phase_s is not None) else "--:--:--"
        eta_global = _format_hms(eta_global_s) if (eta_global_s is not None) else "--:--:--"
        return eta_phase, eta_global

    def _render_progress() -> None:
        nonlocal last_len
        elapsed = max(0.0, time.perf_counter() - start_ts)
        eta_phase, eta_global = _eta_pair()
        line = (
            f"    ... {label} | progress {processed}/{total_files}, "
            f"elapsed {elapsed:.1f}s, ETA phase {eta_phase} | total {eta_global}"
        )
        padded = line
        if len(padded) < last_len:
            padded = padded + (" " * (last_len - len(padded)))
        try:
            sys.stdout.write("\r" + padded)
            sys.stdout.flush()
        except Exception:
            pass
        last_len = max(last_len, len(line))

    def _heartbeat() -> None:
        nonlocal processed, processed_work_s
        # Parent-side heartbeat: derive progress by counting created transcript files
        target_stems: set[str] = set()
        try:
            files_list = _json.loads(filelist_path.read_text(encoding="utf-8"))
            target_stems = {Path(p).stem for p in files_list}
        except Exception:
            target_stems = set()
        while not stop_event.wait(0.5):
            try:
                present = 0
                present_stems: list[str] = []
                if target_stems:
                    for stem in target_stems:
                        if (out_dir / f"{stem}.txt").exists():
                            present += 1
                            present_stems.append(stem)
                    processed = present
                    if durations_by_stem is not None:
                        processed_work_s = sum(durations_by_stem.get(s, 0.0) for s in present_stems)
            except Exception:
                pass
            _render_progress()

    hb = threading.Thread(target=_heartbeat, daemon=False)
    hb.start()

    t0 = time.perf_counter()
    proc = None
    captured: list[str] = []
    reader_thread: threading.Thread | None = None
    try:
        # * Provide a safer environment to reduce runtime conflicts on Windows/OpenMP/CT2
        _env = os.environ.copy()
        _env.setdefault("PYTHONFAULTHANDLER", "1")
        _env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        _env.setdefault("TOKENIZERS_PARALLELISM", "false")
        _env.setdefault("OMP_NUM_THREADS", _env.get("OMP_NUM_THREADS", "1"))
        proc = _subprocess.Popen(
            args,
            stdout=_subprocess.PIPE,
            stderr=_subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=_env,
        )
        # Non-blocking reader in a separate thread so main thread can handle Ctrl+C
        def _read_stdout() -> None:
            try:
                assert proc is not None and proc.stdout is not None
                for raw in proc.stdout:
                    captured.append(raw)
            except Exception:
                pass
        reader_thread = threading.Thread(target=_read_stdout, daemon=False)
        reader_thread.start()
        rc: int | None = None
        while True:
            try:
                rc = proc.poll()
                if rc is not None:
                    break
                # Allow KeyboardInterrupt to be raised between polls
                if stop_event.wait(0.2):
                    continue
            except KeyboardInterrupt:
                # * Gracefully terminate subprocess on Ctrl+C
                if proc is not None:
                    with contextlib.suppress(Exception):
                        proc.terminate()
                        proc.wait(timeout=5.0)
                raise
    except KeyboardInterrupt:
        # * Gracefully terminate subprocess on Ctrl+C
        if proc is not None:
            with contextlib.suppress(Exception):
                proc.terminate()
                proc.wait(timeout=5.0)
        stop_event.set()
        with contextlib.suppress(Exception):
            hb.join(timeout=1.0)
        raise
    except FileNotFoundError as ex:
        stop_event.set()
        with contextlib.suppress(Exception):
            hb.join(timeout=1.0)
        raise RuntimeError(f"phase subprocess failed to start: {ex}") from ex
    elapsed = max(0.0, time.perf_counter() - t0)
    stop_event.set()
    with contextlib.suppress(Exception):
        hb.join(timeout=1.0)
    # ensure reader finishes
    if reader_thread is not None:
        with contextlib.suppress(Exception):
            reader_thread.join(timeout=1.0)
    # finalize live line
    try:
        sys.stdout.write("\r" + (" " * last_len) + "\r")
        sys.stdout.flush()
    except Exception:
        pass
    # * Always inspect sentinel; if it indicates outputs incomplete, treat as failure
    try:
        sentinel = out_dir / ".phase_ok.json"
        if sentinel.exists():
            try:
                info = _json.loads(sentinel.read_text(encoding="utf-8"))
            except Exception:
                info = None
            if isinstance(info, dict):
                if "elapsed_s" in info:
                    try:
                        elapsed = float(info.get("elapsed_s", elapsed))
                    except Exception:
                        pass
                if info.get("ok") is False:
                    # Force failure path: outputs incomplete
                    rc = rc if rc is not None else 1
    except Exception:
        pass

    if rc != 0:
        tail = "".join(captured[-50:])
        # * If all expected transcript files are present and non-empty, consider the phase successful
        try:
            files_list = _json.loads(filelist_path.read_text(encoding="utf-8"))
            target_stems = {Path(p).stem for p in files_list}
        except Exception:
            target_stems = set()
        if target_stems:
            try:
                def _nonempty(p: Path) -> bool:
                    try:
                        if not p.exists():
                            return False
                        # Treat whitespace-only as empty
                        txt = p.read_text(encoding="utf-8", errors="ignore")
                        return bool(txt.strip())
                    except Exception:
                        return False
                all_present = all(_nonempty(out_dir / f"{s}.txt") for s in target_stems)
            except Exception:
                all_present = False
            if all_present:
                # Try to parse child's JSON from captured output as a better elapsed, fallback to local elapsed
                try:
                    for line in reversed(captured[-100:]):
                        line_s = line.strip()
                        if line_s.startswith("{") and ("elapsed_s" in line_s):
                            data = _json.loads(line_s)
                            if isinstance(data, dict) and "elapsed_s" in data:
                                elapsed = float(data.get("elapsed_s", elapsed))
                                break
                except Exception:
                    pass
                return elapsed

        # * Otherwise, classify and raise with helpful diagnostics
        # * Decode Windows error codes for better diagnostics
        if rc == 3221226505 or rc == -1073740791:  # 0xC0000409 STATUS_STACK_BUFFER_OVERRUN
            raise RuntimeError(
                f"phase subprocess crashed (STATUS_STACK_BUFFER_OVERRUN, code={rc})\n"
                "This typically indicates a memory corruption or incompatible library.\n"
                f"Last output:\n{tail}"
            )
        if rc == 3221225477 or rc == -1073741819:  # 0xC0000005 STATUS_ACCESS_VIOLATION
            raise RuntimeError(
                f"phase subprocess crashed (STATUS_ACCESS_VIOLATION, code={rc})\n"
                "This typically indicates an invalid memory access in a dependency.\n"
                f"Last output:\n{tail}"
            )
        raise RuntimeError(f"phase subprocess failed (code={rc})\n{tail}")
    return elapsed


def benchmark(
    unverified_dir: Path,
    verified_dir: Path,
    output_root: Path,
    *,
    include_variants: list[str] | None = None,
    max_unverified: int = -1,
    max_verified: int = -1,
    mode: Literal["full", "cpu", "gpu"] = "full",
    resume_clear: bool = False,
    verbose: bool = False,
) -> None:
    # * Resolve datasets
    uv_files = _list_media_files(unverified_dir, {".mp4", ".mkv", ".mov", ".webm"})
    if max_unverified >= 0:
        uv_files = uv_files[:max_unverified]
    ver_audio = _list_media_files(verified_dir, {".mka", ".wav", ".mp3", ".flac"})
    ver_subs = _list_media_files(verified_dir, {".ass"})
    ass_map: dict[str, Path] = {p.stem: p for p in ver_subs}
    ver_pairs: list[tuple[Path, Path]] = []
    for a in ver_audio:
        s = ass_map.get(a.stem)
        if s is not None:
            ver_pairs.append((a, s))
    if max_verified >= 0:
        ver_pairs = ver_pairs[:max_verified]

    # * Out directories
    output_root.mkdir(parents=True, exist_ok=True)
    report_rows: list[dict[str, object]] = []
    resume_dir = output_root / ".resume"
    if resume_clear and resume_dir.exists():
        for p in resume_dir.glob("*"):
            with contextlib.suppress(Exception):
                p.unlink()
        with contextlib.suppress(Exception):
            resume_dir.rmdir()
    resume_dir.mkdir(parents=True, exist_ok=True)

    reports_dir = output_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "benchmark_summary.json"
    # * Load previous results to allow resume aggregation
    old_rows: list[dict[str, object]] = []
    if not resume_clear and json_path.exists():
        try:
            old_rows = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            old_rows = []

    # * Prepare variant list (print early so user sees work ahead)
    # * Compute-type parameterization
    CPU_CT = ["int8", "int8_float16"]
    GPU_CT = ["float32", "float16", "int8"]

    def _parametrize_variant(var: Variant) -> list[tuple[str, Variant, str | None, str | None]]:
        has_gpu = any(w.device == "cuda" for w in var.workers)
        has_cpu = any(w.device == "cpu" for w in var.workers)
        out: list[tuple[str, Variant, str | None, str | None]] = []
        if has_gpu and has_cpu:
            for g in GPU_CT:
                for c in CPU_CT:
                    name = f"{var.name} [GPU {g} / CPU {c}]"
                    out.append((name, var, g, c))
        elif has_gpu:
            for g in GPU_CT:
                name = f"{var.name} [GPU {g}]"
                out.append((name, var, g, None))
        else:
            for c in CPU_CT:
                name = f"{var.name} [CPU {c}]"
                out.append((name, var, None, c))
        return out

    def _include_this(name: str, base_name: str) -> bool:
        if not include_variants:
            return True
        return (name in include_variants) or (base_name in include_variants)

    # * Filter base variants by mode
    def _all_dev(v: Variant, dev: str) -> bool:
        return all(w.device == dev for w in v.workers)

    if mode == "cpu":
        selected_variants = [v for v in VARIANTS if _all_dev(v, "cpu")]
    elif mode == "gpu":
        selected_variants = [v for v in VARIANTS if _all_dev(v, "cuda")]
    else:
        selected_variants = list(VARIANTS)
    param_variants: list[tuple[str, Variant, str | None, str | None]] = []
    for var in selected_variants:
        for name, v2, gct, cct in _parametrize_variant(var):
            if _include_this(name, v2.name):
                param_variants.append((name, v2, gct, cct))

    def _key(name: str) -> str:
        return (
            name.replace(" ", "_")
            .replace("[", "")
            .replace("]", "")
            .replace("|", "_")
            .replace("+", "_")
            .replace(",", "_")
            .replace("/", "_")
            .replace("\\", "_")
        )

    # * Prefer faster variants earlier (small > large, GPU > CPU, fp16 > int8 > fp32)
    def _speed_rank(v: Variant, gct: str | None, cct: str | None) -> tuple[int, int, int, int]:
        # Model rank: small(0) < large(1)
        m = 0
        for w in v.workers:
            m = max(m, 0 if w.model_name == "small" else 1)
        # Device rank: gpu(0) < cpu(1)
        d = 0
        for w in v.workers:
            d = max(d, 0 if w.device == "cuda" else 1)
        # GPU compute rank
        gmap = {None: 2, "float16": 0, "int8": 1, "float32": 3}
        # CPU compute rank
        cmap = {None: 1, "int8": 0, "int8_float16": 1}
        return (m, d, gmap.get(gct, 2), cmap.get(cct, 1))

    param_variants.sort(key=lambda t: _speed_rank(t[1], t[2], t[3]))

    typer.echo(
        f"Planned: {len(param_variants)} parametrized variants | uv_files={len(uv_files)} | ver_pairs={len(ver_pairs)}"
    )

    # * Unverified baseline (large-v3 GPU float16) for WER reference
    baseline_variant = VARIANTS[0]
    base_name = baseline_variant.name
    if include_variants and base_name not in include_variants:
        pass
    uv_ref_by_stem: dict[str, str] = {}
    if uv_files:
        ref_dir = output_root / "unverified" / "ref_largev3_gpu"
        if ref_dir.exists():
            for f in uv_files:
                t = ref_dir / f"{f.stem}.txt"
                if t.exists():
                    with contextlib.suppress(Exception):
                        uv_ref_by_stem[f.stem] = t.read_text(encoding="utf-8")
        if len(uv_ref_by_stem) != len(uv_files):
            typer.echo(
                f"Baseline: 1x large-v3 GPU [float16] on unverified ({len(uv_files)} files)"
            )
            uv_baseline_out, _ = run_variant_on_files(
                baseline_variant,
                uv_files,
                ref_dir,
                save_transcripts=True,
                gpu_compute="float16",
                variant_label="Baseline (large-v3 GPU float16)",
            )
            uv_ref_by_stem = {o.media.stem: o.transcript for o in uv_baseline_out}
        else:
            typer.echo("Baseline: transcripts already present, skipping")

    # * Collect all variant results
    variant_results: dict[str, dict[str, object]] = {}

    def _sanitize_name(s: str) -> str:
        return (
            s.replace(" ", "_")
            .replace("[", "")
            .replace("]", "")
            .replace("|", "_")
            .replace("+", "_")
            .replace(",", "_")
            .replace("/", "_")
            .replace("\\", "_")
        )

    # * Resume filter
    def _key(name: str) -> str:
        return (
            name.replace(" ", "_")
            .replace("[", "")
            .replace("]", "")
            .replace("|", "_")
            .replace("+", "_")
            .replace(",", "_")
            .replace("/", "_")
            .replace("\\", "_")
        )

    interrupted: list[tuple[str, Variant, str | None, str | None]] = []
    pending: list[tuple[str, Variant, str | None, str | None]] = []
    for var_name, var, gct, cct in param_variants:
        k = _key(var_name)
        done_f = resume_dir / f"{k}.done"
        started_f = resume_dir / f"{k}.started"
        if done_f.exists():
            if verbose:
                typer.echo(f"SKIP done: {var_name}")
            continue
        if started_f.exists():
            if verbose:
                typer.echo(f"RESUME interrupted: {var_name}")
            interrupted.append((var_name, var, gct, cct))
        else:
            pending.append((var_name, var, gct, cct))

    to_run: list[tuple[str, Variant, str | None, str | None]] = interrupted + pending
    typer.echo(
        f"Variants to run: {len(to_run)}/{len(param_variants)} (resume-first: {len(interrupted)})"
    )

    total_variants = len(to_run)
    # * Build per-file duration maps for ETA
    uv_durs: dict[str, float] = _duration_map(uv_files) if uv_files else {}
    ver_files_only = [p for (p, _s) in ver_pairs]
    ver_durs: dict[str, float] = _duration_map(ver_files_only) if ver_files_only else {}
    uv_phase_work = sum(uv_durs.values()) if uv_durs else float(len(uv_files))
    ver_phase_work = sum(ver_durs.values()) if ver_durs else float(len(ver_files_only))
    global_total_work = float(total_variants) * (uv_phase_work + ver_phase_work)
    for idx_v, (var_name, var, gct, cct) in enumerate(to_run, start=1):
        typer.echo(
            f"Run [{idx_v}/{total_variants}]: {var_name} | GPU={gct or '-'} | CPU={cct or '-'} | uv={len(uv_files)} | ver={len(ver_pairs)}"
        )
        # Prepare resume markers early so we can safely mark DONE even if errors occur
        k = _key(var_name)
        started_f = resume_dir / f"{k}.started"
        done_f = resume_dir / f"{k}.done"
        with contextlib.suppress(Exception):
            done_f.unlink()
        with contextlib.suppress(Exception):
            started_f.write_text("", encoding="utf-8")
        variant_status = "ok"
        skip_rest = False
        # Track if a phase failed and capture its error message for richer status reporting
        failed_phase: str | None = None
        failed_ex_msg: str = ""

        # Unverified
        if verbose:
            typer.echo(f"  DEBUG: Starting UV processing, {len(uv_files)} files")
            sys.stdout.flush()
        # * Helper: classify child failure by captured text to avoid mislabeling as incompatible
        def _classify_failure(msg: str) -> tuple[str, str]:
            m = msg.lower()
            if ("out of memory" in m) or ("cuda failed with error out of memory" in m) or ("cuda out of memory" in m):
                return ("skipped-oom", "Out of memory detected; marking as skipped-oom.")
            if ("requested int8_float16 compute type" in m) or ("not support efficient int8_float16" in m):
                return ("unsupported-compute", "Unsupported compute type (int8_float16); marking as unsupported-compute.")
            if (
                ("status_stack_buffer_overrun" in m)
                or ("status_access_violation" in m)
                or ("code=3221226505" in m)
                or ("code=3221225477" in m)
            ):
                return ("skipped-incompatible", "Incompatible runtime crash signature; marking as skipped-incompatible.")
            return ("failed", "Phase failed; marking as failed.")

        try:
            if uv_files:
                uv_out_dir = output_root / "unverified" / _sanitize_name(var_name)
                uv_out_dir.mkdir(parents=True, exist_ok=True)
                tmp_list = resume_dir / f"{_key(var_name)}.uv.files.json"
                tmp_list.write_text(_json.dumps([str(p) for p in uv_files]), encoding="utf-8")
                # * Historical global rate hint from completed variants to stabilize total ETA
                try:
                    ok_bundles = [b for b in variant_results.values() if str(b.get("status", "ok")) == "ok"]  # type: ignore[union-attr]
                except Exception:
                    ok_bundles = []
                hist_elapsed_done_s = 0.0
                for _b in ok_bundles:
                    try:
                        hist_elapsed_done_s += float(_b["unverified"]["elapsed_s"])  # type: ignore[index]
                        hist_elapsed_done_s += float(_b["verified"]["elapsed_s"])  # type: ignore[index]
                    except Exception:
                        pass
                hist_work_done_s = float(len(ok_bundles)) * (uv_phase_work + ver_phase_work)
                global_rate_hint_wps = (hist_work_done_s / hist_elapsed_done_s) if hist_elapsed_done_s > 0.0 else None

                uv_elapsed = _run_phase_subprocess(
                    dataset="uv",
                    variant_base=var.name,
                    gpu_compute=gct,
                    cpu_compute=cct,
                    filelist_path=tmp_list,
                    out_dir=uv_out_dir,
                    variant_idx=idx_v,
                    variants_total=total_variants,
                    verbose=verbose,
                    durations_by_stem=uv_durs or None,
                    phase_total_work_s=uv_phase_work,
                    global_total_work_s=global_total_work,
                    global_base_work_s=(idx_v - 1) * (uv_phase_work + ver_phase_work),
                    global_rate_hint_wps=global_rate_hint_wps,
                )
                uv_out = _collect_outcomes_from_dir(uv_out_dir, uv_files)
            else:
                uv_out, uv_elapsed = ([], 0.0)
            if verbose:
                typer.echo(f"  DEBUG: UV phase completed via subprocess")
                sys.stdout.flush()
        except Exception as ex:  # noqa: BLE001
            typer.echo(f"ERROR uv: {var_name}: {ex}")
            exs = str(ex)
            failed_phase = "uv"
            failed_ex_msg = exs
            status, note = _classify_failure(exs)
            typer.secho(f"  WARNING: {note}", fg="yellow")
            variant_status = status
            skip_rest = True
            import traceback
            traceback.print_exc()
            uv_out, uv_elapsed = ([], 0.0)
            # Best-effort cleanup of child temporary work folder for this UV phase
            try:
                with contextlib.suppress(Exception):
                    _cleanup_work_subdir(uv_out_dir)
            except Exception:
                pass
        if verbose:
            typer.echo(f"  DEBUG: UV done, outcomes={len(uv_out)}")
            sys.stdout.flush()
        uv_dur = _sum_duration(uv_files)
        uv_rtf = (uv_dur / uv_elapsed) if uv_elapsed > 0 else 0.0

        # Verified
        if skip_rest:
            if verbose:
                typer.echo("  DEBUG: Skipping VER due to incompatible configuration detected in UV")
                sys.stdout.flush()
            ver_out, ver_elapsed = ([], 0.0)
        else:
            if verbose:
                typer.echo(f"  DEBUG: Starting VER processing, {len(ver_files_only)} files")
                sys.stdout.flush()
            try:
                if ver_pairs:
                    ver_out_dir = output_root / "verified" / _sanitize_name(var_name)
                    ver_out_dir.mkdir(parents=True, exist_ok=True)
                    tmp_list = resume_dir / f"{_key(var_name)}.ver.files.json"
                    tmp_list.write_text(_json.dumps([str(p) for p in ver_files_only]), encoding="utf-8")
                    # * Historical global rate hint including current UV to stabilize total ETA in VER
                    try:
                        ok_bundles2 = [b for b in variant_results.values() if str(b.get("status", "ok")) == "ok"]  # type: ignore[union-attr]
                    except Exception:
                        ok_bundles2 = []
                    hist_elapsed_done_s2 = 0.0
                    for _b in ok_bundles2:
                        try:
                            hist_elapsed_done_s2 += float(_b["unverified"]["elapsed_s"])  # type: ignore[index]
                            hist_elapsed_done_s2 += float(_b["verified"]["elapsed_s"])  # type: ignore[index]
                        except Exception:
                            pass
                    hist_elapsed_done_s2 += float(uv_elapsed)
                    hist_work_done_s2 = float(len(ok_bundles2)) * (uv_phase_work + ver_phase_work) + uv_phase_work
                    global_rate_hint_wps2 = (hist_work_done_s2 / hist_elapsed_done_s2) if hist_elapsed_done_s2 > 0.0 else None

                    ver_elapsed = _run_phase_subprocess(
                        dataset="ver",
                        variant_base=var.name,
                        gpu_compute=gct,
                        cpu_compute=cct,
                        filelist_path=tmp_list,
                        out_dir=ver_out_dir,
                        variant_idx=idx_v,
                        variants_total=total_variants,
                        verbose=verbose,
                        durations_by_stem=ver_durs or None,
                        phase_total_work_s=ver_phase_work,
                        global_total_work_s=global_total_work,
                        global_base_work_s=(idx_v - 1) * (uv_phase_work + ver_phase_work) + uv_phase_work,
                        global_rate_hint_wps=global_rate_hint_wps2,
                    )
                    ver_out = _collect_outcomes_from_dir(ver_out_dir, ver_files_only)
                else:
                    ver_out, ver_elapsed = ([], 0.0)
                if verbose:
                    typer.echo(f"  DEBUG: VER phase completed via subprocess")
                    sys.stdout.flush()
            except Exception as ex:  # noqa: BLE001
                typer.echo(f"ERROR ver: {var_name}: {ex}")
                exs = str(ex)
                failed_phase = "ver"
                failed_ex_msg = exs
                status, note = _classify_failure(exs)
                typer.secho(f"  WARNING: {note}", fg="yellow")
                variant_status = status
                import traceback
                traceback.print_exc()
                ver_out, ver_elapsed = ([], 0.0)
                # Best-effort cleanup of child temporary work folder for this VER phase
                try:
                    with contextlib.suppress(Exception):
                        _cleanup_work_subdir(ver_out_dir)
                except Exception:
                    pass
        if variant_status == "skipped-incompatible":
            # Full variant skip: do not use partial UV results if any
            uv_out, uv_elapsed, uv_dur, uv_rtf = ([], 0.0, 0.0, 0.0)
            ver_out, ver_elapsed = ([], 0.0)
            if verbose:
                typer.echo("  DEBUG: Variant marked as skipped-incompatible; zeroing metrics")
                sys.stdout.flush()
        if verbose:
            typer.echo(f"  DEBUG: VER done, outcomes={len(ver_out)}")
            sys.stdout.flush()
        ver_dur = _sum_duration(ver_files_only)
        ver_rtf = (ver_dur / ver_elapsed) if ver_elapsed > 0 else 0.0

        if verbose:
            typer.echo(f"  DEBUG: Storing results for {var_name}")

        # Enhance non-ok statuses with context: initialization vs per-file failure index+duration
        try:
            if variant_status != "ok" and not str(variant_status).upper().startswith("ERR"):
                reason = (failed_ex_msg.splitlines()[0] if failed_ex_msg else "unknown").strip()
                short_reason = reason[:200]
                if failed_phase == "uv":
                    try:
                        stems = [p.stem for p in uv_files]
                    except Exception:
                        stems = []
                    present: set[str] = set()
                    try:
                        uod = uv_out_dir
                        if uod.exists():
                            for s in stems:
                                p = uod / f"{s}.txt"
                                try:
                                    if p.exists() and p.read_text(encoding="utf-8", errors="ignore").strip():
                                        present.add(s)
                                except Exception:
                                    pass
                    except Exception:
                        pass
                    if not present:
                        variant_status = f'ERR INIT [{short_reason}]'
                    else:
                        # find first missing/empty transcript
                        first_missing_idx = None
                        first_missing_stem = None
                        for idxi, s in enumerate(stems, start=1):
                            if s not in present:
                                first_missing_idx = idxi
                                first_missing_stem = s
                                break
                        if first_missing_idx is None:
                            variant_status = f'ERR [{short_reason}]'
                        else:
                            dur = uv_durs.get(first_missing_stem, 0.0)
                            secs = int(round(dur))
                            h = secs // 3600
                            m = (secs % 3600) // 60
                            sec = secs % 60
                            dur_str = f"{h}:{m:02d}:{sec:02d}"
                            variant_status = f'ERR [{short_reason}] #{first_missing_idx} [{dur_str}]'
                elif failed_phase == "ver":
                    try:
                        stems = [p.stem for p in ver_files_only]
                    except Exception:
                        stems = []
                    present = set()
                    try:
                        vod = ver_out_dir
                        if vod.exists():
                            for s in stems:
                                p = vod / f"{s}.txt"
                                try:
                                    if p.exists() and p.read_text(encoding="utf-8", errors="ignore").strip():
                                        present.add(s)
                                except Exception:
                                    pass
                    except Exception:
                        pass
                    if not present:
                        variant_status = f'ERR INIT [{short_reason}]'
                    else:
                        first_missing_idx = None
                        first_missing_stem = None
                        for idxi, s in enumerate(stems, start=1):
                            if s not in present:
                                first_missing_idx = idxi
                                first_missing_stem = s
                                break
                        if first_missing_idx is None:
                            variant_status = f'ERR [{short_reason}]'
                        else:
                            dur = ver_durs.get(first_missing_stem, 0.0)
                            secs = int(round(dur))
                            h = secs // 3600
                            m = (secs % 3600) // 60
                            sec = secs % 60
                            dur_str = f"{h}:{m:02d}:{sec:02d}"
                            variant_status = f'ERR [{short_reason}] #{first_missing_idx} [{dur_str}]'
        except Exception:
            # If anything goes wrong enriching status, keep original variant_status
            pass

        variant_results[var_name] = {
            "unverified": {
                "outcomes": uv_out,
                "elapsed_s": uv_elapsed,
                "duration_s": uv_dur,
                "rtf": uv_rtf,
            },
            "verified": {
                "outcomes": ver_out,
                "elapsed_s": ver_elapsed,
                "duration_s": ver_dur,
                "rtf": ver_rtf,
            },
            "status": variant_status,
        }
        typer.echo(f"Variant completed: {var_name}")
        # Mark DONE
        with contextlib.suppress(Exception):
            started_f.unlink()
        with contextlib.suppress(Exception):
            done_f.write_text("", encoding="utf-8")

        # * Incremental report write: compute union across processed variants and write combined report
        if verbose:
            typer.echo(f"  DEBUG: Starting incremental report for {len(variant_results)} variant(s)")
        try:
            recognizer_union_texts = []
            for _vn, _bundle in variant_results.items():
                _ver_outcomes = _bundle["verified"]["outcomes"]  # type: ignore[index]
                recognizer_union_texts.extend(_collect_all_texts(_ver_outcomes))

            partial_rows: list[dict[str, object]] = []
            for _vn, _bundle in variant_results.items():
                _uv_outcomes: list[FileOutcome] = _bundle["unverified"]["outcomes"]  # type: ignore[index]
                _uv_elapsed = float(_bundle["unverified"]["elapsed_s"])  # type: ignore[index]
                _uv_dur = float(_bundle["unverified"]["duration_s"])  # type: ignore[index]
                _uv_rtf = float(_bundle["unverified"]["rtf"])  # type: ignore[index]
                _uv_ed_sum = 0
                _uv_ref_tok_sum = 0
                for o in _uv_outcomes:
                    ref = uv_ref_by_stem.get(o.media.stem, "")
                    _w, ed, n = wer_score(ref, o.transcript)
                    _uv_ed_sum += ed
                    _uv_ref_tok_sum += n
                _uv_wer = (float(_uv_ed_sum) / float(_uv_ref_tok_sum)) if _uv_ref_tok_sum > 0 else 0.0

                _ver_outcomes: list[FileOutcome] = _bundle["verified"]["outcomes"]  # type: ignore[index]
                _ver_elapsed = float(_bundle["verified"]["elapsed_s"])  # type: ignore[index]
                _ver_dur = float(_bundle["verified"]["duration_s"])  # type: ignore[index]
                _ver_rtf = float(_bundle["verified"]["rtf"])  # type: ignore[index]
                _ver_ed_sum = 0
                _ver_ref_tok_sum = 0
                _out_by_stem = {o.media.stem: o.transcript for o in _ver_outcomes}
                for a, s in ver_pairs:
                    hyp = _out_by_stem.get(a.stem, "")
                    ref_filtered = ass_reference_text_with_filter(s, recognizer_union_texts)
                    _w2, ed2, n2 = wer_score(ref_filtered, hyp)
                    _ver_ed_sum += ed2
                    _ver_ref_tok_sum += n2
                _ver_wer = (float(_ver_ed_sum) / float(_ver_ref_tok_sum)) if _ver_ref_tok_sum > 0 else 0.0

                partial_rows.append(
                    {
                        "variant": _vn,
                        "status": str(_bundle.get("status", "ok")),
                        "uv_elapsed_s": round(_uv_elapsed, 3),
                        "uv_duration_s": round(_uv_dur, 3),
                        "uv_rtf": round(_uv_rtf, 3),
                        "uv_wer": round(_uv_wer, 4),
                        "ver_elapsed_s": round(_ver_elapsed, 3),
                        "ver_duration_s": round(_ver_dur, 3),
                        "ver_rtf": round(_ver_rtf, 3),
                        "ver_wer": round(_ver_wer, 4),
                    }
                )

            # Merge old rows for variants not processed in this run (resume across sessions)
            if old_rows:
                existing = {r["variant"] for r in partial_rows}
                for r in old_rows:
                    if r.get("variant") not in existing:
                        # Ensure 'status' key exists for consistency
                        if "status" not in r:
                            r = {**r, "status": "ok"}
                        partial_rows.append(r)

            # Recompute speed ranks across combined rows
            if partial_rows:
                totals = {
                    r["variant"]: float(r.get("uv_elapsed_s", 0.0)) + float(r.get("ver_elapsed_s", 0.0))
                    for r in partial_rows
                }
                best = min((v for v in totals.values() if v > 0.0), default=0.0)
                for r in partial_rows:
                    t = totals.get(str(r["variant"])) or 0.0
                    r["speed_vs_best"] = round((best / t) if (best > 0 and t > 0) else 0.0, 3)

            # Write combined report now
            csv_path = output_root / "reports" / "benchmark_summary.csv"
            json_path = output_root / "reports" / "benchmark_summary.json"
            md_path = output_root / "reports" / "benchmark_summary.md"
            if partial_rows:
                fieldnames = list(partial_rows[0].keys())
                with csv_path.open("w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=fieldnames)
                    w.writeheader()
                    w.writerows(partial_rows)
                json_path.write_text(json.dumps(partial_rows, ensure_ascii=False, indent=2), encoding="utf-8")

                lines = [
                    "# Benchmark Summary",
                    "",
                    "This report summarizes speed and accuracy across parametrized variants.",
                    "",
                    "Legend:",
                    "- Variant: base configuration + compute types in brackets (e.g., `[GPU float16 / CPU int8]`).",
                    "- UV RTF: real-time factor on unverified set (sum duration / elapsed); higher is faster.",
                    "- VER RTF: same on verified set.",
                    "- UV WER: WER vs baseline transcripts (1x large-v3 GPU float16).",
                    "- VER WER: WER vs filtered ASS (visual-only lines removed by token-overlap heuristic).",
                    "- Speed vs Best: relative speed (best total elapsed = 1.0).",
                    "- Status: ok or skipped-incompatible (variant skipped due to crash/incompatibility).",
                    "",
                    "Variant | Status | UV RTF | VER RTF | UV WER | VER WER | Speed vs Best",
                    "---|---|---:|---:|---:|---:|---:",
                ]
                for r in partial_rows:
                    lines.append(
                        f"{r['variant']} | {r.get('status', 'ok')} | {r['uv_rtf']} | {r['ver_rtf']} | {r['uv_wer']} | {r['ver_wer']} | {r.get('speed_vs_best', 0.0)}"
                    )
                md_path.write_text("\n".join(lines), encoding="utf-8")
                typer.echo(f"Reports updated ({idx_v}/{total_variants}): {md_path}")
        except Exception as ex:  # noqa: BLE001
            if verbose:
                typer.echo(f"  DEBUG: ERROR in incremental report: {ex}")
            import traceback
            traceback.print_exc()
        if verbose:
            typer.echo(f"  DEBUG: Incremental report done, continuing to next variant")

    # * Build recognizer union for filtering visual-only ASS lines
    recognizer_union_texts = []
    for var_name, bundle in variant_results.items():
        ver_outcomes = bundle["verified"]["outcomes"]  # type: ignore[index]
        recognizer_union_texts.extend(_collect_all_texts(ver_outcomes))

    # * Compute WERs for processed variants and assemble rows
    for var_name, bundle in variant_results.items():
        # Unverified WER (reference: large-v3 GPU transcripts)
        uv_outcomes: list[FileOutcome] = bundle["unverified"]["outcomes"]  # type: ignore[index]
        uv_elapsed = float(bundle["unverified"]["elapsed_s"])  # type: ignore[index]
        uv_dur = float(bundle["unverified"]["duration_s"])  # type: ignore[index]
        uv_rtf = float(bundle["unverified"]["rtf"])  # type: ignore[index]
        uv_ed_sum = 0
        uv_ref_tok_sum = 0
        for o in uv_outcomes:
            ref = uv_ref_by_stem.get(o.media.stem, "")
            w, ed, n = wer_score(ref, o.transcript)
            uv_ed_sum += ed
            uv_ref_tok_sum += n
        uv_wer = (float(uv_ed_sum) / float(uv_ref_tok_sum)) if uv_ref_tok_sum > 0 else 0.0

        # Verified WER (reference: ASS with visual-only filter)
        ver_outcomes: list[FileOutcome] = bundle["verified"]["outcomes"]  # type: ignore[index]
        ver_elapsed = float(bundle["verified"]["elapsed_s"])  # type: ignore[index]
        ver_dur = float(bundle["verified"]["duration_s"])  # type: ignore[index]
        ver_rtf = float(bundle["verified"]["rtf"])  # type: ignore[index]
        ver_ed_sum = 0
        ver_ref_tok_sum = 0
        # Build map for quick lookup
        out_by_stem = {o.media.stem: o.transcript for o in ver_outcomes}
        for a, s in ver_pairs:
            hyp = out_by_stem.get(a.stem, "")
            ref_filtered = ass_reference_text_with_filter(s, recognizer_union_texts)
            w, ed, n = wer_score(ref_filtered, hyp)
            ver_ed_sum += ed
            ver_ref_tok_sum += n
        ver_wer = (float(ver_ed_sum) / float(ver_ref_tok_sum)) if ver_ref_tok_sum > 0 else 0.0

        # Ensure status is present
        status_str = str(bundle.get("status", "ok"))
        report_rows.append(
            {
                "variant": var_name,
                "status": status_str,
                "uv_elapsed_s": round(uv_elapsed, 3),
                "uv_duration_s": round(uv_dur, 3),
                "uv_rtf": round(uv_rtf, 3),
                "uv_wer": round(uv_wer, 4),
                "ver_elapsed_s": round(ver_elapsed, 3),
                "ver_duration_s": round(ver_dur, 3),
                "ver_rtf": round(ver_rtf, 3),
                "ver_wer": round(ver_wer, 4),
            }
        )

    # * Merge with previous rows for resume, then recompute relative speed
    if old_rows:
        # Keep rows for variants we didn't process in this run
        new_keys = {r["variant"] for r in report_rows}
        for row in old_rows:
            if row.get("variant") not in new_keys:
                report_rows.append(row)

    if report_rows:
        totals = {
            r["variant"]: float(r.get("uv_elapsed_s", 0.0)) + float(r.get("ver_elapsed_s", 0.0))
            for r in report_rows
        }
        best = min((v for v in totals.values() if v > 0.0), default=0.0)
        for r in report_rows:
            t = totals.get(str(r["variant"])) or 0.0
            r["speed_vs_best"] = round((best / t) if (best > 0 and t > 0) else 0.0, 3)

    # * Write reports (CSV + JSON + Markdown)
    (output_root / "reports").mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "reports" / "benchmark_summary.csv"
    json_path = output_root / "reports" / "benchmark_summary.json"
    md_path = output_root / "reports" / "benchmark_summary.md"

    # Signal before final report build
    try:
        _signal_system_sound()
    except NameError:
        pass

    if report_rows:
        # Normalize rows to ensure 'status' exists
        report_rows = [{**r, "status": r.get("status", "ok")} for r in report_rows]
        fieldnames = list(report_rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(report_rows)
        json_path.write_text(json.dumps(report_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        # * Markdown report with legend
        lines = [
            "# Benchmark Summary",
            "",
            "This report summarizes speed and accuracy across parametrized variants.",
            "",
            "Legend:",
            "- Variant: base configuration + compute types in brackets (e.g., `[GPU float16 / CPU int8]`).",
            "- UV RTF: real-time factor on unverified set (sum duration / elapsed); higher is faster.",
            "- VER RTF: same on verified set.",
            "- UV WER: WER vs baseline transcripts (1x large-v3 GPU float16).",
            "- VER WER: WER vs filtered ASS (visual-only lines removed by token-overlap heuristic).",
            "- Speed vs Best: relative speed (best total elapsed = 1.0).",
            "- Status: ok or skipped-incompatible (variant skipped due to crash/incompatibility).",
            "",
            "Variant | Status | UV RTF | VER RTF | UV WER | VER WER | Speed vs Best",
            "---|---|---:|---:|---:|---:|---:",
        ]
        for r in report_rows:
            lines.append(
                f"{r['variant']} | {r.get('status','ok')} | {r['uv_rtf']} | {r['ver_rtf']} | {r['uv_wer']} | {r['ver_wer']} | {r.get('speed_vs_best', 0.0)}"
            )
        md_path.write_text("\n".join(lines), encoding="utf-8")
        typer.echo(f"Reports: {csv_path} | {json_path} | {md_path}")

        # Signal after final report build
        try:
            _signal_system_sound()
        except NameError:
            pass


app = typer.Typer(help="Benchmark Whisper variants (GPU/CPU, small/large-v3) with WER & speed metrics")


@app.command()
def phase(
    dataset: Literal["uv", "ver"] = typer.Option(..., help="Dataset to run: uv or ver"),
    variant_base: str = typer.Option(..., help="Base variant name (without compute decorators)"),
    filelist: Path = typer.Option(..., exists=True, readable=True, help="JSON file with list of media file paths"),
    out: Path = typer.Option(..., help="Output directory to write transcripts"),
    gpu_compute: str | None = typer.Option(None, help="GPU compute type"),
    cpu_compute: str | None = typer.Option(None, help="CPU compute type"),
    variant_idx: int | None = typer.Option(None, help="Index of variant among all to display progress"),
    variants_total: int | None = typer.Option(None, help="Total number of variants to display progress"),
    verbose: bool = typer.Option(False, "--verbose", "--Verbose", help="Verbose logging"),
    silent_progress: bool = typer.Option(False, "--silent-progress", hidden=True, help="Disable child live progress output (managed by parent)"),
):
    """Run a single phase (uv or ver) for a specific base variant.

    Prints a one-line JSON with elapsed_s when done.
    """
    try:
        files_list = _json.loads(filelist.read_text(encoding="utf-8"))
        files = [Path(p) for p in files_list]
    except Exception as ex:  # noqa: BLE001
        raise typer.Exit(code=2) from ex
    # Resolve variant by base name
    base: Variant | None = None
    for v in VARIANTS:
        if v.name == variant_base:
            base = v
            break
    if base is None:
        typer.secho(f"Unknown variant base: {variant_base}", err=True)
        raise typer.Exit(code=2)
    out.mkdir(parents=True, exist_ok=True)
    label = (
        f"Var {variant_idx}/{variants_total}: {variant_base} ({dataset})"
        if (variant_idx is not None and variants_total is not None)
        else f"{variant_base} ({dataset})"
    )
    _outcomes, elapsed = run_variant_on_files(
        base,
        files,
        out,
        gpu_compute=gpu_compute,
        cpu_compute=cpu_compute,
        variant_label=label,
        verbose=verbose,
        silent_progress=silent_progress,
    ) if files else ([], 0.0)
    # * Validate outputs: require that all expected transcripts exist and are non-empty
    ok = True
    missing: list[str] = []
    empty: list[str] = []
    try:
        stems = [p.stem for p in files]
        for s in stems:
            path = out / f"{s}.txt"
            if not path.exists():
                ok = False
                missing.append(s)
                continue
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
                if not content.strip():
                    ok = False
                    empty.append(s)
            except Exception:
                ok = False
                empty.append(s)
    except Exception:
        ok = False
    # * Write sentinel with validation details
    try:
        sentinel = out / ".phase_ok.json"
        payload = {
            "elapsed_s": elapsed,
            "ok": ok,
            "missing": missing,
            "empty": empty,
            "files": [str(p) for p in files],
        }
        sentinel.write_text(_json.dumps(payload), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    print(_json.dumps({"elapsed_s": elapsed, "ok": ok, "missing": missing, "empty": empty}))
    if not ok:
        raise typer.Exit(code=3)


@app.command()
def run(
    unverified: Path = typer.Option(
        Path("temporary/unverified_benchmark"), exists=False, readable=True,
        help="Folder with unverified videos (reference = large-v3 GPU)"
    ),
    verified: Path = typer.Option(
        Path("temporary/verified_benchmark"), exists=False, readable=True,
        help="Folder with verified audios (.mka/.wav) and matching .ass subtitles"
    ),
    output: Path = typer.Option(
        Path("tools_out/benchmark"), help="Output directory for transcripts and reports"
    ),
    variants: str = typer.Option(
        "", help="Comma-separated variant names to run (empty = all). See script header for list."
    ),
    max_unverified: int = typer.Option(-1, help="Limit number of unverified files (-1 = all, 0 = skip)"),
    max_verified: int = typer.Option(-1, help="Limit number of verified pairs (-1 = all, 0 = skip)"),
    full: bool = typer.Option(False, "--full", help="Run full benchmark (default if no CPU/GPU flags)"),
    CPU: bool = typer.Option(False, "--CPU", help="Run CPU-only variants"),
    GPU: bool = typer.Option(False, "--GPU", help="Run GPU-only variants"),
    clear: bool = typer.Option(False, "--clear", help="Ignore resume and start from scratch"),
    verbose: bool = typer.Option(False, "--verbose", "--Verbose", help="Verbose logging"),
):
    """Run benchmark across selected variants and datasets."""
    include = None
    if variants.strip():
        include = [v.strip() for v in variants.split(",") if v.strip()]
    # Resolve mode from flags
    mode: Literal["full", "cpu", "gpu"]
    if CPU and GPU:
        mode = "full"
    elif CPU:
        mode = "cpu"
    elif GPU:
        mode = "gpu"
    else:
        mode = "full" if full or not (CPU or GPU) else "full"

    try:
        benchmark(
            unverified,
            verified,
            output,
            include_variants=include,
            max_unverified=max_unverified,
            max_verified=max_verified,
            mode=mode,
            resume_clear=bool(clear),
            verbose=verbose,
        )
    except KeyboardInterrupt:
        typer.secho("\n\nBenchmark interrupted by user. Partial results may be available.", err=True)
        raise typer.Exit(code=130)
    except Exception as ex:  # noqa: BLE001
        typer.secho(f"FATAL: {ex}", err=True)
        raise


@app.callback(invoke_without_command=True)
def main_entry(
    ctx: typer.Context,
    unverified: Path = typer.Option(
        Path("temporary/unverified_benchmark"), exists=False, readable=True,
        help="Folder with unverified videos (reference = large-v3 GPU)"
    ),
    verified: Path = typer.Option(
        Path("temporary/verified_benchmark"), exists=False, readable=True,
        help="Folder with verified audios (.mka/.wav) and matching .ass subtitles"
    ),
    output: Path = typer.Option(
        Path("tools_out/benchmark"), help="Output directory for transcripts and reports"
    ),
    variants: str = typer.Option(
        "", help="Comma-separated variant names to run (empty = all). See script header for list."
    ),
    max_unverified: int = typer.Option(-1, help="Limit number of unverified files (-1 = all, 0 = skip)"),
    max_verified: int = typer.Option(-1, help="Limit number of verified pairs (-1 = all, 0 = skip)"),
    full: bool = typer.Option(False, "--full", help="Run full benchmark (default if no CPU/GPU flags)"),
    CPU: bool = typer.Option(False, "--CPU", help="Run CPU-only variants"),
    GPU: bool = typer.Option(False, "--GPU", help="Run GPU-only variants"),
    clear: bool = typer.Option(False, "--clear", help="Ignore resume and start from scratch"),
    verbose: bool = typer.Option(False, "--verbose", "--Verbose", help="Verbose logging"),
):
    # If a subcommand is provided, Typer will not invoke this
    if ctx.invoked_subcommand is not None:
        return
    include = None
    if variants.strip():
        include = [v.strip() for v in variants.split(",") if v.strip()]
    if CPU and GPU:
        mode = "full"
    elif CPU:
        mode = "cpu"
    elif GPU:
        mode = "gpu"
    else:
        mode = "full" if full or not (CPU or GPU) else "full"
    try:
        benchmark(
            unverified,
            verified,
            output,
            include_variants=include,
            max_unverified=max_unverified,
            max_verified=max_verified,
            mode=mode,
            resume_clear=bool(clear),
            verbose=verbose,
        )
    except KeyboardInterrupt:
        typer.secho("\n\nBenchmark interrupted by user. Partial results may be available.", err=True)
        raise typer.Exit(code=130)
    except Exception as ex:  # noqa: BLE001
        typer.secho(f"FATAL: {ex}", err=True)
        raise


def main() -> None:
    app()


if __name__ == "__main__":
    main()


