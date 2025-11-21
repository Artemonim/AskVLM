from __future__ import annotations

"""
Minimal tool to find the maximum media duration that WhisperX (large-v3, GPU, float32)
can transcribe without running out of GPU memory (OOM).

Usage:
  python tools/find_oom_threshold.py

Configuration variables are at the top of this file.
"""

from pathlib import Path
import os
import sys
import time
import traceback
from typing import List, Tuple

# * Configuration (edit if needed) ------------------------------------------------------------
MEDIA_DIR = Path("temporary/chaotic")
SUPPORTED_EXTS = {".mp4", ".mkv", ".mov", ".webm", ".wav", ".mka", ".mp3", ".flac"}
MODEL_NAME = "large-v3"
DEVICE = "cuda"
COMPUTE_TYPE = "float32"
WORK_DIR = Path("tools_out/find_oom_threshold")
VERBOSE = True
RECREATE_WRAPPER_PER_TRY = True  # recreate wrapper after each failed attempt for stability
TIMEOUT_S = 600.0  # per-file timeout (not enforced strictly, but used for logging)

# * Implementation ---------------------------------------------------------------------------
def _list_media_files_sorted_by_duration(media_dir: Path) -> List[Tuple[Path, float]]:
    """Return list of (path, duration_s) sorted ascending by duration."""
    from core.ffmpeg import get_media_duration_seconds

    files: List[Tuple[Path, float]] = []
    if not media_dir.exists():
        return []
    for p in sorted(media_dir.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in SUPPORTED_EXTS:
            continue
        try:
            d = max(0.0, get_media_duration_seconds(p))
        except Exception:
            d = 0.0
        files.append((p, d))
    files.sort(key=lambda x: x[1])
    return files


def _make_wrapper() -> "WhisperXWrapper":
    # Lazily import heavy dependency to fail fast if missing
    from core.whisperx_wrapper import WhisperXWrapper

    return WhisperXWrapper(model_name=MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE, model_root=None)


def _transcribe_with_wrapper(wrapper, media: Path, work_dir: Path) -> Tuple[bool, str, float]:
    """
    Try to transcribe `media` with provided `wrapper`.
    Returns (success, message_or_transcript, elapsed_s).
    """
    from core.audio_io import prepare_audio

    t0 = time.perf_counter()
    try:
        wav = prepare_audio(media, work_dir)
        res = wrapper.transcribe(wav)
        txt = str(res.get("text", "")).strip()
        return True, txt, max(0.0, time.perf_counter() - t0)
    except Exception as ex:  # noqa: BLE001
        msg = "".join(traceback.format_exception_only(type(ex), ex)).strip()
        # Classify common OOM signatures
        m = msg.lower()
        if "out of memory" in m or "cuda out of memory" in m or "cudnn" in m:
            return False, "OOM: " + msg, max(0.0, time.perf_counter() - t0)
        return False, "ERROR: " + msg, max(0.0, time.perf_counter() - t0)


def attempt_file(media: Path, work_dir: Path) -> Tuple[bool, str, float]:
    """
    Attempt to transcribe `media`. Creates a fresh wrapper, runs transcription,
    unloads wrapper where possible and returns the result.
    """
    wrapper = None
    try:
        wrapper = _make_wrapper()
    except Exception as ex:
        return False, f"WRAPPER_INIT_ERROR: {ex}", 0.0

    try:
        ok, msg, elapsed = _transcribe_with_wrapper(wrapper, media, work_dir)
        return ok, msg, elapsed
    finally:
        # Attempt explicit unload to free GPU memory
        try:
            unload = getattr(wrapper, "unload", None)
            if callable(unload):
                unload(safe=True)
        except Exception:
            pass
        try:
            import gc as _gc

            _gc.collect()
        except Exception:
            pass


def binary_search_max_success(files_sorted: List[Tuple[Path, float]], work_dir: Path) -> Tuple[int, List[dict]]:
    """
    Binary search on the sorted list of files to find the maximum index that succeeds.
    Returns (best_index, attempts_log).
    attempts_log is a list of dicts with keys: index, path, duration_s, success, msg, elapsed_s
    """
    attempts = []
    if not files_sorted:
        return -1, attempts
    low = 0
    high = len(files_sorted) - 1
    best = -1
    while low <= high:
        mid = (low + high) // 2
        path, dur = files_sorted[mid]
        if VERBOSE:
            print(f"Trying index {mid} -> {path} ({dur:.1f}s)")
        # attempt
        ok, msg, elapsed = attempt_file(path, work_dir)
        attempts.append(
            {"index": mid, "path": str(path), "duration_s": dur, "success": bool(ok), "msg": msg, "elapsed_s": elapsed}
        )
        if ok:
            best = mid
            low = mid + 1
            if VERBOSE:
                print(f"  SUCCESS ({elapsed:.1f}s). Moving low -> {low}")
        else:
            high = mid - 1
            if VERBOSE:
                print(f"  FAIL ({msg}). Moving high -> {high}")
        # Optionally recreate wrapper between attempts is handled inside attempt_file
    return best, attempts


def main():
    out_work_dir = WORK_DIR
    out_work_dir.mkdir(parents=True, exist_ok=True)

    files_sorted = _list_media_files_sorted_by_duration(MEDIA_DIR)
    if not files_sorted:
        print(f"No media found in {MEDIA_DIR} with extensions {SUPPORTED_EXTS}")
        raise SystemExit(2)

    print(f"Found {len(files_sorted)} media files. Longest duration: {files_sorted[-1][1]:.1f}s")
    best_idx, attempts = binary_search_max_success(files_sorted, out_work_dir)

    # Persist attempts
    import json

    out_json = out_work_dir / "attempts.json"
    out_json.write_text(json.dumps(attempts, ensure_ascii=False, indent=2), encoding="utf-8")

    if best_idx >= 0:
        best_path, best_dur = files_sorted[best_idx]
        print(f"\nMax successful index: {best_idx} -> {best_path} ({best_dur:.1f}s)")
    else:
        print("\nNo file could be transcribed successfully (all attempts failed).")


if __name__ == "__main__":
    main()


