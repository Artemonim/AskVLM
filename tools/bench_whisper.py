from __future__ import annotations

import importlib
import time
from pathlib import Path

from core.audio_io import prepare_audio
from core.whisperx_wrapper import WhisperXWrapper


def detect_vram_gb() -> float | None:
    try:
        torch_mod = importlib.import_module("torch")
    except ModuleNotFoundError:
        return None
    if getattr(torch_mod, "cuda", None) is None or not torch_mod.cuda.is_available():
        return None
    try:
        idx = torch_mod.cuda.current_device()
        props = torch_mod.cuda.get_device_properties(idx)
        return float(props.total_memory) / float(1024**3)
    except Exception:  # noqa: BLE001
        return None


def main() -> None:
    input_path = Path("tests/fixtures/test_video_first.mp4")
    models = ["small", "medium", "large-v3"]
    device = "cuda"
    profile = "grid"  # baseline|high|extreme|grid
    compute_type = "auto"
    monitor_vram = True

    vram = detect_vram_gb()
    print(f"[*] Detected VRAM: {vram:.2f} GiB" if vram is not None else "[*] VRAM: unknown")

    out_dir = Path("transcriptions")
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = prepare_audio(input_path, out_dir)

    def recommend_compute_type(model_name: str, vram_gb: float | None, prof: str) -> str:
        if compute_type != "auto":
            return compute_type
        if prof == "baseline":
            if model_name.startswith("large"):
                if vram_gb is not None and vram_gb >= 12.0:
                    return "float16"
                if vram_gb is not None and vram_gb >= 8.0:
                    return "int8_float16"
                return "int8"
            return "float16" if (vram_gb or 0.0) >= 8.0 else "int8_float16"
        if prof == "high":
            if model_name.startswith("large"):
                if vram_gb is not None and vram_gb >= 12.0:
                    return "float16"
                if vram_gb is not None and vram_gb >= 8.0:
                    return "int8_float16"
                return "int8"
            return "float16" if (vram_gb or 0.0) >= 8.0 else "int8_float16"
        if prof == "extreme":
            return "float16"
        return "float16" if (vram_gb or 0.0) >= 8.0 else "int8_float16"

    def decode_kwargs_for_profile(prof: str) -> dict[str, object]:
        if prof == "baseline":
            return {}
        if prof == "high":
            return {"beam_size": 10, "vad_filter": True, "word_timestamps": False}
        if prof == "extreme":
            return {"beam_size": 10, "vad_filter": True, "word_timestamps": True}
        return {}

    def get_nvidia_vram() -> tuple[float | None, float | None]:
        if not monitor_vram:
            return (None, None)
        try:
            import subprocess

            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            line = out.strip().splitlines()[0]
            total_str, used_str = [s.strip() for s in line.split(",")]
            total_mib = float(total_str)
            used_mib = float(used_str)
            return (total_mib, used_mib)
        except Exception:  # noqa: BLE001
            return (None, None)

    results: dict[tuple[str, str], tuple[float | None, str]] = {}
    profiles = ["baseline", "high", "extreme"] if profile == "grid" else [profile]
    for model_name in models:
        for prof in profiles:
            ct = recommend_compute_type(model_name, vram, prof)
            decode_kwargs = decode_kwargs_for_profile(prof)
            total_before, used_before = get_nvidia_vram()
            print(f"[*] Benchmark {model_name} [{prof}] (compute_type={ct})...")
            wrapper = WhisperXWrapper(model_name=model_name, device=device, compute_type=ct, model_root=None)
            t0 = time.perf_counter()
            status = "ok"
            elapsed: float | None = None
            try:
                _ = wrapper.transcribe(wav_path, **decode_kwargs)
                elapsed = time.perf_counter() - t0
                print(f"    -> {elapsed:.2f} s")
            except (RuntimeError, MemoryError) as exc:
                status = f"OOM/Fail: {str(exc).splitlines()[0][:120]}"
                print(f"    -> {status}")
            results[(model_name, prof)] = (elapsed, status)
            try:
                torch_mod = importlib.import_module("torch")
                torch_mod.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
            total_after, used_after = get_nvidia_vram()
            if monitor_vram and None not in (total_before, used_before, total_after, used_after):
                delta = (used_after or 0) - (used_before or 0)
                print(f"    VRAM used delta (nvidia-smi): {delta:.0f} MiB (total {total_after:.0f} MiB)")

    print("\n=== Benchmark Summary ===")
    by_profile: dict[str, list[str]] = {"baseline": [], "high": [], "extreme": []}
    for (name, prof), (elapsed, status) in sorted(results.items()):
        val = f"{name:10s}: {(f'{elapsed:.2f} s' if elapsed is not None else status)}"
        if prof in by_profile:
            by_profile[prof].append(val)
    for prof in profiles:
        print(f"[{prof}]")
        for line in by_profile.get(prof, []):
            print("  " + line)


if __name__ == "__main__":
    main()






















