from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

from core.whisperx_wrapper import WhisperXWrapper


def main() -> int:
    """Attempt to load large-v3 in FP32 on CUDA and report OOM status.

    Prints a one-line JSON with keys:
    - status: ok | oom | no_cuda | fail
    - detail: short message for diagnostics
    """
    # * Check CUDA availability first
    try:
        torch_mod = importlib.import_module("torch")
        if getattr(torch_mod, "cuda", None) is None or not torch_mod.cuda.is_available():
            print(json.dumps({"status": "no_cuda", "detail": "CUDA not available"}))
            return 0
    except ModuleNotFoundError:
        print(json.dumps({"status": "fail", "detail": "torch not installed"}))
        return 0

    wrapper = WhisperXWrapper(
        model_name="large-v3",
        device="cuda",
        compute_type="float32",
        model_root=None,
    )

    try:
        # * Trigger model load into VRAM
        #   Private method usage is acceptable for this diagnostic tool.
        wrapper._load_model()  # noqa: SLF001
        # Optional sync to ensure lazy allocations materialize
        try:
            torch_mod.cuda.synchronize()
        except Exception:
            pass
        print(json.dumps({"status": "ok", "detail": "Loaded large-v3 FP32 on CUDA"}))
        return 0
    except (RuntimeError, MemoryError) as exc:
        msg = str(exc)
        low = msg.lower()
        # * Heuristics for OOM classification
        if (
            "out of memory" in low
            or "cuda error" in low and "memory" in low
            or "cublas" in low and "alloc" in low
            or "cudnn" in low and "alloc" in low
            or "failed to allocate" in low
        ):
            print(json.dumps({"status": "oom", "detail": msg[:300]}))
            return 0
        print(json.dumps({"status": "fail", "detail": msg[:300]}))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"status": "fail", "detail": str(exc)[:300]}))
        return 0
    finally:
        # * Best-effort cleanup
        try:
            wrapper.unload()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())








