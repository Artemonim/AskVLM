### AskVLM — Auto Subtitles: Production Pipeline and DaVinci Resolve Integration

Date: 2025‑10‑19

---

### Summary

- **Goal**: reliable auto-subtitling of finished video with emphasis on timing quality and performance on Windows + CUDA 8 GB.
- **Engine**: WhisperX (forced alignment + word timestamps + optional diarization) as primary; fallback to faster‑whisper‑only when constrained.
- **Default artifacts**: always burn subtitles into video + always save sidecar `.srt` for QA/re-export.
- **Resolve integration**: basic flow via `.srt` import (no plugin at first stage). Plugin/bridge — later.

---

### Hardware Assumptions (default profile)

- OS: Windows
- GPU: CUDA, 8 GB VRAM
- Storage: NVMe for faster weight loading (beneficial for OS file cache)
- Parallelism: one heavy GPU stage at a time; CPU tasks can be parallelized

---

### Architecture Principles

1) **WhisperX as the primary alignment layer**: timing improvement (align), word-level timestamps; diarization (pyannote) is optional.
2) **Fallback strategy**: if align/diar are unavailable (VRAM/environment) — run ASR (faster‑whisper) and continue the pipeline.
3) **RAM↔VRAM orchestration**: keep PyTorch models (Align/Diari) "hot" in RAM; load into VRAM strictly one at a time for the duration of each stage; unload after the stage.
4) **Preload on idle**: lazy CPU-loading of Align/Diari and OS file cache "warm-up" for ASR before start.
5) **Chunking**: split long videos (VAD/timeline) for stable memory usage and predictable compute time.

---

### Memory Orchestration (RAM↔VRAM) and Performance

- **ASR (faster‑whisper / CTranslate2)**:
  - On 8 GB: `compute_type=int8_float16` (or `float16` if stable), `batch_size=8–16`.
  - Keep the GPU instance only during recognition. Repeated loads speed up due to OS page cache.
- **Align (WhisperX align)**:
  - Keep in RAM, move to CUDA only during the step; use a small `batch_size`.
- **Diarization (pyannote)**:
  - On flag; after align. When VRAM is scarce — fall back to CPU for diarization.
- **Strict sequencing**: only one heavy model in VRAM at a time. Between stages: `del model; torch.cuda.empty_cache()`.
- **VRAM estimation**: check free memory before moving to CUDA; if insufficient — reduce `batch_size`/CPU fallback.

---

### Subtitle Export and Burn-in

- Always save `.srt` (UTF-8). By default also burn into video via `ffmpeg` (`subtitles=`/`libass`).
- For complex styling — generate `.ass` and burn via `-vf subtitles=...`.

Examples (for reference):

```bash
# Burn SRT (simple styling — ensure ffmpeg is built with libass)
ffmpeg -i input.mp4 -vf "subtitles='subs.srt':force_style='Fontsize=42,Outline=2,Shadow=0'" -c:v libx264 -crf 18 -preset veryfast -c:a copy output.mp4

# Burn ASS (more precise style control)
ffmpeg -i input.mp4 -vf "subtitles='subs.ass'" -c:v libx264 -crf 18 -preset veryfast -c:a copy output.mp4
```

---

### Subtitle Formatting (readability rules)

- 1–2 lines, up to ~42 characters per line
- Duration: ~1.2–6.0 seconds (merge short ones, split very long ones)
- Reading speed (CPS): target max ~17–18
- Split by words/punctuation using word-timestamps from WhisperX

Export to SRT/WebVTT; optionally `.ass` for style control.

---

### DaVinci Resolve Integration (recommended flow)

1) Assemble the edit in Resolve and lock the final timeline.
2) Export the "master" (or faster: `audio‑only WAV 16 kHz`).
3) In AskVLM, generate the `.srt` (and optionally a preview burned copy).
4) Import `.srt` into Resolve: File → Import → Subtitle → configure Track Style in Inspector.
5) If needed, adjust text/timing in Resolve, then final export:
   - Burn into video — burn during Resolve render
   - Export Subtitle — export sidecar
6) Afterwards — if needed, run the final through a video converter for target platforms.

Note: a Resolve bridge/plugin is possible later; the fast path now is "audio-only export → SRT → import" or watch-folder mode in AskVLM.

---

### CLI and GUI (minimum requirements)

- **CLI (`subtitle`)**: batch over files/folders; flags: `--burn-in` (default true), `--save-srt` (always true), `--diarize`, `--device`, `--compute-type`, `--batch-size`, `--vad`, `--max-cps`, `--max-line-chars`, `--max-lines`, `--min-duration`, `--max-duration`, `--format srt|vtt|ass`.
- **GUI**: checkboxes "Burn Subtitles", "Also Save .srt", "Diarization", style/format selection.

---

### .env (placeholders — replace with your own values)

```
# Base paths/caches
SK_MODELS_DIR=PATH_TO_MODELS
SK_CACHE_DIR=PATH_TO_CACHE
HF_HOME=PATH_TO_HF_CACHE
TRANSFORMERS_CACHE=PATH_TO_HF_CACHE
TORCH_HOME=PATH_TO_TORCH_CACHE
FFMPEG_PATH=PATH_TO_FFMPEG_BIN

# Performance/device
CUDA_VISIBLE_DEVICES=0
SK_DEVICE=auto                 # auto|cuda|cpu
SK_COMPUTE_TYPE=int8_float16   # 8 GB: int8_float16 or float16
SK_BATCH_SIZE=8
SK_VAD=true
SK_DIARIZE=false               # enable selectively
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.9,max_split_size_mb:64
TOKENIZERS_PARALLELISM=false

# Prefetch/orchestration
SK_PREFETCH_ON_IDLE=true       # lazy preload into RAM
SK_RAM_RESIDENCY=true          # keep Align/Diari in CPU RAM

# Secrets/clients (as needed)
HF_TOKEN=YOUR_HF_TOKEN         # for pyannote (diarization)
OPENAI_API_KEY=YOUR_OPENAI_KEY # if using cloud LLM/ASR
ANTHROPIC_API_KEY=YOUR_ANTHROPIC_KEY
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_API_KEY=...
YANDEX_SPEECHKIT_OAUTH=...
YANDEX_SPEECHKIT_FOLDER_ID=...
HTTPS_PROXY=...
HTTP_PROXY=...

# Behavior/logs
SK_OUTPUT_DIR=PATH_TO_OUTPUT
SK_LOG_LEVEL=INFO
SK_BURN_SUBS_DEFAULT=true
SK_SAVE_SRT_ALWAYS=true
```

Important: do not commit `.env` or files with keys; add them to `.gitignore`.

---

### SaaS Aspects (brief)

- **RAM residency** of Align/Diari with strict sequential VRAM migration is more economical than permanent "life" in large VRAM at 1–2 concurrent tasks.
- **ASR (CT2)** relies on NVMe+OS cache — RAM residency provides less benefit than for PyTorch models.
- **LLM post-correction**: competes with ASR on 8 GB; better to offload to a separate service (vLLM) or a provider API.
- **Scaling**: task queue; one GPU worker — one heavy step; CPU stages can be parallelized; weight caching on NVMe.

---

### Quality Control (QA)

- Sample 0–10 minutes to check text/timings
- Validate CPS (≤ ~18), durations (1.2–6.0 s), line breaks
- Check names/numbers/terms; for multi-speaker — speaker labels
- Versioning: `video_v1.srt`, `video_v2_resynced.srt`

---

### Roadmap (first increment)

1) CLI `subtitle` (batch) and pipeline: ASR → Align → (Diar opt) → formatting → `.srt` → burn‑in.
2) `gpu_guard`/`ModelRegistry`: strictly sequential GPU stages, preload Align/Diari into RAM.
3) Exporters: SRT/WebVTT/ASS with rules (CPS/durations/line breaks).
4) FFmpeg wrapper: `burn_subtitles(video, srt_or_ass, style, out)`.
5) GUI minimum: "Burn Subtitles", "Save .srt", "Diarization", format.
6) Tests: integration WAV → JSON → SRT; short video → SRT → burn-in; CPS/format validation.

---

### Open Questions / Deferred Decisions

- Resolve plugin/bridge — stage 2 (after pipeline stabilization).
- Extracting `littletools_video` into a separate project — after UX/codec/profile requirements are finalized.
