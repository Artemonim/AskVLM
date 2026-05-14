## Main Flow

1. Launch transcriber
1.1. Input
1.1.1. file (audio/video)
1.1.2. folder (flat); recursive mode — planned
1.2. Output (destination directory and export format)
1.2.1. txt | srt | vtt | json (docx/odt/md — planned)

2. Processing settings (before start)
2.1. Engine: Auto (faster-whisper; alignment/words — via whisperx when available)
2.2. Language: auto or specified code (e.g. ru)
2.3. Diarization: on/off (VAD+speakers; improvements — planned)
2.4. Dialog blocks (LLM formatting): on/off
2.5. Model: Auto (selected by VRAM/speed); manual selection — in "Advanced" (optional)
2.6. Device/compute type: auto (GPU → FP16, otherwise INT8/FP32)
2.7. Engine and model selection hidden in UI: "Auto" by default; toggles appear only in "Advanced".

3. Automatic processing
3.1. Plain text — no post-processing
3.2. Phrases — standard Whisper output: "timing → phrase"
3.3. Dialog — text split into blocks by speaker; with LLM enabled — improved readability
3.4. Word alignment (whisperx) — optional, when available

4. Manual processing
4.1. WYSIWYG editor: view/edit text
4.2. Planned: split/merge segments, change speaker, remove timecodes, find/replace, spellcheck

5. Export
5.1. TXT — full text
5.2. SRT/VTT — subtitles with timecodes and speakers
5.3. JSON — segments with structure {speaker, start, end, text}

6. CLI (quick scenario)
6.1. Example: `python cli.py transcribe -i INPUT -o output --engine whisperx --diarization --dialog-blocks --export srt`

7. Status and progress
7.1. GUI — status bar, non-blocking operations
7.2. CLI — informative output; ETA — planned

8. Secrets and models
8.1. HF_TOKEN — for diarization models (ENV)
8.2. LLM_GGUF_PATH — path to local model for formatting (ENV)

11. Decision on engine and model selection
11.1. Do not show a "Whisper/WhisperX" toggle in the UI. Use faster-whisper by default; if whisperx is installed — enable alignment/words when the "Alignment" option is selected.
11.2. Model presets are hidden. "Auto" mode selects the model by VRAM/speed. Manual selection is available for advanced users in "Advanced".

9. Errors and resilience
9.1. OOM/GPU unavailable — auto-downgrade compute type; recommend disabling diarization/LLM
9.2. Unsupported format — auto-convert to WAV 16 kHz mono

10. UX principles
10.1. Sensible defaults and remember last settings
10.2. Clear error messages and possible remediation steps
