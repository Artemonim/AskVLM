### Disfluency Cleanup (RU) — Design Document

Date: 2025‑10‑23

Version: 0.1 (Draft)

Project: AskVLM

Scope: offline automatic disfluency cleanup in Russian speech: sighs, filler sounds ("э‑э", "мм"), filler words (e.g. "ну", "типа", "короче").

---

### 1) Goals and Criteria

- Minimize manual audio cleanup through automatic detection and safe application of edits.
- Do not break video/subtitle timings: default — non-destructive processing (attenuation), optional — cut with crossfade.
- Controllability: flags, thresholds, limits, strict/normal/aggressive modes.
- Traceability: JSON cutlist, EDL for NLE, metrics log, rollback capability.

---

### 2) Processing Targets

- Sighs/breathing: short noise/sustain intervals outside speech, 200–1200 ms.
- Filler sounds: "э‑э", "мм", stretched vowels with no semantic content.
- Filler words (ru): "ну", "типа", "как бы", "короче", "в общем", "по сути", "получается", "значит", "это самое", "вот", "скажем так", "так сказать", "как сказать", "да", "в принципе", "в итоге", etc.

---

### 3) Integration into the Pipeline

High-level flow:

1. Audio preparation (`core/ffmpeg.py`).
2. ASR + word-level alignment (WhisperX) → word timestamps.
3. DisfluencyDetection (new step):
   - FillerWordsDetector (dictionary + alignment)
   - FillerSoundsDetector (tokens + acoustic heuristics)
   - BreathsDetector (heuristics + optional PANNs classifier)
4. Post-processing of intervals: merging, boundaries, limits, confidence.
5. Cutlist JSON generation.
6. Application: Attenuate (default) or Cut+Crossfade (option) to WAV copy.
7. Export: updated SRT/VTT, EDL for NLE, JSON metrics report.

Position in code: optional step in `core/pipelines.LocalPipeline` after Alignment and before Exporters.

---

### 4) Detection Methods

#### 4.1 Filler Words (ru)

- Base: dictionary of stable markers ("ну", "типа", "как бы", …). Lists are extendable in settings.
- Audio anchoring: word-timestamps from WhisperX.
- Normalization: simple lemmatization for stable phrases (optional). For MVP, lowercased exact-match on WhisperX tokens is sufficient.
- False positive protection:
  - strict/normal/aggressive modes.
  - Allowlist of words ("да" in answers) and context-based exclusions.
  - Deletion fraction limit per phrase/minute (e.g. ≤10%).

#### 4.2 Filler Sounds ("э‑э", "мм")

- Text channel: sequences of tokens like "э", "эээ", "м", "ммм" from the transcript.
- Acoustic heuristics for ASR gaps:
  - Sustained vowel with low spectral dynamics (low spectral flux), low formant variability.
  - Duration 80–400 ms, energy above noise threshold, no clear consonant attack.
- Merging of close intervals, min/max duration, confidence from combined channels.

#### 4.3 Sighs/Breathing

- Heuristics:
  - Consider non-speech segments (VAD/ASR pauses).
  - Features: spectral slope, band ratios (e.g. 300–1500 Hz), envelope, ZCR.
  - Duration 200–1200 ms, margins from adjacent words.
- Optional PANNs (AudioSet class "Breathing/Sigh") for additional confidence.
- Default action: attenuate (−12 dB, linear/cosine envelope 10–30 ms).

---

### 5) Applying Edits (Safety)

- Attenuate (default):
  - Gain: −12 dB (configurable), fade-in/out 10–30 ms, envelope type — cosine/exponential.
  - Virtually does not break rhythm, minimal artifacts.
- Cut + Crossfade (option):
  - Choose ripple/non-ripple:
    - Non-ripple: timeline stays unchanged, replacing the cut with silence/attenuation.
    - Ripple: time shifts; suitable for audio podcast version, but requires care with subtitles/video.
  - Crossfade 2–40 ms (adaptive by speech rate and surroundings).
- Restrictions:
  - Maximum number of edits per unit of speech (protection against "jumpy" phrases).
  - Do not cut if the interval overlaps a word by >30–40% (switch to attenuate).

---

### 6) Cutlist JSON Format

```json
{
  "version": "1.0",
  "source": {
    "file": "PATH_TO_INPUT",
    "sample_rate": 16000
  },
  "items": [
    {
      "id": "cl_00001",
      "type": "filler_word",  
      "start_ms": 1250,
      "end_ms": 1420,
      "confidence": 0.93,
      "action": "attenuate",  
      "margin_before_ms": 20,
      "margin_after_ms": 20,
      "tags": ["ru_dict:ну"]
    }
  ]
}
```

Types: `filler_word` | `filler_sound` | `sigh`.
Action: `attenuate` | `cut`.

---

### 7) Settings (settings.json)

- cleanup.enable: true|false
- cleanup.types: { filler_words: true, filler_sounds: true, breaths: true }
- cleanup.strategy: "attenuate" | "cut"
- cleanup.limits: { per_sentence_max_fraction: 0.1, per_minute_max_ms: 8000 }
- cleanup.margins_ms: { before: 20, after: 20 }
- cleanup.fade_ms: { in: 15, out: 15 }
- cleanup.gain_db: -12
- cleanup.filler_words.dict: ["ну", "типа", "как бы", ...]
- cleanup.thresholds: { sigh_conf: 0.6, sound_conf: 0.6 }

All values are overridable via GUI/CLI.

---

### 8) CLI

```bash
speach-kit transcribe \
  --input PATH_TO_MEDIA \
  --cleanup-disfluencies \
  --cleanup-types filler_words,filler_sounds,breaths \
  --cleanup-strategy attenuate \
  --cleanup-gain-db -12 \
  --cleanup-margins 20,20 \
  --export edl,json,srt
```

---

### 9) GUI

- "Quick Transcribe" panel: checkboxes — "Remove Sighs", "Remove Filler Sounds", "Remove Filler Words".
- Action selector: "Attenuate / Cut".
- Expandable parameters: thresholds, durations, crossfade, limits.
- Preview: interval highlighting; context menu "Undo edit for this location".

---

### 10) Quality Assessment and Automation

- Auto-labeling: positive examples generated from transcript (dictionary) and heuristics (sighs/sounds). Only 30–60 minutes of manual audit needed for low-confidence cases.
- Metrics: precision/recall per type, total edit fraction, artifact count (manual-revert counter in GUI).
- CI gates: build fails if metrics drop below threshold.

---

### 11) Datasets and Sources

- Russian speech corpora without disfluencies: `OpenSTT`, `Golos`, `Common Voice (ru)` — suitable for acoustic calibrations/thresholds.
- Universal audio classes: pretrained PANNs on `AudioSet` with breathing/sigh classes — usable for breath validation without training.
- Custom auto-labeling from aligned text provides a cold start without manual annotation.

Note: respect dataset licenses. Secrets (e.g. tokens) must be stored in environment variables (`YOUR_HF_TOKEN`).

---

### 12) Alternative Approaches

1) DAW plugins (iZotope RX Breath Control, Waves DeBreath):
   - Pros: quality breath processing out of the box.
   - Cons: proprietary, no word-level edits, harder to automate in our pipeline.

2) Contextual disfluency model (sequence tagging, RuBERT-class):
   - Pros: better at distinguishing semantic cases ("да" as a reply vs. disfluency).
   - Cons: requires training set and inference infrastructure; later as Stage 4.

3) YAMNet instead of PANNs:
   - Pros: ready-made embeddings/classes.
   - Cons: TensorFlow dependency; project is in Torch ecosystem — not chosen.

4) Cloud editors (Descript-like):
   - Pros: ready-made "Remove filler words".
   - Cons: cloud-based, often EN-focused, no offline guarantees or open API for our needs.

5) Spectral masking instead of cutting (dynamic EQ/spectral gate):
   - Pros: minimal splices, suitable for filler sounds and breathing over a word.
   - Cons: possible tonal naturalness degradation; leaves a noise trace.

---

### 13) Risks and Mitigations

- Splices/clicks at cuts → default: attenuate; for cut — short crossfades.
- Semantic words ("да/вот/ну") → strict mode, allowlists, rollback metrics in GUI.
- Complex noisy environments → adaptive thresholds, fallback to attenuate.
- Performance → CPU-compatible; PANNs enabled optionally; batch sizes are manageable.

---

### 14) Implementation Plan (stages)

- Stage 1 (MVP):
  - FillerWordsDetector via dictionary + alignment; Cutlist JSON; action: attenuate.
  - CLI/GUI flags; basic metrics.

- Stage 2:
  - FillerSoundsDetector: tokens + acoustic heuristics; EDL export; preview in GUI.

- Stage 3:
  - BreathsDetector: heuristics + optional PANNs; threshold fine-tuning.

- Stage 4:
  - Contextual disfluency model (RuBERT), white/black-lists, metrics improvement.

---

### 15) Interfaces (proposal)

```python
# editing/disfluency.py
from dataclasses import dataclass
from typing import List, Literal

DisfluencyType = Literal["filler_word", "filler_sound", "sigh"]
ActionType = Literal["attenuate", "cut"]

@dataclass
class CutItem:
    start_ms: int
    end_ms: int
    type: DisfluencyType
    confidence: float
    action: ActionType
    margin_before_ms: int = 20
    margin_after_ms: int = 20
    tags: List[str] | None = None

def build_cutlist(
    alignment_words: list,
    waveform_path: str,
    settings: dict,
) -> list[CutItem]:
    """* Produces a unified cutlist from detectors using alignment and audio."""
    ...

def apply_cutlist(
    input_wav: str,
    cutlist: list[CutItem],
    output_wav: str,
) -> None:
    """* Applies attenuate or cut+crossfade via FFmpeg wrapper safely."""
    ...
```

```python
# core/pipelines.py (fragment, API concept)
def run_disfluency_cleanup(self, alignment, wav_path, settings) -> dict:
    """* Runs detectors, returns { cutlist, metrics } and optional processed wav."""
    ...
```

---

### 16) Security and Secrets

- Do not store keys/tokens in code. Use environment variables (`YOUR_HF_TOKEN`, `YOUR_MODEL_PATH`).
- Add appropriate placeholders and ensure `.env`/secrets are in `.gitignore`.

---

### 17) Performance Notes

- All stages run on CPU; GPU acceleration is optional.
- PANNs enabled only on user request.
- Process in large windows with 10–20 ms step; batch-orient where possible.
