"""Synthetic inputs of the ``memory-growth`` scenario (never real user audio).

A *source* speech recording is either synthesized with the Windows SAPI voice
``Microsoft Pavel`` through PowerShell 7 (the voice the repository's heavy
tests use) or taken from a file given on the command line (for example the
repository fixture ``tests/fixtures/test_video_short.mp4``). The source is
repeated end to end into one PCM WAV long enough for the longest clip after
any start offset (FFmpeg's ``-stream_loop`` breaks the timestamps at the loop
seam and lengthens clips that cross it); every job input is then cut from it
with FFmpeg at a seeded random offset within the first repetition, trimmed to
its planned duration and encoded like a Telegram voice note (Ogg Opus, 16 kHz
mono) or as a 16 kHz mono PCM WAV.

Durations come from a :class:`DurationPlan`: ``varied`` draws every duration
uniformly from ``[min_s, max_s]`` with a fixed seed (millisecond resolution,
so practically every job has its own length, like production voice notes);
``fixed`` gives every job the same duration (only the content differs), which
keeps every model input shape constant.
"""

from __future__ import annotations

import math
import random
import shutil
import subprocess
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .workload import WorkloadError

if TYPE_CHECKING:
    from collections.abc import Sequence

# * User-tunable defaults.
DURATION_VARIED = "varied"
DURATION_FIXED = "fixed"
DURATION_MODES = (DURATION_VARIED, DURATION_FIXED)
AUDIO_OPUS = "opus"
AUDIO_WAV = "wav"
AUDIO_FORMATS = (AUDIO_OPUS, AUDIO_WAV)
SOURCE_SAPI = "sapi"
SAPI_VOICE = "Microsoft Pavel"
# * Neutral Russian narration, about 40 s with the SAPI voice; clips loop it.
SAPI_TEXT = (
    "Добрый день. Сегодня в городе облачно, к вечеру обещают небольшой дождь, "
    "поэтому возьмите зонт. Курьеры начинают смену в девять утра и "
    "заканчивают в девять вечера. Если заказ задерживается, напишите в "
    "поддержку, и вам ответят в течение пяти минут. Новые правила оплаты "
    "начнут действовать первого числа следующего месяца. Пожалуйста, проверяйте "
    "адрес доставки перед тем, как подтвердить заказ. Спасибо за работу, "
    "хорошего дня и спокойной дороги."
)
SAMPLE_RATE = 16000
OPUS_BITRATE = "32k"
# * Shortest accepted clip: GigaAM needs at least 640 samples per window.
MIN_DURATION_S = 0.5
MAX_DURATION_S = 3600.0
MIN_SOURCE_S = 1.0
FFMPEG_TIMEOUT_S = 120.0
SAPI_TIMEOUT_S = 120.0
CLIP_WORKERS = 4
# * Offsets use their own stream, so a duration plan yields the same durations
# * whatever the source length.
_OFFSET_SEED_SALT = 7919
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_DETAIL_CHARS = 500


@dataclass(frozen=True, slots=True)
class DurationPlan:
    """How the job durations of a run are chosen.

    Attributes:
        mode: ``varied`` (uniform over ``[min_s, max_s]``) or ``fixed``.
        min_s: Shortest duration of ``varied``.
        max_s: Longest duration of ``varied``.
        fixed_s: Duration of every job of ``fixed``.
        seed: Seed of the duration and offset draws.

    """

    mode: str
    min_s: float
    max_s: float
    fixed_s: float
    seed: int

    def validate(self) -> None:
        """Check the plan.

        Raises:
            WorkloadError: For an unknown mode or durations outside the bounds.

        """
        if self.mode not in DURATION_MODES:
            msg = f"--durations must be one of {', '.join(DURATION_MODES)}"
            raise WorkloadError(msg)
        bounds = (
            (self.fixed_s,) if self.mode == DURATION_FIXED else (self.min_s, self.max_s)
        )
        for value in bounds:
            if not MIN_DURATION_S <= value <= MAX_DURATION_S:
                msg = (
                    f"durations must lie in [{MIN_DURATION_S:g}, {MAX_DURATION_S:g}] s "
                    f"(got {value:g})"
                )
                raise WorkloadError(msg)
        if self.mode == DURATION_VARIED and self.min_s > self.max_s:
            msg = (
                f"--duration-min-s ({self.min_s:g}) exceeds --duration-max-s "
                f"({self.max_s:g})"
            )
            raise WorkloadError(msg)

    def durations(self, count: int) -> list[float]:
        """Return the planned duration of each of *count* jobs, in seconds."""
        if self.mode == DURATION_FIXED:
            return [round(self.fixed_s, 3)] * count
        # * A reproducible workload, not a security token.
        rng = random.Random(self.seed)  # noqa: S311
        return [round(rng.uniform(self.min_s, self.max_s), 3) for _ in range(count)]

    def describe(self) -> dict[str, object]:
        """Return the plan for the run metadata."""
        if self.mode == DURATION_FIXED:
            return {"mode": self.mode, "fixed_s": self.fixed_s, "seed": self.seed}
        return {
            "mode": self.mode,
            "min_s": self.min_s,
            "max_s": self.max_s,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class InputClip:
    """One generated job input.

    Attributes:
        index: 1-based job number.
        path: Absolute path of the clip.
        duration_s: Planned duration.
        offset_s: Start offset in the looped source.

    """

    index: int
    path: Path
    duration_s: float
    offset_s: float


def _require_tool(name: str, purpose: str) -> str:
    """Return the path of an executable on ``PATH``.

    Raises:
        WorkloadError: When the executable is missing.

    """
    path = shutil.which(name)
    if path is None:
        msg = f"{name} is required on PATH to {purpose}"
        raise WorkloadError(msg)
    return path


def _run_tool(argv: Sequence[str], *, timeout_s: float, what: str) -> None:
    """Run an external tool and turn its failure into :class:`WorkloadError`.

    Raises:
        WorkloadError: When the tool cannot start, fails or times out.

    """
    try:
        subprocess.run(  # noqa: S603 - resolved executables, argv built here
            list(argv),
            check=True,
            capture_output=True,
            timeout=timeout_s,
            creationflags=_NO_WINDOW,
        )
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace").strip()[-_DETAIL_CHARS:]
        msg = f"{what} failed (exit {exc.returncode}): {detail}"
        raise WorkloadError(msg) from exc
    except subprocess.TimeoutExpired as exc:
        msg = f"{what} timed out after {timeout_s:g} s"
        raise WorkloadError(msg) from exc
    except OSError as exc:
        msg = f"{what} could not start: {exc}"
        raise WorkloadError(msg) from exc


def _ps_quote(value: str) -> str:
    """Quote *value* as a single-quoted PowerShell string."""
    return "'" + value.replace("'", "''") + "'"


def synthesize_sapi(
    path: Path, *, voice: str = SAPI_VOICE, text: str = SAPI_TEXT
) -> None:
    """Synthesize *text* with a SAPI voice into a 16 kHz mono 16-bit WAV.

    Args:
        path: Output WAV.
        voice: Installed SAPI voice (``Microsoft Pavel`` is visible to
            PowerShell 7 only).
        text: Text to speak.

    Raises:
        WorkloadError: When PowerShell 7 is missing or the synthesis fails.

    """
    pwsh = _require_tool("pwsh", "synthesize the SAPI source (or pass --source FILE)")
    script = (
        "$ErrorActionPreference = 'Stop';"
        "Add-Type -AssemblyName System.Speech;"
        "$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        f"try {{ $speaker.SelectVoice({_ps_quote(voice)});"
        f"$speaker.SetOutputToWaveFile({_ps_quote(str(path))}, "
        "(New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo"
        f"({SAMPLE_RATE}, 'Sixteen', 'Mono')));"
        f"$speaker.Speak({_ps_quote(text)}) }} finally {{ $speaker.Dispose() }}"
    )
    _run_tool(
        [pwsh, "-NoProfile", "-NonInteractive", "-Command", script],
        timeout_s=SAPI_TIMEOUT_S,
        what=f"SAPI synthesis with {voice!r}",
    )


def wav_seconds(path: Path) -> float:
    """Return the duration of a PCM WAV from its header.

    Raises:
        WorkloadError: When the file is not a readable WAV.

    """
    try:
        with wave.open(str(path), "rb") as reader:
            rate = reader.getframerate()
            frames = reader.getnframes()
    except (OSError, EOFError, wave.Error) as exc:
        msg = f"cannot read WAV {path}: {exc}"
        raise WorkloadError(msg) from exc
    return frames / rate if rate > 0 else 0.0


def build_source(source: str, work_dir: Path) -> Path:
    """Return the 16 kHz mono WAV every clip is cut from.

    Args:
        source: ``sapi`` (synthesize) or the path of an audio or video file.
        work_dir: Directory for the source WAV.

    Returns:
        The source WAV inside *work_dir*.

    Raises:
        WorkloadError: When the source cannot be built or is shorter than
            :data:`MIN_SOURCE_S`.

    """
    work_dir.mkdir(parents=True, exist_ok=True)
    target = work_dir / "source.wav"
    if source == SOURCE_SAPI:
        synthesize_sapi(target)
    else:
        given = Path(source)
        if not given.is_file():
            msg = f"--source must be {SOURCE_SAPI!r} or an existing file: {source}"
            raise WorkloadError(msg)
        ffmpeg = _require_tool("ffmpeg", "decode the --source file")
        _run_tool(
            [
                ffmpeg,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(given),
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(SAMPLE_RATE),
                "-c:a",
                "pcm_s16le",
                str(target),
            ],
            timeout_s=FFMPEG_TIMEOUT_S,
            what=f"decoding {given.name}",
        )
    seconds = wav_seconds(target)
    if seconds < MIN_SOURCE_S:
        msg = f"the source is too short ({seconds:.2f} s < {MIN_SOURCE_S:g} s)"
        raise WorkloadError(msg)
    return target


def _codec_args(audio_format: str) -> tuple[list[str], str]:
    """Return the FFmpeg codec arguments and the file suffix of a clip format.

    Raises:
        WorkloadError: For an unknown format.

    """
    if audio_format == AUDIO_OPUS:
        return ["-c:a", "libopus", "-b:a", OPUS_BITRATE, "-application", "voip"], ".ogg"
    if audio_format == AUDIO_WAV:
        return ["-c:a", "pcm_s16le"], ".wav"
    msg = f"--audio-format must be one of {', '.join(AUDIO_FORMATS)}"
    raise WorkloadError(msg)


def repeat_wav(source_wav: Path, target: Path, min_seconds: float) -> float:
    """Write *source_wav* repeated end to end until it lasts at least *min_seconds*.

    Args:
        source_wav: PCM WAV to repeat.
        target: Output WAV (same format).
        min_seconds: Shortest duration of the output.

    Returns:
        The duration of *target* in seconds.

    Raises:
        WorkloadError: When the source cannot be read or the target written.

    """
    try:
        with wave.open(str(source_wav), "rb") as reader:
            params = reader.getparams()
            frames = reader.readframes(reader.getnframes())
        source_s = params.nframes / params.framerate if params.framerate > 0 else 0.0
        if source_s <= 0:
            msg = f"the source {source_wav} holds no audio"
            raise WorkloadError(msg)
        copies = max(1, math.ceil(min_seconds / source_s))
        with wave.open(str(target), "wb") as writer:
            writer.setparams(params)
            for _ in range(copies):
                writer.writeframes(frames)
    except (OSError, EOFError, wave.Error) as exc:
        msg = f"cannot repeat {source_wav} into {target}: {exc}"
        raise WorkloadError(msg) from exc
    return copies * source_s


def _cut_clip(
    ffmpeg: str, source_wav: Path, clip: InputClip, codec: Sequence[str]
) -> None:
    """Cut one clip from the repeated source (input seeking is sample exact)."""
    _run_tool(
        [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{clip.offset_s:.3f}",
            "-i",
            str(source_wav),
            "-t",
            f"{clip.duration_s:.3f}",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            *codec,
            str(clip.path),
        ],
        timeout_s=FFMPEG_TIMEOUT_S,
        what=f"cutting clip {clip.path.name}",
    )


def plan_clips(
    plan: DurationPlan, count: int, *, source_s: float, out_dir: Path, suffix: str
) -> list[InputClip]:
    """Return the clips of a run (durations, offsets and paths; nothing written).

    Args:
        plan: Duration plan.
        count: Number of jobs.
        source_s: Duration of the source WAV.
        out_dir: Directory of the clips.
        suffix: File suffix of the clip format.

    Returns:
        One clip per job, numbered from 1.

    """
    # * A reproducible workload, not a security token.
    offsets = random.Random(plan.seed + _OFFSET_SEED_SALT)  # noqa: S311
    clips: list[InputClip] = []
    for index, duration in enumerate(plan.durations(count), start=1):
        offset = round(offsets.uniform(0.0, source_s), 3)
        name = f"job-{index:04d}-{duration:07.3f}s{suffix}"
        clips.append(InputClip(index, out_dir / name, duration, offset))
    return clips


def prepare_inputs(
    plan: DurationPlan,
    count: int,
    *,
    source: str,
    work_dir: Path,
    audio_format: str,
) -> list[InputClip]:
    """Build the source and cut every job input into *work_dir*.

    Args:
        plan: Duration plan.
        count: Number of jobs.
        source: ``sapi`` or the path of a speech file.
        work_dir: Directory of the source and the clips (created).
        audio_format: ``opus`` or ``wav``.

    Returns:
        The clips in job order.

    Raises:
        WorkloadError: When a tool is missing or a clip cannot be cut.

    """
    plan.validate()
    codec, suffix = _codec_args(audio_format)
    ffmpeg = _require_tool("ffmpeg", "cut the job inputs")
    source_wav = build_source(source, work_dir)
    source_s = wav_seconds(source_wav)
    clips = plan_clips(plan, count, source_s=source_s, out_dir=work_dir, suffix=suffix)
    # * Offsets lie in the first repetition, so the longest clip plus one source
    # * length (and a second of slack) always fits without a loop seam.
    longest = max((clip.duration_s for clip in clips), default=0.0)
    repeated = work_dir / "repeated.wav"
    repeat_wav(source_wav, repeated, source_s + longest + 1.0)
    with ThreadPoolExecutor(max_workers=CLIP_WORKERS) as pool:
        # * list() re-raises the first clip failure.
        list(pool.map(lambda clip: _cut_clip(ffmpeg, repeated, clip, codec), clips))
    return clips
