from __future__ import annotations

import argparse
import importlib
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_MODEL_NAME = "small"
_DEFAULT_DEVICE = "cuda"
_DEFAULT_COMPUTE_TYPE = "auto"

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _load_runtime_components() -> tuple[object, object, object]:
    """Load project modules after adding the repository root to ``sys.path``."""
    prepare_audio = importlib.import_module("core.audio_io").prepare_audio
    get_project_cache_dir = importlib.import_module(
        "core.settings"
    ).get_project_cache_dir
    whisper_cls = importlib.import_module("core.whisperx_wrapper").WhisperXWrapper
    return prepare_audio, get_project_cache_dir, whisper_cls


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a real Whisper transcribe + aggressive unload in a child process."
        )
    )
    parser.add_argument(
        "media_path", type=Path, help="Path to the committed short clip."
    )
    parser.add_argument("work_dir", type=Path, help="Directory for prepared audio.")
    parser.add_argument(
        "--model-name",
        default=_DEFAULT_MODEL_NAME,
        help="Whisper model name to load before aggressive unload.",
    )
    return parser


def main() -> int:
    """Run the real transcribe + aggressive unload sequence in a standalone process."""
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    prepare_audio, get_project_cache_dir, whisper_cls = _load_runtime_components()
    logger = logging.getLogger(__name__)
    media_path = args.media_path.resolve()
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    logger.info("child_media=%s", media_path)
    logger.info("child_work_dir=%s", work_dir)
    logger.info("child_model_name=%s", args.model_name)
    prepared_audio = prepare_audio(media_path, work_dir)
    logger.info("prepared_audio=%s", prepared_audio)
    whisper = whisper_cls(
        model_name=str(args.model_name),
        device=_DEFAULT_DEVICE,
        compute_type=_DEFAULT_COMPUTE_TYPE,
        model_root=get_project_cache_dir() / "models",
    )
    transcription = whisper.transcribe(prepared_audio, language=None)
    transcript_text = str(transcription.get("text", "")).strip()
    segments = transcription.get("segments", []) or []
    logger.info("segment_count=%d", len(segments))
    logger.info("transcript_chars=%d", len(transcript_text))
    logger.info("before aggressive unload")
    whisper.unload(safe=False)
    logger.info("aggressive_unload_complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
