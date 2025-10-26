# Placeholder for main.py
import argparse
from pathlib import Path

from core.pipelines import LocalPipeline

# * Command-line interface for local processing pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Artemonim's Speech Kit - Local Processing CLI"
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to input media file (audio/video) or directory",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=".",
        help="Working directory for intermediate files and output",
    )
    parser.add_argument(
        "--whisper-model",
        default="large-v3",
        help="Name of Whisper model to use (default: large-v3)",
    )
    parser.add_argument(
        "--llm-model",
        default="gguf-q4_0",
        help="Name of LLM model for formatting (default: gguf-q4_0)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input path not found: {input_path}")
        return

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = LocalPipeline(
        model_root=None,
        whisper_model=args.whisper_model,
        llm_model=args.llm_model,
    )

    if input_path.is_dir():
        media_files = list(input_path.glob("*.*"))
    else:
        media_files = [input_path]

    for media in media_files:
        print(f"Processing {media}...")
        doc = pipeline.process(media, output_dir)
        text = doc.get_full_text()
        output_file = output_dir / f"{media.stem}_transcript.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Saved transcript to {output_file}")


if __name__ == "__main__":
    main()
