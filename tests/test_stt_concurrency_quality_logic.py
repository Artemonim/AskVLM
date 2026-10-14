"""Dense-logic tests of the sweep stop rules, knees, WER, and cross-file batching.

Expected values are derived by hand from the documented rules: WER is the word
edit distance summed over items divided by the summed reference words; the sweep
stops on a drop of more than 10 % below the best step, on two consecutive steps
with less than 5 % gain, on a WER increase above 0.5 pp, or on a mean similarity
below 0.97; the efficient knee is the smallest N within 5 % of the best.
"""

from __future__ import annotations

import json
import types
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

# * The support module puts `tools/` on sys.path for the imports below.
from tests import stt_concurrency_support as support

# isort: split

from stt_concurrency import runner
from stt_concurrency.combo import auto_combos, parse_combo
from stt_concurrency.configs import (
    ConfigError,
    ResourceEstimate,
    ThreadDefaults,
    parse_config_name,
)
from stt_concurrency.engines import WhisperEngine
from stt_concurrency.protocol import ChildSpec
from stt_concurrency.quality import (
    ManifestError,
    compare_items,
    corpus_wer,
    load_manifest,
    normalize_for_wer,
    repeat_consistency,
    word_errors,
)
from stt_concurrency.store import ResultStore, fingerprint_of
from stt_concurrency.sweep import StopRules, evaluate_stop, knees
from stt_concurrency.sysmetrics import CommitInfo, GpuSample
from stt_concurrency.workload import AudioItem, WorkStream

if TYPE_CHECKING:
    from pathlib import Path

DEFAULTS = ThreadDefaults(cpu_budget=16, cuda_threads=4)
RULES = StopRules()
GIB = 1024**3


# --- WER ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "words"),
    [
        ("Ёлка, ёж!", ["елка", "еж"]),
        ("ЁЖ", ["еж"]),
        ("из-за  угла.", ["из", "за", "угла"]),
        ("Hello,   World 42", ["hello", "world", "42"]),
        ("  ...  ", []),
    ],
)
def test_wer_normalization(text: str, words: list[str]) -> None:
    """Catches: punctuation, case, or yo/ye differences counted as word errors."""
    assert normalize_for_wer(text) == words


@pytest.mark.parametrize(
    ("ref", "hyp", "expected"),
    [
        ("a b c", "a c", 1),  # one deletion
        ("a b", "a x b", 1),  # one insertion
        ("a b c", "a x c", 1),  # one substitution
        ("a b", "b a", 2),  # no transposition shortcut
        ("", "x y", 2),
        ("a b", "", 2),
        ("a b c d", "a b c d", 0),
    ],
)
def test_word_errors(ref: str, hyp: str, expected: int) -> None:
    """Catches: an edit distance that is not the sub+ins+del minimum."""
    assert word_errors(ref.split(), hyp.split()) == expected


def _manifest(tmp_path: Path, records: list[dict[str, str]]) -> Any:  # noqa: ANN401
    path = tmp_path / "manifest.jsonl"
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )
    return load_manifest(path)


def test_manifest_prefers_normalized_reference_and_keys_by_file_name(
    tmp_path: Path,
) -> None:
    """Catches: scoring against the raw reference or the wrong file."""
    manifest = _manifest(
        tmp_path,
        [
            {
                "wav": "dir/a.wav",
                "ref_raw": "Raw, TEXT.",
                "ref_fleurs_norm": "norm text",
            },
            {"wav": "b.wav", "ref_raw": "Ёж идёт."},
        ],
    )

    assert manifest.reference_for("a.wav") == ("norm", "text")
    assert manifest.reference_for("b.wav") == ("еж", "идет")
    assert manifest.reference_for("dir/a.wav") is None


@pytest.mark.parametrize(
    "content",
    ["", "{not json}\n", '{"wav": "a.wav"}\n', '{"ref_raw": "x"}\n'],
)
def test_bad_manifest_is_rejected(tmp_path: Path, content: str) -> None:
    """Catches: an unusable manifest silently producing an empty WER column."""
    path = tmp_path / "m.jsonl"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ManifestError):
        load_manifest(path)


def test_corpus_wer_is_word_weighted_and_skips_failed_items(tmp_path: Path) -> None:
    """Catches: a mean of per-item WERs, or failed items counted as 100 % errors."""
    manifest = _manifest(
        tmp_path,
        [
            {"wav": "a.wav", "ref_raw": "один два три"},
            {"wav": "b.wav", "ref_raw": "четыре"},
            {"wav": "c.wav", "ref_raw": "пять шесть"},
        ],
    )
    items = [
        {"name": "a.wav", "text": "Один, три", "error": None},  # 1 deletion of 3
        {"name": "b.wav", "text": "пять", "error": None},  # 1 substitution of 1
        {"name": "c.wav", "text": "", "error": "boom"},  # excluded
        {"name": "zzz.wav", "text": "x", "error": None},  # no reference
    ]

    result = corpus_wer(items, manifest)

    # * (1 + 1) errors / (3 + 1) words = 50 %, not the mean of 33.3 and 100.
    assert result["wer_percent"] == pytest.approx(50.0)
    assert result["ref_words"] == 4
    assert result["scored_items"] == 2
    assert result["items_without_reference"] == 1


def _item(clip: int, text: str, error: str | None = None) -> dict[str, Any]:
    return {
        "index": clip,
        "clip": clip,
        "name": f"c{clip}.wav",
        "text": text,
        "error": error,
    }


def test_compare_items_uses_first_occurrence_on_common_clips(tmp_path: Path) -> None:
    """Catches: WER delta distorted by repeats, extra clips, or failed items."""
    manifest = _manifest(
        tmp_path,
        [
            {"wav": "c0.wav", "ref_raw": "один два три"},
            {"wav": "c1.wav", "ref_raw": "четыре пять"},
            {"wav": "c2.wav", "ref_raw": "шесть"},
            {"wav": "c3.wav", "ref_raw": "семь"},
        ],
    )
    reference = [
        _item(0, "один два три"),
        _item(1, "четыре пять"),
        _item(2, "шесть"),
    ]
    step = [
        _item(0, "один два три"),
        _item(1, "четыре"),  # one deletion
        _item(2, "семь"),  # one substitution
        _item(0, "один"),  # repeat: ignored by the WER delta, counted by identity
        _item(3, "семь"),  # not transcribed by the reference: ignored
    ]

    result = compare_items(step, reference, manifest)

    # * Common clips 0..2 hold 6 reference words with 2 errors in the step.
    assert result["wer_common_items"] == 3
    assert result["wer_common_percent"] == pytest.approx(100.0 * 2 / 6)
    assert result["wer_reference_common_percent"] == 0.0
    assert result["wer_delta_pp"] == pytest.approx(100.0 * 2 / 6)
    # * Identity counts the four occurrences of reference clips: only clip 0 matches.
    assert result["compared"] == 4
    assert result["identical_percent"] == pytest.approx(25.0)


def test_compare_items_without_common_clips_reports_nothing() -> None:
    """Catches: a comparison of disjoint clips reported as 100 % identical."""
    assert compare_items([_item(5, "x")], [_item(1, "x")], None) == {"compared": 0}


def test_repeat_consistency() -> None:
    """Catches: races between concurrent calls going unnoticed."""
    items = [
        _item(0, "один"),
        _item(1, "два"),
        _item(0, "один"),
        _item(0, "три"),
        _item(1, "два", error="boom"),
    ]

    assert repeat_consistency(items) == pytest.approx(50.0)
    assert repeat_consistency([_item(0, "один"), _item(1, "два")]) is None


# --- stop rules ---------------------------------------------------------------


def _step(
    width: int,
    xrt: float | None,
    **extra: Any,  # noqa: ANN401
) -> dict[str, Any]:
    status = extra.pop("status", "ok")
    return {
        "name": f"cfg-{width}",
        "width": width,
        "status": status,
        "throughput_x": xrt,
        "error_count": 0,
        "reason": None,
        "cuda_oom": False,
        "wer_delta_pp": None,
        "similarity_mean": None,
        **extra,
    }


def _history(*xrts: float) -> list[dict[str, Any]]:
    return [_step(i + 1, xrt) for i, xrt in enumerate(xrts)]


@pytest.mark.parametrize(
    ("xrts", "scan", "expected"),
    [
        ([10, 20, 17.9], True, "regression"),  # 17.9 < 20 * 0.9
        ([10, 20, 18.0], True, None),  # exactly 10 % below is still allowed
        ([10, 10.4, 10.7], True, "plateau"),  # two steps below +5 % of the best
        ([10, 10.4, 11.0], True, None),  # 11.0 is +5.8 % over 10.4
        ([10, 10.4], True, None),  # one flat step is not a plateau yet
        ([10], True, None),
        ([10, 5], False, None),  # budget groups have hard stops only
        ([10, 10.1, 10.2], False, None),
    ],
)
def test_throughput_stop_rules(
    xrts: list[float], *, scan: bool, expected: str | None
) -> None:
    """Catches: wrong knee/plateau/regression thresholds in the saturation search."""
    reason = evaluate_stop(_history(*xrts), RULES, scan=scan)

    if expected is None:
        assert reason is None
    else:
        assert reason is not None
        assert reason.startswith(expected)


@pytest.mark.parametrize(
    ("extra", "expected"),
    [
        ({"wer_delta_pp": 0.5}, None),  # the limit itself is allowed
        ({"wer_delta_pp": 0.51}, "quality"),
        ({"similarity_mean": 0.97}, None),
        ({"similarity_mean": 0.969}, "quality"),
        ({"status": "skipped", "throughput_x": None, "reason": "no memory"}, "guard"),
        ({"status": "timeout", "throughput_x": None}, "timeout"),
        ({"status": "partial", "error_count": 2}, "errors"),
        ({"status": "error", "cuda_oom": True}, "cuda_oom"),
        ({"status": "partial", "error_count": 1, "cuda_oom": True}, "cuda_oom"),
    ],
)
@pytest.mark.parametrize("scan", [True, False])
def test_hard_and_quality_stops(
    extra: dict[str, Any], expected: str | None, *, scan: bool
) -> None:
    """Catches: a failing or degraded step that lets the sweep continue."""
    history = [_step(1, 10.0), _step(2, 19.0, **extra)]

    reason = evaluate_stop(history, RULES, scan=scan)

    if expected is None:
        assert reason is None
    else:
        assert reason is not None
        assert reason.startswith(expected)


def test_knees_best_and_efficient() -> None:
    """Catches: the efficient step chosen by throughput instead of smallest N."""
    history = _history(10.0, 19.5, 20.0, 20.5)

    result = knees(history, 0.05)

    assert result["best"] == "cfg-4"
    # * 20.5 * 0.95 = 19.475: N=2 (19.5) is the smallest step inside the band.
    assert result["efficient"] == "cfg-2"
    assert result["efficient_xrt"] == 19.5


def test_knees_ignore_failed_steps() -> None:
    """Catches: an OOM or timed-out step counted as the best one."""
    history = [_step(1, 10.0), _step(2, 12.0), _step(3, 99.0, status="error")]

    assert knees(history, 0.05)["best"] == "cfg-2"
    assert knees([_step(1, None, status="error")], 0.05) == {
        "best": None,
        "efficient": None,
    }


# --- work stream ------------------------------------------------------------


def _clips(count: int) -> list[AudioItem]:
    from pathlib import Path as _Path  # noqa: PLC0415

    return [AudioItem(i, _Path(f"c{i}.wav"), 1.0 + i) for i in range(count)]


def test_work_stream_cycles_until_min_items_and_min_time() -> None:
    """Catches: an uneven or short workload that makes steps incomparable."""
    stream = WorkStream(_clips(3), min_items=5, min_wall_s=2.0, max_items=100)

    first = stream.take(4, 0.0)
    second = stream.take(4, 0.5)
    late = stream.take(4, 2.0)

    assert [o.clip.index for o in first] == [0, 1, 2, 0]
    assert [o.cycle for o in first] == [0, 0, 0, 1]
    assert [o.item_id for o in first + second] == list(range(8))
    # * Past min_items the stream goes on until min_wall_s elapsed, then closes.
    assert late == []
    assert stream.closed
    assert stream.take(1, 0.0) == []


def test_work_stream_respects_max_items_and_reports_owed_items() -> None:
    """Catches: unbounded steps; aborted steps hiding the items they still owed."""
    capped = WorkStream(_clips(2), min_items=1, min_wall_s=1e9, max_items=3)
    owed = WorkStream(_clips(2), min_items=4, min_wall_s=0.0, max_items=4)

    assert len(capped.take(10, 0.0)) == 3
    assert owed.take(1, 0.0)[0].item_id == 0
    assert [o.item_id for o in owed.planned()] == [0, 1, 2, 3]
    assert len(owed.produced) == 1


# --- fingerprints and store ---------------------------------------------------


def _fingerprint(tmp_path: Path, **changes: Any) -> str:  # noqa: ANN401
    options = support.make_run_options(tmp_path, **changes.pop("options", {}))
    clips = changes.pop("clips", None) or support.make_audio_items(tmp_path, [1, 2])
    workload = runner.Workload(
        clips=tuple(clips),
        long_items=(),
        warmup_path=clips[0].path,
        manifest=changes.pop("manifest", None),
    )
    return fingerprint_of(runner.fingerprint_payload(options, workload))


def test_fingerprint_follows_the_inputs_that_shape_a_measurement(
    tmp_path: Path,
) -> None:
    """Catches: results silently reused (or discarded) after the inputs changed."""
    base = _fingerprint(tmp_path)

    assert _fingerprint(tmp_path) == base
    # * Inputs that do not change the measurement keep the fingerprint.
    assert _fingerprint(tmp_path, options={"cooldown_s": 99.0, "dry_run": True}) == base
    assert _fingerprint(tmp_path, options={"retry_failed": True}) == base
    changed = [
        {"max_clips": 7},
        {"language": "en"},
        {"whisper_model": "base"},
        {"whisper_options": {"beam_size": 1}},
        {"whisper_chunk_s": 20.0},
        {"gigaam_cuda_streams": False},
        {"thread_defaults": ThreadDefaults(cpu_budget=8, cuda_threads=4)},
        {"thread_defaults": ThreadDefaults(cpu_budget=16, cuda_threads=2)},
    ]
    seen = {base}
    for change in changed:
        value = _fingerprint(tmp_path, options=change)
        assert value not in seen, change
        seen.add(value)
    longer = support.make_audio_items(tmp_path, [1, 3])
    assert _fingerprint(tmp_path, clips=longer) not in seen


def test_result_store_reuses_only_matching_final_results(tmp_path: Path) -> None:
    """Catches: reuse of foreign or non-final results; stale results not archived."""
    old = ResultStore(tmp_path / "results", "fp-old")
    new = ResultStore(tmp_path / "results", "fp-new")
    old.save("a", {"status": "ok"})
    old.save("b", {"status": "skipped"})

    assert old.reusable("a", statuses={"ok"}) is not None
    assert old.reusable("b", statuses={"ok"}) is None
    assert old.reusable("missing", statuses={"ok"}) is None
    assert new.reusable("a", statuses={"ok"}) is None
    assert new.describe("a").startswith("stale")
    assert [d["status"] for d in new.current()] == []

    new.save("a", {"status": "ok"})

    stale = [p.name for p in (tmp_path / "results" / "stale").glob("*.json")]
    assert stale == ["a__fp-old.json"]
    assert new.reusable("a", statuses={"ok"}) is not None
    assert [d["fingerprint"] for d in new.current()] == ["fp-new"]


# --- guard of a pair ----------------------------------------------------------


def test_pair_guard_sums_both_sides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches: a combo that passes the guard because each side fits alone."""
    estimates = {
        **support.default_estimates(),
        "whisper-cpu": ResourceEstimate(3.0, 0.0, 0.0, 0.0),
        "gigaam-cuda": ResourceEstimate(3.0, 2.0, 0.0, 0.0),
    }
    options = support.make_run_options(
        tmp_path, estimates=estimates, min_free_commit_gb=5.0, min_free_vram_mb=100.0
    )
    monkeypatch.setattr(
        runner, "query_commit", lambda: CommitInfo(10 * GIB, None, None, "test")
    )
    monkeypatch.setattr(
        runner, "query_gpu", lambda _exe: GpuSample(0.0, 6000.0, 8192.0, 0.0)
    )
    cpu = parse_config_name("whisper-cpu-P1x4", DEFAULTS)
    gpu = parse_config_name("gigaam-cuda-P1x4", DEFAULTS)

    alone_cpu, _r, _d = runner.evaluate_guard_configs([cpu], options, "smi")
    alone_gpu, _r, _d = runner.evaluate_guard_configs([gpu], options, "smi")
    both, reason, details = runner.evaluate_guard_configs([cpu, gpu], options, "smi")

    # * Each side leaves 7 GB of commit; together 6 GB are needed: 4 GB < 5 GB.
    assert alone_cpu
    assert alone_gpu
    assert not both
    assert reason is not None
    assert details["need_commit_gb"] == 6.0
    two_gpu, _r, _d = runner.evaluate_guard_configs([gpu, gpu], options, "smi")
    # * Free VRAM 2192 MB - 4096 MB needed: refused although one fits (192 MB).
    assert not two_gpu


# --- combos -------------------------------------------------------------------


def test_parse_combo() -> None:
    """Catches: malformed or unsupported pairs reaching the measurement."""
    first, second = parse_combo("gigaam-cuda-B16 + gigaam-cpu-S8x2", DEFAULTS)

    assert (first.name, second.name) == ("gigaam-cuda-B16x4", "gigaam-cpu-S8x2")
    for bad in ("gigaam-cuda-B16", "a+b+c", "gigaam-cuda-B16+", "x-cpu-P1+y"):
        with pytest.raises(ConfigError):
            parse_combo(bad, DEFAULTS)
    with pytest.raises(ConfigError):
        parse_combo("whisper-cuda-P1-long+gigaam-cpu-P1", DEFAULTS)


def test_auto_combos_pair_the_knees_of_the_current_sweep(tmp_path: Path) -> None:
    """Catches: auto combos built from stale sweeps or from the wrong knee."""
    groups = tmp_path / "sweep" / "groups"
    groups.mkdir(parents=True)

    def put(
        name: str, fingerprint: str, best: tuple[str, float], eff: tuple[str, float]
    ) -> None:
        doc = {
            "group": name,
            "fingerprint": fingerprint,
            "knees": {
                "best": best[0],
                "best_xrt": best[1],
                "efficient": eff[0],
                "efficient_xrt": eff[1],
            },
        }
        (groups / f"{name}.json").write_text(json.dumps(doc), encoding="utf-8")

    put(
        "whisper-cuda-S", "fp", ("whisper-cuda-S4x4", 30.0), ("whisper-cuda-S2x4", 28.0)
    )
    put(
        "whisper-cuda-P", "fp", ("whisper-cuda-P2x4", 25.0), ("whisper-cuda-P2x4", 25.0)
    )
    put("gigaam-cuda-B", "fp", ("gigaam-cuda-B16x4", 90.0), ("gigaam-cuda-B8x4", 85.0))
    put("whisper-cpu-P-T1", "fp", ("whisper-cpu-P8x1", 8.0), ("whisper-cpu-P4x1", 7.5))
    put("gigaam-cpu-S-T2", "fp", ("gigaam-cpu-S3x2", 30.0), ("gigaam-cpu-S2x2", 27.0))
    put(
        "gigaam-cpu-P-T1",
        "other",
        ("gigaam-cpu-P9x1", 999.0),
        ("gigaam-cpu-P9x1", 999.0),
    )

    best = auto_combos(tmp_path, "fp", DEFAULTS, knee="best")
    efficient = auto_combos(tmp_path, "fp", DEFAULTS, knee="efficient")

    def names(pairs: list[Any]) -> list[tuple[str, str]]:
        return [(a.name, b.name) for a, b in pairs]

    assert names(best) == [
        ("gigaam-cuda-B16x4", "gigaam-cpu-S3x2"),
        ("gigaam-cuda-B16x4", "whisper-cpu-P8x1"),
        ("whisper-cuda-S4x4", "gigaam-cpu-S3x2"),
        ("whisper-cuda-S4x4", "whisper-cpu-P8x1"),
    ]
    assert names(efficient) == [
        ("gigaam-cuda-B8x4", "gigaam-cpu-S2x2"),
        ("gigaam-cuda-B8x4", "whisper-cpu-P4x1"),
        ("whisper-cuda-S2x4", "gigaam-cpu-S2x2"),
        ("whisper-cuda-S2x4", "whisper-cpu-P4x1"),
    ]


# --- Whisper cross-file batching ----------------------------------------------

_SAMPLE_RATE = 16000


class _FakePipeline:
    """Stands in for BatchedInferencePipeline: one segment per clip chunk.

    Mirrors the documented library behavior: with ``clip_timestamps`` every clip
    is one chunk, and each segment carries ``seek = int(chunk offset * 100)``
    (100 frames per second), the offset being the clip start in the input.
    """

    def __init__(self) -> None:
        self.options: dict[str, Any] = {}

    def transcribe(
        self,
        audio: Any,  # noqa: ANN401
        *,
        clip_timestamps: list[dict[str, float]],
        batch_size: int,
        **options: Any,  # noqa: ANN401
    ) -> tuple[list[Any], None]:
        assert batch_size >= 1
        self.options = options
        segments = []
        for clip in clip_timestamps:
            start = int(clip["start"] * _SAMPLE_RATE)
            end = int(clip["end"] * _SAMPLE_RATE)
            chunk = audio[start:end]
            # * The median ignores a boundary sample that float truncation moves.
            label = round(float(np.median(chunk)) * 1000)
            seconds = round((end - start) / _SAMPLE_RATE)
            text = "" if label == 9 else f"v{label}d{seconds}"
            segments.append(
                types.SimpleNamespace(
                    seek=int((start / _SAMPLE_RATE) * 100),
                    start=start / _SAMPLE_RATE,
                    text=text,
                ),
            )
        return segments, None


def _whisper_engine(pipeline: _FakePipeline) -> WhisperEngine:
    config = parse_config_name("whisper-cpu-B3x2", DEFAULTS)
    spec = ChildSpec(
        config=config,
        child_index=0,
        items=(),
        warmup_path=support.PROJECT_ROOT,
        language="ru",
        whisper_model="small",
        whisper_options={"language": "ru", "beam_size": 5},
        whisper_chunk_s=30.0,
        gigaam_cuda_streams=False,
    )
    engine = WhisperEngine(spec)
    engine._model = types.SimpleNamespace(frames_per_second=100)  # noqa: SLF001
    engine._pipeline = pipeline  # noqa: SLF001
    engine._numpy = np  # noqa: SLF001
    engine._sample_rate = _SAMPLE_RATE  # noqa: SLF001
    return engine


def _tone(label: int, seconds: float) -> Any:  # noqa: ANN401
    return np.full(int(seconds * _SAMPLE_RATE), label / 1000.0, dtype=np.float32)


def test_cross_file_batching_returns_every_segment_to_its_own_file() -> None:
    """Catches: transcripts attributed to the wrong clip when clips share a pass."""
    pipeline = _FakePipeline()
    engine = _whisper_engine(pipeline)
    audios = [
        _tone(1, 3.0),  # shorter than a chunk
        _tone(2, 65.0),  # 30 s + 30 s + 5 s chunks
        _tone(3, 0.005),  # 80 samples: below the minimum, dropped
        _tone(9, 1.0),  # the model says nothing: empty transcript
        _tone(4, 30.0),  # exactly one chunk
    ]

    texts = engine._transcribe_audios_batched(audios)  # noqa: SLF001

    assert texts == [
        "v1d3",
        "v2d30 v2d30 v2d5",
        "",
        "",
        "v4d30",
    ]
    # * Timestamps stay on (the sequential path predicts them too).
    assert pipeline.options["without_timestamps"] is False


def test_cross_file_batching_keeps_an_explicit_timestamp_choice() -> None:
    """Catches: the aligned default overriding the user's decode option."""
    pipeline = _FakePipeline()
    engine = _whisper_engine(pipeline)
    engine._spec.whisper_options["without_timestamps"] = True  # noqa: SLF001

    engine._transcribe_audios_batched([_tone(1, 1.0)])  # noqa: SLF001

    assert pipeline.options["without_timestamps"] is True


def test_cross_file_batching_of_only_silent_files_returns_empty_texts() -> None:
    """Catches: a crash on a pass without any usable chunk."""
    engine = _whisper_engine(_FakePipeline())

    assert engine._transcribe_audios_batched([_tone(1, 0.001)]) == [""]  # noqa: SLF001


# --- combo window accounting ----------------------------------------------------


def _measurement(streams: list[WorkStream], tmp_path: Path) -> runner.Measurement:
    sides = [
        runner.SideSpec(
            config=parse_config_name(f"whisper-cpu-P1x{i + 1}", DEFAULTS), stream=s
        )
        for i, s in enumerate(streams)
    ]
    return runner.Measurement(
        sides=sides,
        run_id="unit",
        warmup_path=tmp_path,
        options=support.make_run_options(tmp_path),
        job=None,  # type: ignore[arg-type]
        logs_dir=tmp_path,
        nvidia_smi=None,
    )


def _fill(
    side: Any,  # noqa: ANN401
    rows: list[tuple[float, float, str | None]],
) -> None:
    """Produce one item per row and record (dispatched, finished, error)."""
    stream = side.spec.stream
    produced = stream.take(len(rows), 0.0)
    for occurrence, (began, ended, error) in zip(produced, rows, strict=True):
        item = occurrence.item_id
        side.outcomes[item] = runner._Outcome(  # noqa: SLF001
            item_id=item,
            text="t",
            proc_s=ended - began,
            child=0,
            slot=0,
            batch=1,
            error=error,
        )
        side.dispatched_at[item] = began
        side.finished_at[item] = ended


def test_window_throughput_counts_the_inside_share_of_crossing_items(
    tmp_path: Path,
) -> None:
    """Catches: window xRT that drops, double counts, or fully counts crossing items."""
    from pathlib import Path as _Path  # noqa: PLC0415

    ten = [AudioItem(i, _Path(f"a{i}.wav"), 10.0) for i in range(4)]
    twenty = [AudioItem(0, _Path("b0.wav"), 20.0)]
    stream_a = WorkStream(ten, min_items=4, min_wall_s=0.0, max_items=4)
    stream_b = WorkStream(twenty, min_items=1, min_wall_s=0.0, max_items=1)
    m = _measurement([stream_a, stream_b], tmp_path)
    # * Clock starts at 100 s. Side a ends at 112 s, side b at 108 s: window 8 s.
    m._t_go = 100.0  # noqa: SLF001
    m._t_end = 112.0  # noqa: SLF001
    side_a, side_b = m._sides  # noqa: SLF001
    side_a.t_end = 112.0
    side_b.t_end = 108.0
    _fill(
        side_a,
        [
            (100.0, 104.0, None),  # fully inside: 10 s of audio
            (104.0, 110.0, None),  # 4 of 6 s inside: 10 * 4/6 s of audio
            (110.0, 112.0, None),  # after the window: 0
            (100.0, 101.0, "boom"),  # failed: never counted
        ],
    )
    _fill(side_b, [(100.0, 108.0, None)])  # 20 s fully inside

    timing_a = m._side_timing(side_a)  # noqa: SLF001
    timing_b = m._side_timing(side_b)  # noqa: SLF001
    pair = m._combined_timing([{"timing": timing_a}, {"timing": timing_b}])  # noqa: SLF001

    assert timing_a["window"] == "common"
    assert timing_a["wall_s"] == pytest.approx(8.0)
    assert timing_a["audio_s"] == pytest.approx(10.0 + 10.0 * 4 / 6, abs=1e-3)
    assert timing_a["throughput_x"] == pytest.approx((10.0 + 40 / 6) / 8.0, abs=1e-3)
    # * The whole run stays available: 30 s of ok audio over 12 s.
    assert timing_a["full_wall_s"] == pytest.approx(12.0)
    assert timing_a["full_audio_s"] == pytest.approx(30.0)
    assert timing_a["full_throughput_x"] == pytest.approx(2.5)
    assert timing_b["audio_s"] == pytest.approx(20.0)
    assert timing_b["throughput_x"] == pytest.approx(2.5)
    # * Pair: (16.667 + 20) audio s / 8 s; full run: 50 audio s / 12 s.
    assert pair["wall_s"] == pytest.approx(8.0)
    assert pair["throughput_x"] == pytest.approx((10.0 + 40 / 6 + 20.0) / 8.0, abs=2e-3)
    assert pair["full_throughput_x"] == pytest.approx(50.0 / 12.0, abs=2e-3)


def test_single_side_has_no_window(tmp_path: Path) -> None:
    """Catches: a window applied to a plain run, shortening its wall clock."""
    from pathlib import Path as _Path  # noqa: PLC0415

    clips = [AudioItem(0, _Path("a.wav"), 10.0)]
    m = _measurement(
        [WorkStream(clips, min_items=1, min_wall_s=0.0, max_items=1)], tmp_path
    )
    m._t_go = 10.0  # noqa: SLF001
    side = m._sides[0]  # noqa: SLF001
    side.t_end = 15.0
    _fill(side, [(10.0, 15.0, None)])

    timing = m._side_timing(side)  # noqa: SLF001

    assert "window" not in timing
    assert timing["wall_s"] == pytest.approx(5.0)
    assert timing["throughput_x"] == pytest.approx(2.0)


def test_hold_open_feeds_past_the_minimum_but_not_past_the_cap() -> None:
    """Catches: a held-open side that stops early or never stops."""
    stream = WorkStream(_clips(2), min_items=2, min_wall_s=0.0, max_items=5)

    assert stream.minimum_met(0.0) is False
    stream.take(2, 0.0)
    assert stream.minimum_met(0.0) is True
    held = stream.take(2, 0.0, hold_open=True)
    assert len(held) == 2
    assert not stream.closed
    rest = stream.take(10, 0.0, hold_open=True)
    assert len(rest) == 1
    assert stream.take(1, 0.0, hold_open=True) == []
    assert stream.closed


def test_minimum_met_needs_both_items_and_time() -> None:
    """Catches: a combo releasing a side before its seconds or items are reached."""
    stream = WorkStream(_clips(2), min_items=2, min_wall_s=1.0, max_items=10)

    stream.take(2, 0.0)

    assert stream.minimum_met(0.99) is False
    assert stream.minimum_met(1.0) is True
    assert (
        WorkStream(_clips(2), min_items=2, min_wall_s=0.0, max_items=10).minimum_met(
            5.0
        )
        is False
    )


# --- device mismatch --------------------------------------------------------------


def test_device_mismatch_is_not_a_cuda_oom() -> None:
    """Catches: an environment failure reported as the CUDA saturation point."""
    mismatch = (
        "child 0 failed during load: DeviceMismatchError: device-mismatch: GigaAM "
        "model placed on 'cpu' instead of 'cuda'"
    )

    assert not runner.is_cuda_oom({"reason": mismatch})
    assert runner.is_cuda_oom({"reason": "CUDA out of memory. Tried to allocate"})
    assert runner.is_cuda_oom({"errors": [{"error": "CUBLAS_STATUS_ALLOC_FAILED"}]})
    assert runner.is_cuda_oom(
        {"reason": mismatch, "errors": [{"error": "CUDA out of memory"}]}
    )
    assert evaluate_stop(
        [
            _step(1, 10.0),
            _step(
                2,
                None,
                status="error",
                reason=mismatch,
                cuda_oom=runner.is_cuda_oom({"reason": mismatch}),
            ),
        ],
        RULES,
        scan=True,
    ).startswith("errors")  # type: ignore[union-attr]


def test_gigaam_load_fails_with_a_tagged_mismatch_when_the_device_differs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches: a silent CUDA->CPU fallback reported as a CUDA measurement."""
    import importlib  # noqa: PLC0415

    from stt_concurrency import engines  # noqa: PLC0415

    class _Wrapper:
        active_device = "cpu"

        def __init__(self, **_kwargs: Any) -> None:  # noqa: ANN401
            pass

        def _load_model(self) -> None:
            pass

    real_import = importlib.import_module
    torch_stub = types.SimpleNamespace(set_num_threads=lambda _n: None)
    monkeypatch.setattr(
        engines.importlib,
        "import_module",
        lambda name, *a: torch_stub if name == "torch" else real_import(name, *a),
    )
    monkeypatch.setattr(engines, "GigaAME2ECtcWrapper", _Wrapper)
    monkeypatch.setattr(engines.shutil, "which", lambda _name: "ffmpeg")
    monkeypatch.setattr(engines, "_configure_project_caches", lambda: tmp_path)
    spec = ChildSpec(
        config=parse_config_name("gigaam-cuda-P1x4", DEFAULTS),
        child_index=0,
        items=(),
        warmup_path=tmp_path,
        language="ru",
        whisper_model="small",
        whisper_options={},
        whisper_chunk_s=30.0,
        gigaam_cuda_streams=False,
    )

    with pytest.raises(engines.DeviceMismatchError) as caught:
        engines.GigaAMEngine(spec).load()

    message = str(caught.value)
    assert engines.DEVICE_MISMATCH_TAG in message
    assert "out of memory" not in message.lower()
    assert "'cpu'" in message
    assert "'cuda'" in message
