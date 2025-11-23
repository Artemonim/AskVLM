from utils import exporters as ex


def test_fill_empty_gaps_in_srt() -> None:
    """fill_empty_gaps_in_srt stretches cues to fill silence."""
    srt_in = (
        "1\n00:00:01,000 --> 00:00:02,000\nA\n\n2\n00:00:03,000 --> 00:00:04,000\nB\n\n"
    )
    # Gap from 2.0s to 3.0s. Expect cue 1 to end at 3.0s (start of cue 2).
    srt_out = ex.fill_empty_gaps_in_srt(srt_in)
    assert "00:00:01,000 --> 00:00:03,000" in srt_out
    assert "00:00:03,000 --> 00:00:04,000" in srt_out
