"""build_mute_filters must stay text-identical to upstream cleanvid's own mute filters."""

import re

import pysrt
from cleanvid.cleanvid import VidCleaner

from wordmute.cleaner import WordMuteCleaner, build_mute_filters


def upstream_filters(media, out):
    cleaner = VidCleaner(
        str(media["video"]), str(media["subs"]), str(out / "u.mkv"), str(out / "u_clean.srt"), str(media["swears"])
    )
    cleaner.CreateCleanSubAndMuteList()
    return list(cleaner.muteTimeList)


def test_whole_line_spans_give_upstream_filters(media):
    out = media["tmp"]
    expected = upstream_filters(media, out)
    assert len(expected) == 4  # two flagged lines, a fade-out and a fade-in each

    subs = pysrt.open(str(media["subs"]))
    line_spans = [(s.start.ordinal / 1000.0, s.end.ordinal / 1000.0) for s in subs if s.index in (2, 4)]
    # upstream peeks at a dummy subtitle it appends after the last one for the final fade-in
    final_peek = float(re.search(r"between\(t,[\d.]+,([\d.]+)\)':t=in", expected[-1]).group(1))
    assert build_mute_filters(line_spans, final_peek) == expected


def test_subclass_reports_flagged_lines_and_uses_planner_spans(media):
    out = media["tmp"]
    seen = []

    def planner(lines):
        seen.extend(lines)
        return [(3.4, 3.8), (8.0, 8.3)]

    cleaner = WordMuteCleaner(
        str(media["video"]), str(media["subs"]), str(out / "w.mkv"), str(out / "w_clean.srt"), str(media["swears"]),
        planner=planner, final_peek=12.0,
    )  # fmt: skip
    cleaner.CreateCleanSubAndMuteList()

    assert [(l.index, l.start, l.end, l.matches) for l in seen] == [
        (2, 3.0, 5.0, ["shit", "damn"]),
        (4, 8.0, 9.5, ["Fuck", "Fuck"]),
    ]
    assert cleaner.lineMuteTimeList == upstream_filters(media, out)
    assert cleaner.muteTimeList == build_mute_filters([(3.4, 3.8), (8.0, 8.3)], 12.0)
    assert "between(t,3.400,3.800)" in cleaner.muteTimeList[0]
    # upstream still wrote its cleaned subtitles
    clean = (out / "w_clean.srt").read_text()
    assert "*****" in clean and "darn" in clean and "shit" not in clean


def test_no_spans_still_gives_upstream_a_filter(media):
    out = media["tmp"]
    cleaner = WordMuteCleaner(
        str(media["video"]), str(media["subs"]), str(out / "n.mkv"), str(out / "n_clean.srt"), str(media["swears"]),
        planner=lambda lines: [], final_peek=12.0,
    )  # fmt: skip
    cleaner.CreateCleanSubAndMuteList()
    assert cleaner.muteTimeList == ["anull"]
