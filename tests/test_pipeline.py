"""End to end on a generated test video, with a stand-in transcriber."""

import csv
import json
import re
import subprocess
from pathlib import Path

import pytest

from wordmute.config import Config
from wordmute.match import Word
from wordmute import pipeline
from wordmute.pipeline import Refused, process

# what is "spoken" in the test video (episode time)
SPOKEN = [
    ("Well,", 3.05, 3.3), ("shit.", 3.4, 3.8), ("That's", 4.0, 4.15), ("a", 4.15, 4.2), ("damn", 4.3, 4.6),
    ("shame.", 4.6, 4.9), ("Fuck!", 8.0, 8.3), ("Fuck", 8.6, 8.85), ("this.", 8.9, 9.2),
]  # fmt: skip


class FakeTranscriber:
    name = "fake"

    def __init__(self, spoken=SPOKEN):
        self.spoken = spoken
        self.calls = 0

    def transcribe(self, wav_path):
        self.calls += 1
        _, start_ms, end_ms, _ = Path(wav_path).stem.split("_")
        start, end = int(start_ms) / 1000, int(end_ms) / 1000
        return [
            Word(" " + t, s - start, e - start, p[0] if p else 0.95)
            for t, s, e, *p in self.spoken
            if s >= start and e <= end
        ]


def config(media):
    tmp = media["tmp"]
    return Config(
        output_root=str(tmp / "out"), reports_root=str(tmp / "reports"), work_dir=str(tmp / "work"),
        swears_file=str(media["swears"]), audio_params="-c:a aac -b:a 128k",
    )  # fmt: skip


def loudness(path, start, length):
    err = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-ss", str(start), "-t", str(length), "-i", str(path),
         "-map", "0:a:0", "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True, text=True, check=True,
    ).stderr  # fmt: skip
    return float(re.search(r"mean_volume: (-?[\d.]+|-inf) dB", err).group(1))


def snapshot(folder):
    return sorted((p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in folder.iterdir())


def rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


def test_only_the_words_are_muted_and_source_is_untouched(media):
    cfg, fake = config(media), FakeTranscriber()
    before = snapshot(media["video"].parent)
    result = process(media["video"], cfg, lambda: fake, subs=str(media["subs"]), log=lambda _: None)

    assert result.status == "ok"
    out = media["tmp"] / "out" / "Show" / "Season 01" / "Show - S01E02.mkv"
    assert result.output == out and out.is_file()
    assert snapshot(media["video"].parent) == before

    # the word is silent, the rest of the same subtitle line is not
    assert loudness(out, 3.45, 0.3) < -60  # "shit" 3.4-3.8
    assert loudness(out, 3.0, 0.2) > -30  # same line, before the word
    assert loudness(out, 3.98, 0.15) > -30  # same line, between the two muted words
    assert loudness(out, 4.35, 0.2) < -60  # "damn" 4.3-4.6
    assert loudness(out, 8.4, 0.08) > -30  # between the two "fuck"s
    assert loudness(out, 6.0, 1.0) > -30  # unflagged line

    report = media["tmp"] / "reports" / "Show" / "Season 01" / "Show - S01E02"
    mutes = rows(report / "mutes.csv")
    assert [(m["word"], m["mute_start"], m["mute_end"]) for m in mutes] == [
        ("shit", "0:00:03.300", "0:00:03.900"), ("damn", "0:00:04.200", "0:00:04.700"),
        ("fuck", "0:00:07.900", "0:00:08.400"), ("fuck", "0:00:08.500", "0:00:08.950"),
    ]  # fmt: skip
    assert rows(report / "review.csv") == []
    status = json.loads((report / "status.json").read_text())
    assert status["status"] == "ok" and status["mutes"] == 4 and status["needs_review"] == 0
    assert not (report / "manual.csv").exists()
    assert len((media["tmp"] / "reports" / "mutes.log").read_text().splitlines()) == 4
    streams = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(out)],
        capture_output=True, text=True,
    ).stdout.split()  # fmt: skip
    assert sorted(streams) == ["audio", "subtitle", "video"]

    # a second run never overwrites
    stamp = out.stat().st_mtime_ns
    again = process(media["video"], cfg, lambda: fake, subs=str(media["subs"]), log=lambda _: None)
    assert again.status == "skipped" and out.stat().st_mtime_ns == stamp


def test_dry_run_writes_no_video_or_reports(media):
    cfg, fake = config(media), FakeTranscriber()
    result = process(media["video"], cfg, lambda: fake, dry_run=True, subs=str(media["subs"]), log=lambda _: None)
    assert result.status == "ok" and len(result.mutes) == 4 and result.output is None
    assert not (media["tmp"] / "out").exists() and not (media["tmp"] / "reports").exists()
    # the real run reuses the dry run's transcriptions
    calls = fake.calls
    process(media["video"], cfg, lambda: fake, subs=str(media["subs"]), log=lambda _: None)
    assert fake.calls == calls


def test_unplaced_word_sends_episode_to_review_then_manual_span_finishes_it(media):
    cfg = config(media)
    fake = FakeTranscriber([w for w in SPOKEN if w[1] != 8.6])  # the second "Fuck" is not heard
    kw = dict(subs=str(media["subs"]), log=lambda _: None)
    result = process(media["video"], cfg, lambda: fake, **kw)

    clean = media["tmp"] / "out" / "Show" / "Season 01" / "Show - S01E02.mkv"
    held = media["tmp"] / "out" / "_needs-review" / "Show" / "Season 01" / "Show - S01E02.mkv"
    report = media["tmp"] / "reports" / "Show" / "Season 01" / "Show - S01E02"
    assert result.status == "needs_review" and held.is_file() and not clean.exists()
    review = rows(report / "review.csv")
    assert len(review) == 1
    assert review[0]["time"] == "0:00:08.000" and review[0]["word"] == "fuck" and "1 of 2" in review[0]["reason"]
    assert "Fuck! Fuck this." in review[0]["subtitle_text"]
    assert review[0]["suggested_start"] == "" and review[0]["muted_preview_clip"] == ""
    assert (report / review[0]["original_clip"]).is_file() and review[0]["clips_start_at"] == "0:00:07.500"
    assert json.loads((report / "status.json").read_text())["status"] == "needs_review"
    # the line was not muted as a whole: only the word that was found
    assert loudness(held, 8.05, 0.2) < -60 and loudness(held, 8.6, 0.6) > -30

    # nothing is replaced without being asked
    assert process(media["video"], cfg, lambda: fake, **kw).status == "skipped"

    manual = report / "manual.csv"
    text = manual.read_text()
    assert "\n,4,fuck,,," in text
    manual.write_text(text.replace("\n,4,fuck,,,", "\nmute,4,fuck,8.55,0:00:08.900,"))
    result = process(media["video"], cfg, lambda: fake, replace_review=True, **kw)
    assert result.status == "ok" and result.settled_by_hand == 1
    assert clean.is_file() and not held.exists()
    assert loudness(clean, 8.6, 0.25) < -60
    assert [m["source"] for m in rows(report / "mutes.csv")].count("manual") == 1
    assert rows(report / "review.csv") == [] and not list((report / "clips").glob("*.wav"))


def test_sidecar_subtitles_are_found_and_never_written_next_to_the_source(media):
    sidecar = media["video"].with_name(media["video"].stem + ".eng.srt")
    sidecar.write_text(media["subs"].read_text())
    before = snapshot(media["video"].parent)
    result = process(media["video"], config(media), lambda: FakeTranscriber(), log=lambda _: None)
    assert result.status == "ok" and len(result.mutes) == 4
    assert snapshot(media["video"].parent) == before


def test_refuses_to_write_into_the_source_folder(media):
    cfg = config(media)
    cfg.reports_root = str(media["video"].parent)
    with pytest.raises(Refused):
        process(media["video"], cfg, lambda: FakeTranscriber(), subs=str(media["subs"]), log=lambda _: None)


def respoken(**changes):
    """SPOKEN with some words replaced: start time -> (text, start, end[, probability])."""
    return [changes.get(f"t{w[1]}", w) for w in SPOKEN]


def test_misheard_word_is_muted_between_its_neighbours(media):
    fake = FakeTranscriber(respoken(**{"t3.4": ("ship.", 3.4, 3.8)}))
    result = process(media["video"], config(media), lambda: fake, subs=str(media["subs"]), log=lambda _: None)
    assert result.status == "ok"
    report = media["tmp"] / "reports" / "Show" / "Season 01" / "Show - S01E02"
    first = rows(report / "mutes.csv")[0]
    # from the end of "Well," (3.3) to the start of "That's" (4.0), plus the pad
    assert (first["word"], first["match"], first["heard_as"]) == ("shit", "between", "ship.")
    assert (first["mute_start"], first["mute_end"]) == ("0:00:03.200", "0:00:04.100")
    assert loudness(result.output, 3.35, 0.6) < -60 and loudness(result.output, 3.0, 0.15) > -30


def test_doubtful_word_is_suggested_with_clips_and_approved_in_the_manual_file(media):
    cfg = config(media)
    fake = FakeTranscriber(respoken(**{"t3.4": ("shit.", 3.4, 3.8, 0.04)}))  # heard, but far too unsure
    kw = dict(subs=str(media["subs"]), log=lambda _: None)
    result = process(media["video"], cfg, lambda: fake, **kw)
    assert result.status == "needs_review" and len(result.review) == 1

    report = media["tmp"] / "reports" / "Show" / "Season 01" / "Show - S01E02"
    (item,) = rows(report / "review.csv")
    assert (item["word"], item["suggested_start"], item["suggested_end"]) == ("shit", "0:00:03.300", "0:00:03.900")
    assert "confidence is low" in item["suggestion_from"] and item["clips_start_at"] == "0:00:02.500"
    original, preview = report / item["original_clip"], report / item["muted_preview_clip"]
    assert original.name.endswith("_shit_original.wav") and preview.name.endswith("_shit_muted-preview.wav")
    # both clips cover the subtitle line and the suggested span; only the preview has the span silenced
    assert loudness(original, 0.85, 0.5) > -30 and loudness(preview, 0.85, 0.5) < -60
    assert loudness(preview, 0.0, 0.75) > -30 and loudness(preview, 1.45, 0.7) > -30
    # nothing was muted in the video for it
    held = media["tmp"] / "out" / "_needs-review" / "Show" / "Season 01" / "Show - S01E02.mkv"
    assert loudness(held, 3.45, 0.3) > -30

    manual = report / "manual.csv"
    row = "\n,2,shit,0:00:03.300,0:00:03.900,"
    assert row in manual.read_text()
    manual.write_text(manual.read_text().replace(row, "\nmute" + row[1:]))
    result = process(media["video"], cfg, lambda: fake, replace_review=True, **kw)
    assert result.status == "ok" and result.settled_by_hand == 1
    assert loudness(result.output, 3.35, 0.5) < -60 and not held.exists()
    assert [r["action"] for r in rows(manual)] == ["mute"]  # kept; no rows left to review


class PrimedTranscriber(FakeTranscriber):
    """Hears the missing word once it is told the subtitle text."""

    def relisten(self, wav_path, text):
        assert text == "Well, shit. That's a damn shame."
        _, start_ms, _, _ = Path(wav_path).stem.split("_")
        return [Word(" shit.", 3.4 - int(start_ms) / 1000, 3.8 - int(start_ms) / 1000, self.confidence)]


def test_word_heard_only_when_primed_with_the_subtitle(media):
    cfg = config(media)
    cfg.between = False
    fake = PrimedTranscriber([w for w in SPOKEN if w[1] != 3.4])
    fake.confidence = 0.8
    result = process(media["video"], cfg, lambda: fake, dry_run=True, subs=str(media["subs"]), log=lambda _: None)
    assert result.status == "ok"
    (mute,) = [m for m in result.mutes if m.word == "shit"]
    assert (mute.match, mute.mute_start, mute.mute_end) == ("relisten+relisten/tight", 3.3, 3.9)


def test_two_unsure_ways_that_agree_are_enough(media):
    cfg = config(media)
    cfg.between = False
    fake = PrimedTranscriber([w for w in SPOKEN if w[1] != 3.4])
    fake.confidence = 0.02  # both the padded and the tight clip give it, neither confidently
    result = process(media["video"], cfg, lambda: fake, dry_run=True, subs=str(media["subs"]), log=lambda _: None)
    assert result.status == "ok"
    assert [m.match for m in result.mutes if m.word == "shit"] == ["relisten+relisten/tight"]


def test_redo_with_an_added_mute(media):
    cfg, fake = config(media), FakeTranscriber()
    kw = dict(subs=str(media["subs"]), log=lambda _: None)
    first = process(media["video"], cfg, lambda: fake, **kw)
    assert first.status == "ok" and loudness(first.output, 6.2, 0.5) > -30

    # something noticed while watching: reprocess with that span
    again = process(media["video"], cfg, lambda: fake, redo=True, add_mutes=[(6.2, 6.7)], **kw)
    assert again.status == "ok" and again.output == first.output
    assert loudness(again.output, 6.25, 0.4) < -60 and loudness(again.output, 3.45, 0.3) < -60
    manual = media["tmp"] / "reports" / "Show" / "Season 01" / "Show - S01E02" / "manual.csv"
    (row,) = rows(manual)
    assert (row["sub_index"], row["action"], row["start"], row["end"]) == ("", "mute", "0:00:06.200", "0:00:06.700")

    # without --redo the finished file is left alone; with it the saved span is applied again
    assert process(media["video"], cfg, lambda: fake, **kw).status == "skipped"
    third = process(media["video"], cfg, lambda: fake, redo=True, **kw)
    assert [m.source for m in third.mutes].count("manual") == 1 and len(rows(manual)) == 1


def test_embedded_subtitles_beside_a_stream_without_a_language(media, monkeypatch):
    # upstream's own stream listing raises on the untagged stream
    folder = media["tmp"] / "Embedded" / "Season 1"
    folder.mkdir(parents=True)
    video = folder / "Embedded.S01E03.mkv"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-i", str(media["video"]),
         "-i", str(media["subs"]), "-i", str(media["subs"]), "-map", "0:v", "-map", "0:a", "-map", "1", "-map", "2",
         "-c", "copy", "-c:s", "srt", "-metadata:s:s:0", "language=eng", str(video)],
        check=True,
    )  # fmt: skip
    used = []
    real = pipeline._extract_subtitles
    monkeypatch.setattr(pipeline, "_extract_subtitles", lambda *a: used.append(a) or real(*a))
    result = process(video, config(media), lambda: FakeTranscriber(), dry_run=True, log=lambda _: None)
    assert used and result.status == "ok"
    assert sorted(m.word for m in result.mutes) == ["damn", "fuck", "fuck", "shit"]


def test_subtitles_are_tidied_in_the_work_copy_only(tmp_path):
    assert pipeline.tidy_text("{\\an8}<i>- Oh,\u200b shit.</i>\n\ufeff- Caf\u00e9\u00a0\u2026 \u00a35 ") == "<i>- Oh, shit.</i>\n- Caf\u00e9 \u2026 \u00a35"
    work, elsewhere = tmp_path / "work", tmp_path / "library"
    work.mkdir(), elsewhere.mkdir()
    srt = "1\n00:00:01,000 --> 00:00:02,000\n{\\an8}Hello\n\n2\n00:00:03,000 --> 00:00:04,000\n{\\an8}\n\n"
    for folder in (work, elsewhere):
        (folder / "subs.srt").write_text(srt)
        pipeline._tidy_subtitles(folder / "subs.srt", work)
    assert (elsewhere / "subs.srt").read_text() == srt
    assert (work / "subs.srt").read_text() == "1\n00:00:01,000 --> 00:00:02,000\nHello\n\n"


def test_unheard_word_in_a_line_of_little_else_mutes_the_whole_line(media):
    cfg = config(media)
    cfg.whole_line = True
    fake = FakeTranscriber([w for w in SPOKEN if w[1] != 8.6])  # the second "Fuck" of "Fuck! Fuck this." is not heard
    result = process(media["video"], cfg, lambda: fake, subs=str(media["subs"]), log=lambda _: None)
    clean = media["tmp"] / "out" / "Show" / "Season 01" / "Show - S01E02.mkv"
    assert result.status == "ok" and not result.review and clean.is_file()
    (whole,) = [m for m in result.mutes if m.match == "whole-line"]
    assert (whole.word, whole.mute_start, whole.mute_end) == ("fuck", 7.75, 9.75)
    assert loudness(clean, 7.8, 1.9) < -60 and loudness(clean, 7.0, 0.7) > -30 and loudness(clean, 9.8, 1.0) > -30


def test_a_line_with_more_said_in_it_is_not_muted_whole(media):
    cfg = config(media)
    cfg.whole_line, cfg.between = True, False
    fake = FakeTranscriber([w for w in SPOKEN if w[1] != 3.4])  # "shit." of "Well, shit. That's a damn shame."
    result = process(media["video"], cfg, lambda: fake, subs=str(media["subs"]), log=lambda _: None)
    assert result.status == "needs_review" and not [m for m in result.mutes if m.match == "whole-line"]


def test_best_guess_is_muted_when_asked(media):
    cfg = config(media)
    cfg.mute_best_guess = True
    fake = FakeTranscriber(respoken(**{"t3.4": ("shit.", 3.4, 3.8, 0.04)}))  # heard, but far too unsure
    result = process(media["video"], cfg, lambda: fake, subs=str(media["subs"]), log=lambda _: None)
    assert result.status == "ok" and not result.review
    (guess,) = [m for m in result.mutes if m.match.startswith("guess:")]
    assert (guess.word, guess.match, guess.mute_start, guess.mute_end) == ("shit", "guess:heard", 3.3, 3.9)
    assert loudness(result.output, 3.4, 0.4) < -60


def test_unheard_word_without_a_guess_does_not_hold_the_episode_when_told_so(media):
    cfg = config(media)
    cfg.between, cfg.hold_unheard = False, False
    fake = FakeTranscriber([w for w in SPOKEN if w[1] != 3.4])  # "shit." is not heard at all
    result = process(media["video"], cfg, lambda: fake, subs=str(media["subs"]), log=lambda _: None)
    clean = media["tmp"] / "out" / "Show" / "Season 01" / "Show - S01E02.mkv"
    report = media["tmp"] / "reports" / "Show" / "Season 01" / "Show - S01E02"
    assert result.status == "ok" and result.output == clean and clean.is_file()
    assert not (media["tmp"] / "out" / "_needs-review").exists()
    # it is still listed, with a clip to listen to, and nothing was muted for it
    (item,) = rows(report / "review.csv")
    assert item["word"] == "shit" and (report / item["original_clip"]).is_file()
    status = json.loads((report / "status.json").read_text())
    assert (status["status"], status["needs_review"], status["not_heard_left_unmuted"]) == ("ok", 0, 1)
    assert loudness(clean, 3.4, 0.4) > -30


def test_extra_words_are_added_to_a_copy_of_the_word_list(media):
    cfg = config(media)
    cfg.extra_swears = ["shame", "Fuck"]
    before = media["swears"].read_text()
    result = process(media["video"], cfg, lambda: FakeTranscriber(), subs=str(media["subs"]), log=lambda _: None)
    assert result.status == "ok" and media["swears"].read_text() == before
    (shame,) = [m for m in result.mutes if m.word == "shame"]
    assert (shame.mute_start, shame.mute_end) == (4.5, 5.0) and loudness(result.output, 4.65, 0.2) < -60
    assert len([m for m in result.mutes if m.word == "fuck"]) == 2
