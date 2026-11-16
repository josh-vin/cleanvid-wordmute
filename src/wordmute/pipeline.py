"""Process one video: scan with cleanvid, time each flagged word, mute just those spans."""

import errno
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import cleanvid.cleanvid as upstream
import pysrt

from . import __version__
from .cleaner import WordMuteCleaner
from .clips import cut_clip, cut_listen_clip, duration_seconds, has_centre_channel, pick_audio_stream
from .match import (
    Guess,
    Hit,
    Word,
    clean_text,
    locate_between,
    merge_spans,
    resolve_line,
    settle_guesses,
    target_key,
    tokenize,
)
from .naming import derive
from .report import ManualRow, append_mute_log, fmt_time, load_manual, manual_key, write_reports

MIN_GAP_SECONDS = 0.12  # a gap between two heard words shorter than this cannot hold a word
MIN_ALIGNED_SECONDS = 0.08  # an aligned word shorter than this has not really been placed
NO_LENGTH_SECONDS = 0.4  # span suggested for a word that was heard but came back without a length


@dataclass
class Mute:
    mute_start: float
    mute_end: float
    word: str
    heard: str
    match: str
    word_start: float | None
    word_end: float | None
    pad: float
    probability: float | None
    source: str  # auto | manual
    sub_index: int | None
    line_start: float | None
    line_end: float | None
    text: str


@dataclass
class ReviewItem:
    """One instance of a flagged word that was not muted."""

    sub_index: int
    line_start: float
    line_end: float
    text: str
    word: str
    reason: str
    suggest_start: float | None = None  # a span that may well hold the word, pad included
    suggest_end: float | None = None
    basis: str = ""  # where the suggestion comes from
    instance: str = ""  # "2 of 3" when the line has several unresolved instances of the word
    listen_start: float | None = None  # what the listening clips cover
    listen_end: float | None = None
    clips: dict = field(default_factory=dict)  # original / muted-preview -> file
    hold: bool = True  # keeps the episode in the needs-review folder


@dataclass
class EpisodeResult:
    source: Path
    name: str
    status: str = "error"  # ok | needs_review | skipped | error
    message: str = ""
    dry_run: bool = False
    flagged_lines: int = 0
    mutes: list = field(default_factory=list)
    review: list = field(default_factory=list)  # held ones and, with unheard.hold off, the ones only listed
    settled_by_hand: int = 0
    output: Path | None = None
    report_dir: Path | None = None
    transcriber: str = ""
    seconds: dict = field(default_factory=dict)


class Refused(Exception):
    """The run would overwrite something or write where it must not."""


def _inside(path: Path, folder: Path) -> bool:
    path, folder = path.resolve(), folder.resolve()
    return path == folder or folder in path.parents


def _default_swears() -> str:
    return os.path.join(os.path.dirname(os.path.realpath(upstream.__file__)), "swears.txt")


def _swears_file(cfg, work: Path) -> str:
    """The word list to use; with extra words, a copy in the work folder that has them added."""
    base = cfg.swears_file or _default_swears()
    if not cfg.extra_swears:
        return base
    lines = Path(base).read_text().splitlines()
    have = {line.split("|")[0].strip().lower() for line in lines}
    lines += [word for word in cfg.extra_swears if word.split("|")[0].strip().lower() not in have]
    merged = work / "swears.txt"
    merged.write_text("\n".join(line for line in lines if line.strip()) + "\n")
    return str(merged)


def _extract_subtitles(video: Path, lang: str, work: Path) -> str:
    """Upstream's choice of embedded subtitles (the first stream in the language, or the forced stream index)."""
    name, _, forced = lang.partition(":")
    listed = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "s", "-show_entries", "stream=index:stream_tags=language",
         "-of", "json", str(video)],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    streams = json.loads(listed).get("streams", [])
    index = int(forced) if forced else next(
        (s["index"] for s in streams if s.get("tags", {}).get("language") == name), None
    )  # fmt: skip
    if index is None:
        return ""
    dest = work / "subs.srt"
    done = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y", "-i", str(video),
         "-map", f"0:{index}", str(dest)],
        capture_output=True, text=True,
    )  # fmt: skip
    return str(dest) if done.returncode == 0 and dest.is_file() else ""


# {\an8}-style positioning tags, which many players show as text; characters that are invisible or control codes
_STYLE_TAG = re.compile(r"\{\\[^}]*\}")
_INVISIBLE = re.compile("[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f-\u009f\u200b-\u200f\u202a-\u202e\u2060\ufeff]")


def tidy_text(text: str) -> str:
    text = _INVISIBLE.sub("", _STYLE_TAG.sub("", text)).replace("\u00a0", " ")
    return "\n".join(line.strip() for line in text.split("\n") if line.strip())


def _tidy_subtitles(path: Path, work: Path):
    """Tidy the working copy of the subtitles in place; the masked subtitles in the output are made from it."""
    if work.resolve() not in path.resolve().parents:
        return  # only ever the copy in the work folder
    try:
        subs = pysrt.open(str(path), encoding="utf-8-sig")
    except UnicodeDecodeError:
        return  # left for upstream's own conversion
    for sub in subs:
        sub.text = tidy_text(sub.text)
    for sub in [s for s in subs if not s.text]:
        subs.remove(sub)
    subs.save(str(path), encoding="utf-8")


def _get_subtitles(source: Path, link: Path, work: Path, lang: str, explicit: str | None) -> str:
    if explicit:
        if not os.path.isfile(explicit):
            raise FileNotFoundError(f"subtitle file not found: {explicit}")
        return str(shutil.copyfile(explicit, work / "subs.srt"))
    # upstream writes the extracted subtitles next to the video it is given: here the link in the work folder
    try:
        found = upstream.GetSubtitles(str(link), lang, True)
    except IndexError:  # upstream's stream listing fails on a subtitle stream without a language tag
        found = _extract_subtitles(link, lang, work)
    if found and os.path.isfile(found):
        return found
    short = {"eng": "en"}.get(lang.split(":")[0], lang.split(":")[0])
    for suffix in (f".{lang.split(':')[0]}.srt", f".{short}.srt", ".srt"):
        sidecar = source.with_name(source.stem + suffix)
        if sidecar.is_file():
            return str(shutil.copyfile(sidecar, work / "subs.srt"))
    raise FileNotFoundError(f"no {lang} subtitles embedded in or next to {source.name}")


def _publish(partial: Path, final: Path, replace: bool):
    """Move the finished file into place without ever overwriting (unless replacing a review copy)."""
    final.parent.mkdir(parents=True, exist_ok=True)
    staged = partial
    if os.stat(partial).st_dev != os.stat(final.parent).st_dev:
        staged = final.parent / f".{final.stem}.partial{final.suffix}"
        shutil.copyfile(partial, staged)
        partial.unlink()
    if replace and final.exists():
        os.replace(staged, final)
        return
    try:
        os.link(staged, final)  # fails if final exists, so this can never overwrite
    except FileExistsError:
        raise Refused(f"{final} appeared while working; finished file left at {staged}") from None
    except OSError as e:
        if e.errno not in (errno.EPERM, errno.ENOTSUP, errno.EOPNOTSUPP, errno.EXDEV):
            raise
        if final.exists():
            raise Refused(f"{final} appeared while working; finished file left at {staged}") from None
        os.rename(staged, final)
        return
    staged.unlink()


def process(
    source, cfg, get_transcriber, *, dry_run=False, replace_review=False, redo=False, add_mutes=(), subs=None,
    show=None, log=print,
):  # fmt: skip
    """redo: reprocess even though this tool's output for the episode exists, and replace that output.
    add_mutes: (start, end) spans to add to the episode's manual.csv as mute rows before processing.
    """
    source = Path(os.path.abspath(source))
    names = derive(source, cfg.name_template, show)
    result = EpisodeResult(source=source, name=names.name, dry_run=dry_run)
    if not source.is_file():
        raise FileNotFoundError(f"not a file: {source}")

    clean_path = Path(cfg.output_root) / names.rel / f"{names.name}.mkv"
    review_path = cfg.review_dir / names.rel / f"{names.name}.mkv"
    report_dir = Path(cfg.reports_root) / names.rel / names.name
    work = Path(cfg.work_dir) / names.work_key
    result.report_dir = report_dir
    replace_review = replace_review or redo

    for target in (clean_path.parent, review_path.parent, report_dir, work):
        if _inside(target, source.parent):
            raise Refused(f"{target} is inside the source folder {source.parent}; nothing is ever written there")
    if not dry_run:
        if clean_path.exists() and not redo:
            result.status = "skipped"
            result.message = f"already exists, not overwriting: {clean_path} (--redo reprocesses and replaces it)"
            return result
        if review_path.exists() and not replace_review:
            result.status = "skipped"
            result.message = (
                f"a needs-review copy already exists: {review_path} (fill in {report_dir / 'manual.csv'} "
                "and rerun with --replace-review)"
            )
            return result

    # scratch folder; cached clips and transcriptions are only reused for the same source file
    work.mkdir(parents=True, exist_ok=True)
    info = source.stat()
    stamp = {"source": str(source), "size": info.st_size, "mtime": int(info.st_mtime)}
    stamp_file = work / "source.json"
    if not stamp_file.is_file() or json.loads(stamp_file.read_text()) != stamp:
        for stale in ("clips", "words"):
            shutil.rmtree(work / stale, ignore_errors=True)
        stamp_file.write_text(json.dumps(stamp))
    clips_dir, words_dir, listen_dir = work / "clips", work / "words", work / "listen"
    shutil.rmtree(listen_dir, ignore_errors=True)
    for folder in (clips_dir, words_dir, listen_dir):
        folder.mkdir(exist_ok=True)
    link = work / f"input{source.suffix.lower()}"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(source)

    subs_file = _get_subtitles(source, link, work, cfg.subtitle_language, subs)
    if cfg.tidy_subs:
        _tidy_subtitles(Path(subs_file), work)
    duration = duration_seconds(link)
    streams = (upstream.GetAudioStreamsInfo(str(link)) or {}).get("streams", [])
    audio_position, audio_stream = pick_audio_stream(streams, cfg.audio_stream_index)
    centre_only = cfg.clip_channel == "auto" and has_centre_channel(audio_stream)
    cache_tag = hashlib.sha1(
        f"{cfg.engine}|{cfg.model}|{cfg.compute_type}|{cfg.language}|{centre_only}".encode()
    ).hexdigest()[:8]

    manual_rows = load_manual(report_dir / "manual.csv")
    for start, end in add_mutes:
        if end <= start:
            raise ValueError(f"--mute {fmt_time(start)}-{fmt_time(end)}: the end must be after the start")
        manual_rows.append(ManualRow(None, "", "mute", start, end, f"added with --mute on {datetime.now():%Y-%m-%d}"))
    for row in manual_rows:
        if row.action == "mute" and row.end > duration + 1:
            raise ValueError(f"{fmt_time(row.end)} is past the end of the video ({report_dir / 'manual.csv'} or --mute)")

    mutes, review = [], []
    spent = {"transcribe": 0.0, "mux": 0.0}
    cleaner = None

    def heard_words(line, clip_start, clip_end, centre, how="transcribe", text=""):
        """Words of one clip in episode time. how: transcribe, or align / relisten with the subtitle text.

        None when the transcriber cannot do `how`. Clips and results are cached in the work folder.
        """
        tag = f"{line.index:04d}_{round(clip_start * 1000)}_{round(clip_end * 1000)}_{'c' if centre else 'm'}"
        wav = clips_dir / f"{tag}.wav"
        if not wav.is_file():
            tmp = clips_dir / f"{tag}.tmp.wav"
            cut_clip(link, audio_position, clip_start, clip_end - clip_start, tmp, centre)
            tmp.rename(wav)
        cache = words_dir / f"{tag}.{cache_tag}.json"
        if how != "transcribe":
            cache = words_dir / f"{tag}.{cache_tag}.{how}-{hashlib.sha1(text.encode()).hexdigest()[:6]}.json"
        if cache.is_file():
            words = [Word(**w) for w in json.loads(cache.read_text())]
        else:
            engine = get_transcriber()
            if how != "transcribe" and not hasattr(engine, how):
                return None
            started = time.monotonic()
            words = engine.transcribe(str(wav)) if how == "transcribe" else getattr(engine, how)(str(wav), text)
            spent["transcribe"] += time.monotonic() - started
            cache.write_text(json.dumps([asdict(w) for w in words]))
        return [Word(w.text, w.start + clip_start, w.end + clip_start, w.probability) for w in words]

    def plan_line(line, pattern, subtitles):
        clip_start = max(0.0, line.start - cfg.clip_padding)
        clip_end = min(duration, line.end + cfg.clip_padding)
        targets = [target_key(m) for m in line.matches]
        shown = {target_key(m): m.lower() for m in line.matches}
        lo, hi = line.start - cfg.window_tolerance, line.end + cfg.window_tolerance

        def resolve(words, floor=cfg.min_word_probability):
            return resolve_line(
                targets, words, line.start, line.end, is_swear=lambda norm: pattern.fullmatch(norm) is not None,
                tolerance=cfg.window_tolerance, min_probability=floor, max_word_seconds=cfg.max_word_seconds,
            )  # fmt: skip

        try:
            if clip_end <= clip_start:
                raise RuntimeError("the subtitle's time is outside the video")
            words = heard_words(line, clip_start, clip_end, centre_only)
        except Exception as e:
            for target in targets:
                review.append(
                    ReviewItem(line.index, line.start, line.end, line.text, shown[target],
                               f"clip could not be transcribed: {e}")
                )  # fmt: skip
            return
        placed = resolve(words)
        found = [(hit, hit.kind) for hit in placed.hits]
        left = {miss.target: miss.count for miss in placed.missing}
        why = {miss.target: miss.reason for miss in placed.missing}

        # Words the transcription did not time: gather guesses made other ways, best first
        guesses = []

        def doubts(resolved, how, said):
            for hit in resolved.doubtful:
                basis, end = f"{said}: {hit.why}", hit.end
                if end <= hit.start:
                    end = hit.start + NO_LENGTH_SECONDS
                    basis += f", so {NO_LENGTH_SECONDS} s from there is assumed"
                guesses.append(Guess(hit.target, hit.start, end, how, hit.heard, hit.probability, basis=basis))

        def gaps(heard, how, said):
            tokens = []
            for number, (start, end, text) in enumerate(subtitles):
                if end > clip_start and start < clip_end:
                    in_line = abs(start - line.start) < 0.0005
                    tokens += tokenize(line.text if in_line else text, pattern, number * 1000, in_line)
            for gap in locate_between(tokens, heard):
                if gap.target not in left or gap.start is None or gap.end is None:
                    continue
                length = gap.end - gap.start
                if length < MIN_GAP_SECONDS or gap.start < lo or gap.end > hi:
                    continue
                where = f'between "{gap.before}" and "{gap.after}"'
                basis = f"{said}: it must be {where}, {length:.2f} s apart"
                if gap.extra_words:
                    basis += f", along with {gap.extra_words} other unheard subtitle word(s)"
                strong = length <= cfg.between_max_seconds and gap.extra_words <= cfg.between_max_extra_words
                guesses.append(
                    Guess(gap.target, gap.start, gap.end, how, gap.heard or f"(nothing, {where})", None, strong,
                          True, basis)
                )  # fmt: skip

        if left:
            try:
                doubts(placed, "heard", "the transcription")
                if cfg.between:
                    gaps(words, "between", "from the transcription")
                if centre_only and cfg.retry_with_mix:
                    mix = heard_words(line, clip_start, clip_end, False)
                    again = resolve(mix)
                    for hit in again.hits:
                        basis = f'heard "{hit.heard}" in the full mix'
                        guesses.append(Guess(hit.target, hit.start, hit.end, hit.kind + "/mix", hit.heard, hit.probability,
                                             True, basis=basis))  # fmt: skip
                    doubts(again, "heard/mix", "the transcription of the full mix")
                    if cfg.between:
                        gaps(mix, "between/mix", "from the transcription of the full mix")
                text = clean_text(line.text)
                if cfg.relisten and text:
                    tight = (max(clip_start, line.start - 0.3), min(clip_end, line.end + 0.3))
                    spans = {(clip_start, clip_end): "relisten", tight: "relisten/tight"}  # one entry if they are equal
                    for (start, end), how in spans.items():
                        primed = heard_words(line, start, end, centre_only, "relisten", text)
                        for hit in resolve(primed or [], 0.0).hits:
                            basis = f'heard "{hit.heard}" when told the subtitle text (confidence {hit.probability:.2f})'
                            guesses.append(
                                Guess(hit.target, hit.start, hit.end, how, hit.heard, hit.probability,
                                      hit.probability >= cfg.relisten_min_probability, basis=basis)
                            )  # fmt: skip
                if cfg.align and text:
                    aligned = heard_words(line, clip_start, clip_end, centre_only, "align", text)
                    for hit in resolve(aligned or [], 0.0).hits:
                        strong = (
                            hit.probability >= cfg.align_min_probability and hit.end - hit.start >= MIN_ALIGNED_SECONDS
                        )
                        basis = f"lining the subtitle text up with the audio (confidence {hit.probability:.2f})"
                        guesses.append(
                            Guess(hit.target, hit.start, hit.end, "aligned", "(subtitle text)", hit.probability, strong,
                                  basis=basis)
                        )  # fmt: skip
                guesses.sort(key=lambda g: (not g.strong, g.bound))

            except Exception as e:
                for target in why:
                    why[target] += f"; the further search failed: {e}"

        unheard = []  # (target, best remaining guess or None), one per instance still not timed
        for target, count in left.items():
            taken = [(hit.start, hit.end) for hit, _ in found if hit.target == target]
            accepted, leftover = settle_guesses(
                [g for g in guesses if g.target == target], count, taken, cfg.between_max_seconds
            )
            for guess in accepted:
                found.append((Hit(target, guess.heard, guess.how, guess.start, guess.end, guess.probability), guess.how))
            suggested = []
            for _ in range(count - len(accepted)):
                idea = next(
                    (g for g in leftover if not any(g.start < s.end and s.start < g.end for s in suggested)), None
                )
                if idea:
                    suggested.append(idea)
                unheard.append((target, idea))

        if unheard and cfg.whole_line and line.end - line.start <= cfg.whole_line_max_seconds:
            others = sum(1 for token in tokenize(line.text, pattern) if token.target is None)
            if others <= cfg.whole_line_other_words:
                # little or nothing but flagged words is said in this line: mute all of it
                words = ", ".join(dict.fromkeys(shown[target] for target, _ in unheard))
                mutes.append(
                    Mute(
                        max(0.0, line.start - cfg.whole_line_pad), min(duration, line.end + cfg.whole_line_pad),
                        words, "", "whole-line", line.start, line.end, cfg.whole_line_pad, None,
                        "auto", line.index, line.start, line.end, line.text,
                    )
                )  # fmt: skip
                unheard = []
        for target, idea in unheard:
            if idea and cfg.mute_best_guess and idea.end - idea.start <= cfg.max_word_seconds:
                how = f"guess:{idea.how}"
                found.append((Hit(target, idea.heard, how, idea.start, idea.end, idea.probability), how))
                continue
            item = ReviewItem(line.index, line.start, line.end, line.text, shown[target], why[target])
            item.hold = cfg.hold_unheard
            if idea:
                item.suggest_start = max(0.0, idea.start - cfg.word_pad)
                item.suggest_end = min(duration, idea.end + cfg.word_pad)
                item.basis = idea.basis
            review.append(item)
        for hit, how in found:
            # a word timed from well before its line (or to well after it) has been stretched over silence
            start, end = max(hit.start, lo), min(hit.end, hi)
            if end <= start:
                start, end = hit.start, hit.end
            mutes.append(
                Mute(
                    max(0.0, start - cfg.word_pad), min(duration, end + cfg.word_pad),
                    shown[hit.target], hit.heard, how, start, end, cfg.word_pad, hit.probability,
                    "auto", line.index, line.start, line.end, line.text,
                )
            )  # fmt: skip

    def planner(flagged):
        result.flagged_lines = len(flagged)
        pattern = cleaner.SwearPattern()
        for number, line in enumerate(flagged, 1):
            plan_line(line, pattern, cleaner.subtitles)
            if number % 10 == 0 or number == len(flagged):
                log(f"  {names.name}: {number}/{len(flagged)} flagged lines checked")

        # manual spans: every mute row is applied; each mute/skip row settles one review instance of its line + word
        lines = {line.index: line for line in flagged}
        before = len(review)
        for row in manual_rows:
            if row.action == "mute":
                line = lines.get(row.sub_index)
                mutes.append(
                    Mute(row.start, min(duration, row.end), row.word, "", "manual", None, None, 0.0, None, "manual",
                         row.sub_index, line.start if line else None, line.end if line else None,
                         line.text if line else row.note)
                )  # fmt: skip
            if not row.action or row.sub_index is None:
                continue
            same = [r for r in review if manual_key(r.sub_index, r.word) == manual_key(row.sub_index, row.word)]
            covered = [
                r for r in same
                if row.action == "mute" and r.suggest_start is not None
                and row.start < r.suggest_end and r.suggest_start < row.end
            ]  # fmt: skip
            if same:
                review.remove((covered or same)[0])
        result.settled_by_hand = before - len(review)

        review.sort(key=lambda r: (r.line_start, r.suggest_start if r.suggest_start is not None else r.line_end))
        for item in review:
            same = [r for r in review if (r.sub_index, r.word) == (item.sub_index, item.word)]
            if len(same) > 1:
                item.instance = f"{same.index(item) + 1} of {len(same)}"
            # the whole subtitle line, so the word is in the clip even when the suggestion is wrong
            item.listen_start = item.line_start - cfg.window_tolerance
            item.listen_end = item.line_end + cfg.window_tolerance
            if item.suggest_start is not None:
                item.listen_start = min(item.listen_start, item.suggest_start - cfg.review_clip_margin)
                item.listen_end = max(item.listen_end, item.suggest_end + cfg.review_clip_margin)
            item.listen_start, item.listen_end = max(0.0, item.listen_start), min(duration, item.listen_end)
        return merge_spans((m.mute_start, m.mute_end) for m in mutes)

    cleaner = WordMuteCleaner(
        str(link),
        subs_file,
        str(work / "output-not-set.mkv"),
        str(work / "clean.srt"),
        _swears_file(cfg, work),
        embedSubs=cfg.embed_subs,
        fullSubs=cfg.full_subs,
        subsLang=cfg.subtitle_language.split(":")[0],
        audioStreamIdx=cfg.audio_stream_index,
        aParams=cfg.audio_params,
        planner=planner,
        final_peek=duration,
    )
    cleaner.CreateCleanSubAndMuteList()

    result.mutes, result.review = mutes, review
    held = [item for item in review if item.hold]
    result.status = "needs_review" if held else "ok"
    result.seconds = spent
    if dry_run:
        return result

    final = review_path if held else clean_path
    if clean_path.exists() and not redo:
        raise Refused(f"already exists, not overwriting: {clean_path}")
    for item in review:
        if item.listen_end is None or item.listen_end <= item.listen_start:
            continue
        stem = f"line{item.sub_index:04d}_{fmt_time(item.line_start).replace(':', '-')}_{target_key(item.word)}"
        if item.instance:
            stem += "_" + item.instance.replace(" ", "")
        length = item.listen_end - item.listen_start
        item.clips["original"] = listen_dir / f"{stem}_original.wav"
        cut_listen_clip(link, audio_position, item.listen_start, length, item.clips["original"])
        if item.suggest_start is not None:
            item.clips["muted-preview"] = listen_dir / f"{stem}_muted-preview.wav"
            cut_listen_clip(
                link, audio_position, item.listen_start, length, item.clips["muted-preview"],
                silence=(item.suggest_start, item.suggest_end),
            )  # fmt: skip
    partial = work / "output.partial.mkv"
    if partial.exists():
        partial.unlink()
    cleaner.outputVidFileSpec = str(partial)
    started = time.monotonic()
    cleaner.MultiplexCleanVideo()
    spent["mux"] = time.monotonic() - started
    out_duration = duration_seconds(partial)
    if abs(out_duration - duration) > 2.0:
        raise RuntimeError(f"output is {out_duration:.1f}s long but the source is {duration:.1f}s; left at {partial}")
    _publish(partial, final, replace=replace_review if held else redo)
    if not held and review_path.exists() and replace_review:
        review_path.unlink()  # the earlier needs-review copy this run supersedes
    if held and clean_path.exists():
        result.message = f"the earlier clean file is still in place and does not have this run's changes: {clean_path}"
    result.output = final

    status = {
        "status": result.status,
        "episode": names.name,
        "source": str(source),
        "output": str(final),
        "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "flagged_lines": result.flagged_lines,
        "mutes": len(mutes),
        "mutes_by_match": {how: sum(1 for m in mutes if m.match == how) for how in sorted({m.match for m in mutes})},
        "mute_spans_after_merging": len(cleaner.muteSpans),
        "needs_review": len(held),
        "not_heard_left_unmuted": len(review) - len(held),
        "settled_by_hand": result.settled_by_hand,
        "seconds": {k: round(v, 1) for k, v in spent.items()},
        "duration": round(duration, 3),
        "settings": {
            "clip_padding": cfg.clip_padding,
            "clip_channel": "centre" if centre_only else "mix",
            "word_pad": cfg.word_pad,
            "window_tolerance": cfg.window_tolerance,
            "min_word_probability": cfg.min_word_probability,
            "max_word_seconds": cfg.max_word_seconds,
            "retry_with_mix": cfg.retry_with_mix,
            "between": cfg.between,
            "between_max_seconds": cfg.between_max_seconds,
            "between_max_extra_words": cfg.between_max_extra_words,
            "relisten": cfg.relisten,
            "relisten_min_probability": cfg.relisten_min_probability,
            "align": cfg.align,
            "align_min_probability": cfg.align_min_probability,
            "whole_line": cfg.whole_line,
            "whole_line_other_words": cfg.whole_line_other_words,
            "mute_best_guess": cfg.mute_best_guess,
            "hold_unheard": cfg.hold_unheard,
            "engine": cfg.engine,
            "model": cfg.model,
            "device": cfg.device,
            "compute_type": cfg.compute_type,
            "audio_params": cfg.audio_params,
        },
        "versions": {"cleanvid-wordmute": __version__, "cleanvid": version("cleanvid")},
    }
    write_reports(
        report_dir, status=status, mutes=mutes, review=review, clean_subs=work / "clean.srt", manual_rows=manual_rows
    )
    append_mute_log(Path(cfg.reports_root) / "mutes.log", names.name, final, mutes)
    return result
