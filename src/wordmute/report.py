"""Per-episode report files and the manual-spans file."""

import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .match import target_key

MUTES_HEADER = [
    "mute_start", "mute_end", "word", "heard_as", "match", "word_start", "word_end", "pad", "probability",
    "source", "sub_index", "line_start", "line_end", "subtitle_text",
]  # fmt: skip
REVIEW_HEADER = [
    "time", "subtitle_text", "word", "instance", "reason", "suggested_start", "suggested_end", "suggestion_from",
    "original_clip", "muted_preview_clip", "clips_start_at", "sub_index", "line_end",
]  # fmt: skip
MANUAL_HEADER = ["action", "sub_index", "word", "start", "end", "note"]
MANUAL_HELP = """\
# Decisions made by hand for this episode. Each row below the header line is one flagged word that was not
# muted: it could not be heard (see review.csv and the clips folder). Leave a row alone to keep it as it is.
# Type your decision at the very start of the row, before its first comma:
#   mute : silence the row's start-end span (episode time, seconds or H:MM:SS.mmm), exactly as given
#   skip : the word is not actually spoken, nothing to mute
#   e.g.   ,370,fuck,0:23:55.900,0:23:56.400,...   becomes   mute,370,fuck,0:23:55.900,0:23:56.400,...
# Where a span could be suggested it is already in start/end: listen to the row's muted-preview clip and, if
# the word is gone, type mute. Otherwise correct start/end first. Rows you have not decided are rewritten on
# every run; decided rows are kept. Then rerun the episode (--replace-review, or --redo for a finished one).
# To mute something else (say a word noticed while watching), add a row:   mute,,,START,END,
"""
def fmt_time(seconds: float) -> str:
    ms = round(seconds * 1000)
    return f"{ms // 3600000}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d}.{ms % 1000:03d}"


def parse_time(text: str) -> float:
    parts = text.strip().replace(",", ".").split(":")
    if not 1 <= len(parts) <= 3 or not all(parts):
        raise ValueError(f"cannot read time {text!r}")
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + float(part)
    return seconds


@dataclass
class ManualRow:
    sub_index: int | None
    word: str
    action: str  # mute | skip | "" (not filled in yet)
    start: float | None
    end: float | None
    note: str


def load_manual(path: Path) -> list:
    if not path.is_file():
        return []
    rows = []
    with open(path, newline="") as f:
        lines = [line for line in f if not line.lstrip().startswith("#")]
    for number, raw in enumerate(csv.DictReader(lines), 2):
        action = (raw.get("action") or "").strip().lower()
        if action not in ("", "mute", "skip"):
            raise ValueError(f"{path}: row {number}: action must be mute or skip, not {action!r}")
        index = (raw.get("sub_index") or "").strip()
        start = end = None
        if action == "mute":
            try:
                start, end = parse_time(raw.get("start") or ""), parse_time(raw.get("end") or "")
            except ValueError as e:
                raise ValueError(f"{path}: row {number}: {e}") from None
            if end <= start:
                raise ValueError(f"{path}: row {number}: end must be after start")
        rows.append(
            ManualRow(int(index) if index else None, (raw.get("word") or "").strip(), action, start, end,
                      (raw.get("note") or "").strip())
        )  # fmt: skip
    return rows


def manual_key(sub_index, word):
    return (sub_index, target_key(word))


def _write_csv(path: Path, header, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def _one_line(text: str) -> str:
    return " / ".join(part.strip() for part in text.splitlines() if part.strip())


def mute_rows(mutes):
    for m in sorted(mutes, key=lambda m: m.mute_start):
        yield [
            fmt_time(m.mute_start), fmt_time(m.mute_end), m.word, m.heard, m.match,
            "" if m.word_start is None else f"{m.word_start:.3f}", "" if m.word_end is None else f"{m.word_end:.3f}",
            f"{m.pad:.3f}", "" if m.probability is None else f"{m.probability:.2f}", m.source,
            m.sub_index or "", "" if m.line_start is None else fmt_time(m.line_start),
            "" if m.line_end is None else fmt_time(m.line_end), _one_line(m.text),
        ]  # fmt: skip


def _time_or_blank(seconds):
    return "" if seconds is None else fmt_time(seconds)


def write_manual(path: Path, manual_rows, review):
    """Rewrite manual.csv: every row with an action is kept, then one empty-action row per word still in review."""
    kept = [r for r in manual_rows if r.action]
    if not kept and not review and not path.exists():
        return
    partial = path.with_name(path.name + ".partial")
    with open(partial, "w", newline="") as f:
        f.write(MANUAL_HELP)
        writer = csv.writer(f)
        writer.writerow(MANUAL_HEADER)
        for r in kept:
            writer.writerow(
                [r.action, "" if r.sub_index is None else r.sub_index, r.word, _time_or_blank(r.start),
                 _time_or_blank(r.end), r.note]
            )  # fmt: skip
        for item in sorted(review, key=lambda r: (r.line_start, r.suggest_start or 0.0)):
            note = f"line {fmt_time(item.line_start)}: {_one_line(item.text)}"
            if item.suggest_start is not None:
                note = f"suggested from {item.basis} | {note}"
            writer.writerow(
                ["", item.sub_index, item.word, _time_or_blank(item.suggest_start), _time_or_blank(item.suggest_end),
                 note]
            )  # fmt: skip
    partial.replace(path)


def write_reports(report_dir: Path, *, status: dict, mutes, review, clean_subs: Path | None, manual_rows):
    """(Re)write this episode's report folder."""
    report_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(report_dir / "mutes.csv", MUTES_HEADER, mute_rows(mutes))

    clips_dir = report_dir / "clips"
    if clips_dir.is_dir():
        for old in clips_dir.glob("*.wav"):
            old.unlink()
    rows = []
    for item in sorted(review, key=lambda r: (r.line_start, r.suggest_start or 0.0)):
        names = {}
        for kind, clip in item.clips.items():
            if Path(clip).is_file():
                clips_dir.mkdir(exist_ok=True)
                names[kind] = f"clips/{Path(clip).name}"
                shutil.copyfile(clip, report_dir / names[kind])
        rows.append(
            [fmt_time(item.line_start), _one_line(item.text), item.word, item.instance, item.reason,
             _time_or_blank(item.suggest_start), _time_or_blank(item.suggest_end), item.basis,
             names.get("original", ""), names.get("muted-preview", ""), _time_or_blank(item.listen_start),
             item.sub_index, fmt_time(item.line_end)]
        )  # fmt: skip
    _write_csv(report_dir / "review.csv", REVIEW_HEADER, rows)

    if clean_subs and clean_subs.is_file():
        shutil.copyfile(clean_subs, report_dir / "clean.srt")
    write_manual(report_dir / "manual.csv", manual_rows, review)

    with open(report_dir / "status.json", "w") as f:
        json.dump(status, f, indent=2)
        f.write("\n")


def append_mute_log(log_path: Path, episode: str, output, mutes):
    """One line per mute, appended across all runs."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    with open(log_path, "a") as f:
        for m in sorted(mutes, key=lambda m: m.mute_start):
            f.write(
                f"{now}\t{episode}\t{fmt_time(m.mute_start)}\t{fmt_time(m.mute_end)}\t{m.word}\t{m.source}\t{output}\n"
            )
