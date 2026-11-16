import argparse
import os
import sys
from pathlib import Path

from .config import load_config
from .pipeline import EpisodeResult, Refused, process
from .report import fmt_time, parse_time

VIDEO_SUFFIXES = {".mkv", ".mp4", ".m4v", ".avi", ".mov", ".ts"}


def find_videos(inputs):
    videos = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            for found in sorted(path.rglob("*")):
                hidden = any(part.startswith((".", "_needs-review")) for part in found.relative_to(path).parts)
                if found.is_file() and found.suffix.lower() in VIDEO_SUFFIXES and not hidden:
                    videos.append(found)
        else:
            videos.append(path)
    return videos


def _one_line(text):
    return " / ".join(part.strip() for part in text.splitlines() if part.strip())


def print_result(result: EpisodeResult):
    head = f"== {result.name}: "
    if result.status in ("skipped", "error"):
        print(f"{head}{result.status.upper()} - {result.message}")
        return
    planned = "planned (dry run, no video written)" if result.dry_run else "done"
    held = [r for r in result.review if r.hold]
    print(
        f"{head}{'NEEDS REVIEW' if result.status == 'needs_review' else 'OK'} - {planned}: "
        f"{result.flagged_lines} flagged lines, {len(result.mutes)} mutes, {len(held)} for review"
        + (f", {len(result.review) - len(held)} not heard and left alone" if len(result.review) > len(held) else "")
        + (f", {result.settled_by_hand} settled by hand" if result.settled_by_hand else "")
    )
    for m in sorted(result.mutes, key=lambda m: m.mute_start):
        how = "manual"
        if m.source != "manual":
            how = f'heard "{m.heard}", {m.match}' + ("" if m.probability is None else f", p={m.probability:.2f}")
        line = f" | line {m.sub_index}: {_one_line(m.text)}" if m.sub_index else ""
        print(f"  MUTE   {fmt_time(m.mute_start)} - {fmt_time(m.mute_end)}  {m.word!r} ({how}){line}")
    for r in result.review:
        which = f" ({r.instance})" if r.instance else ""
        print(f"  {'REVIEW' if r.hold else 'UNMUTED'} {fmt_time(r.line_start)}  line {r.sub_index}: {_one_line(r.text)!r} -> {r.word!r}{which}: {r.reason}")
        if r.suggest_start is not None:
            print(f"         suggested {fmt_time(r.suggest_start)} - {fmt_time(r.suggest_end)}, {r.basis}")
    if result.message:
        print(f"  note:    {result.message}")
    if not result.dry_run:
        print(f"  output:  {result.output}")
        print(f"  reports: {result.report_dir}")
    if result.seconds:
        print("  time:    " + ", ".join(f"{k} {v:.0f}s" for k, v in result.seconds.items() if v))


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="wordmute",
        description="cleanvid, but muting only the flagged word. Lines whose word cannot be timed go to review.",
    )
    parser.add_argument("inputs", nargs="+", help="video files, or folders to search for video files")
    parser.add_argument("-c", "--config", default=os.environ.get("WORDMUTE_CONFIG", "config.toml"))
    parser.add_argument("-n", "--dry-run", action="store_true", help="print the planned mutes; write no video or reports")
    parser.add_argument(
        "--replace-review",
        action="store_true",
        help="allow replacing this tool's earlier copy under the needs-review folder (never a clean file)",
    )
    parser.add_argument(
        "--redo",
        action="store_true",
        help="reprocess an episode this tool already finished and replace its output (never a source file)",
    )
    parser.add_argument(
        "--mute",
        action="append",
        default=[],
        metavar="START-END",
        help="also mute this span (episode time, e.g. 12:34.5-12:35.4); it is saved in the episode's manual.csv. "
        "Repeatable, single input only, implies --redo",
    )
    parser.add_argument("--subs", help="subtitle file to use (single input only)")
    parser.add_argument("--show", help="show name for the output folder, if it cannot be read from the path")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    videos = find_videos(args.inputs)
    if not videos:
        parser.error("no video files found")
    if args.subs and len(videos) != 1:
        parser.error("--subs needs exactly one input video")
    if args.mute and (len(videos) != 1 or args.dry_run):
        parser.error("--mute needs exactly one input video and cannot be combined with --dry-run")
    spans = []
    for text in args.mute:
        try:
            start, end = (parse_time(part) for part in text.split("-"))
        except ValueError:
            parser.error(f"--mute {text}: expected START-END, e.g. 12:34.5-12:35.4")
        spans.append((start, end))

    loaded = []

    def get_transcriber():
        if not loaded:
            from .transcribe import make_transcriber

            print(f"loading {cfg.engine} model {cfg.model} ({cfg.device}, {cfg.compute_type})...", flush=True)
            loaded.append(make_transcriber(cfg))
        return loaded[0]

    results = []
    for video in videos:
        try:
            result = process(
                video, cfg, get_transcriber, dry_run=args.dry_run, replace_review=args.replace_review,
                redo=args.redo or bool(spans), add_mutes=spans, subs=args.subs, show=args.show,
                log=lambda text: print(text, flush=True),
            )  # fmt: skip
        except (Refused, OSError, ValueError, RuntimeError) as e:
            result = EpisodeResult(source=Path(video), name=Path(video).stem, status="error", message=str(e))
        except Exception as e:  # one episode going wrong must not end a batch
            message = f"{type(e).__name__}: {e}"
            result = EpisodeResult(source=Path(video), name=Path(video).stem, status="error", message=message)
        print_result(result)
        sys.stdout.flush()
        results.append(result)

    counts = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    print("\nsummary: " + ", ".join(f"{count} {status}" for status, count in sorted(counts.items())))
    if "error" in counts:
        return 1
    return 2 if "needs_review" in counts else 0


if __name__ == "__main__":
    sys.exit(main())
