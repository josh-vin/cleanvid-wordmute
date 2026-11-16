"""cleanvid's VidCleaner with one change: the mute list is built from word spans, not subtitle lines.

Subtitle scanning, the word list, the cleaned subtitle file and the ffmpeg mux are all upstream's.
"""

import os
import re
from dataclasses import dataclass

import pysrt
from cleanvid.caselessdictionary import CaselessDictionary
from cleanvid.cleanvid import VidCleaner


@dataclass
class FlaggedLine:
    index: int  # subtitle number in the source subtitle file
    start: float
    end: float
    text: str
    matches: list  # the flagged words exactly as they appear in the subtitle, in order


def build_mute_filters(spans, final_peek):
    """ffmpeg audio filters muting each (start, end) span.

    Same filter text as the tail of upstream's VidCleaner.CreateCleanSubAndMuteList (cleanvid 1.7.1);
    tests/test_filters.py checks the two produce identical output for whole-line spans.
    """
    filters = []
    for i, (lineStart, lineEnd) in enumerate(spans):
        lineStartPeek = spans[i + 1][0] if i + 1 < len(spans) else final_peek
        filters.append(
            "afade=enable='between(t,"
            + format(lineStart, '.3f')
            + ","
            + format(lineEnd, '.3f')
            + ")':t=out:st="
            + format(lineStart, '.3f')
            + ":d=10ms"
        )
        filters.append(
            "afade=enable='between(t,"
            + format(lineEnd, '.3f')
            + ","
            + format(lineStartPeek, '.3f')
            + ")':t=in:st="
            + format(lineEnd, '.3f')
            + ":d=10ms"
        )
    return filters


def _seconds(srt_time):
    return pysrt.SubRipTime.from_string(srt_time).ordinal / 1000.0


class WordMuteCleaner(VidCleaner):
    """Runs upstream's scan, then asks `planner` which spans to mute instead of whole lines.

    planner(flagged_lines) -> list of (start, end) seconds, already padded and merged.
    """

    def __init__(self, *args, planner, final_peek, **kwargs):
        kwargs["swearsPadSec"] = 0.0  # line padding is meaningless here; word padding is the planner's job
        kwargs["jsonDump"] = True  # upstream's own record of the lines it flagged
        super().__init__(*args, **kwargs)
        self.swearsMap = CaselessDictionary({})  # upstream keeps this on the class, shared between instances
        self.planner = planner
        self.finalPeek = final_peek
        self.flaggedLines = []
        self.subtitles = []  # every line of the source subtitles: (start, end, text)
        self.lineMuteTimeList = []
        self.muteSpans = []

    def SwearPattern(self):
        # the expression upstream compiles in CreateCleanSubAndMuteList
        return re.compile(r'\b(' + '|'.join(self.swearsMap.keys()) + r')\b', re.IGNORECASE)

    def CreateCleanSubAndMuteList(self):
        super().CreateCleanSubAndMuteList()
        self.lineMuteTimeList = list(self.muteTimeList)  # what upstream would have muted; kept for reference

        replacer = self.SwearPattern()
        numbers = {}
        self.subtitles = []
        for sub in pysrt.open(self.tmpSubsFileSpec):
            numbers.setdefault(sub.start.ordinal, sub.index)
            self.subtitles.append((sub.start.ordinal / 1000.0, sub.end.ordinal / 1000.0, sub.text))
        # done with upstream's UTF-8 copy; removing it here keeps a later clean-up of this object
        # from deleting the copy a rerun of the same episode has just made
        os.remove(self.tmpSubsFileSpec)
        self.tmpSubsFileSpec = ""
        self.flaggedLines = []
        for edit in self.jsonDumpList or []:
            start = pysrt.SubRipTime.from_string(edit['start'])
            self.flaggedLines.append(
                FlaggedLine(
                    index=numbers.get(start.ordinal, 0),
                    start=start.ordinal / 1000.0,
                    end=_seconds(edit['end']),
                    text=edit['old'],
                    matches=[m.group() for m in replacer.finditer(edit['old'])],
                )
            )

        self.muteSpans = list(self.planner(self.flaggedLines))
        self.muteTimeList = build_mute_filters(self.muteSpans, self.finalPeek)
        if not self.muteTimeList:
            # upstream's mux always maps a filtered audio stream, so give it a filter that changes nothing
            self.muteTimeList = ["anull"]
