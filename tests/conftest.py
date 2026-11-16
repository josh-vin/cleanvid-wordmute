import subprocess

import pytest

SRT = """1
00:00:01,000 --> 00:00:02,500
Hello there.

2
00:00:03,000 --> 00:00:05,000
Well, shit. That's a damn shame.

3
00:00:06,000 --> 00:00:07,000
Nothing to see here.

4
00:00:08,000 --> 00:00:09,500
Fuck! Fuck this.
"""


@pytest.fixture
def media(tmp_path):
    """A 12 s test video with a steady tone, an English subtitle file and a small word list."""
    video = tmp_path / "Show" / "Season 1" / "Show.S01E02.Test.mkv"
    video.parent.mkdir(parents=True)
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10:duration=12",
         "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=12",
         "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-ac", "2", str(video)],
        check=True,
    )  # fmt: skip
    subs = tmp_path / "subs.srt"
    subs.write_text(SRT)
    swears = tmp_path / "swears.txt"
    swears.write_text("shit\nfuck\ndamn|darn\n")
    return {"video": video, "subs": subs, "swears": swears, "tmp": tmp_path}
