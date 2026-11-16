"""Audio probing and clip cutting (ffmpeg/ffprobe). The source file is only ever read."""

import json
import subprocess

_CENTRE_LAYOUTS = ("3.0", "3.1", "4.0", "4.1", "5.0", "5.1", "6.0", "6.1", "7.0", "7.1")


def duration_seconds(path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    return float(json.loads(out)["format"]["duration"])


def pick_audio_stream(streams, stream_index=None):
    """Same choice upstream's mux makes: the only audio stream, or the one whose stream index was given.

    Returns (position among the audio streams, stream info).
    """
    if not streams:
        raise ValueError("no audio streams found")
    if stream_index is None:
        if len(streams) > 1:
            raise ValueError("multiple audio streams: set cleanvid.audio_stream_index in the config")
        return 0, streams[0]
    for position, stream in enumerate(streams):
        if stream.get("index", -1) == stream_index:
            return position, stream
    raise ValueError(f"audio stream index {stream_index} is not an audio stream of this file")


def has_centre_channel(stream) -> bool:
    return str(stream.get("channel_layout", "")).startswith(_CENTRE_LAYOUTS)


def cut_clip(source, audio_position, start, length, dest, centre_only=False):
    """Write `length` seconds of one audio stream from `start` as 16 kHz mono WAV."""
    base = [
        "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
        "-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", str(source),
        "-map", f"0:a:{audio_position}", "-vn", "-sn",
    ]  # fmt: skip
    tail = ["-ar", "16000", "-c:a", "pcm_s16le", str(dest)]
    attempts = [["-af", "pan=mono|c0=FC"], ["-ac", "1"]] if centre_only else [["-ac", "1"]]
    error = ""
    for mix in attempts:
        done = subprocess.run(base + mix + tail, capture_output=True, text=True)
        if done.returncode == 0:
            return
        error = done.stderr.strip()
    raise RuntimeError(f"ffmpeg could not cut the clip: {error[-300:]}")


def cut_listen_clip(source, audio_position, start, length, dest, silence=None):
    """Write a clip for a person to listen to: the normal stereo mix, as WAV.

    silence: (start, end) in episode time to mute inside the clip, to preview what a mute would sound like.
    """
    filters = []
    if silence:
        a, b = max(0.0, silence[0] - start), min(length, silence[1] - start)
        if b > a:
            filters = ["-af", f"volume=enable='between(t,{a:.3f},{b:.3f})':volume=0"]
    done = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
         "-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", str(source),
         "-map", f"0:a:{audio_position}", "-vn", "-sn", "-ac", "2", *filters, "-c:a", "pcm_s16le", str(dest)],
        capture_output=True, text=True,
    )  # fmt: skip
    if done.returncode != 0:
        raise RuntimeError(f"ffmpeg could not cut the clip: {done.stderr.strip()[-300:]}")
