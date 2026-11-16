"""Word-level transcription behind a small interface, so the engine can be swapped (e.g. for WhisperX)."""

import contextlib
import io
import wave
from typing import Protocol

from .match import Word


def load_wav(wav_path):
    """A clip from clips.cut_clip (16 kHz mono 16-bit WAV) as float32 samples.

    Passing samples instead of a path keeps faster-whisper from decoding the file with PyAV, whose
    open() arguments it depends on change between PyAV releases.
    """
    import numpy as np

    with wave.open(str(wav_path), "rb") as clip:
        if (clip.getframerate(), clip.getnchannels(), clip.getsampwidth()) != (16000, 1, 2):
            raise ValueError(f"{wav_path}: expected a 16 kHz mono 16-bit WAV")
        frames = clip.readframes(clip.getnframes())
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0


class Transcriber(Protocol):
    name: str

    def transcribe(self, wav_path: str) -> list[Word]:
        """Words heard in the clip, with start/end in seconds from the start of the clip."""

    # Optional, used by the pipeline when the transcriber has them; both are given the subtitle text:
    #   relisten(wav_path, text) -> list[Word]   transcribe again, primed with what the subtitle says
    #   align(wav_path, text) -> list[Word]      the words of the text itself, timed against the clip


class StableTsTranscriber:
    """stable-ts on the faster-whisper backend."""

    def __init__(self, cfg):
        import stable_whisper

        options = {"device": cfg.device, "compute_type": cfg.compute_type}
        if cfg.cpu_threads:
            options["cpu_threads"] = cfg.cpu_threads
        self.model = stable_whisper.load_faster_whisper(cfg.model, **options)
        self.language = cfg.language
        self.name = f"stable-ts {stable_whisper.__version__} / faster-whisper {cfg.model} ({cfg.device}, {cfg.compute_type})"

    @staticmethod
    def _words(result):
        if result is None:
            return []
        return [
            Word(w.word, float(w.start), float(w.end), float(w.probability if w.probability is not None else 0.0))
            for segment in result.segments
            for w in segment.words
        ]

    def transcribe(self, wav_path, prompt=None):
        return self._words(
            self.model.transcribe(
                load_wav(wav_path),
                language=self.language,
                word_timestamps=True,
                condition_on_previous_text=False,
                initial_prompt=prompt,
                verbose=None,
            )
        )

    def relisten(self, wav_path, text):
        return self.transcribe(wav_path, prompt=text)

    def align(self, wav_path, text):
        with contextlib.redirect_stderr(io.StringIO()):  # stable-ts draws a progress bar here even when quiet
            return self._words(self.model.align(load_wav(wav_path), text, language=self.language, verbose=None))


def make_transcriber(cfg) -> Transcriber:
    if cfg.engine == "stable-ts":
        return StableTsTranscriber(cfg)
    raise ValueError(f"unknown transcribe.engine {cfg.engine!r} (available: stable-ts)")
