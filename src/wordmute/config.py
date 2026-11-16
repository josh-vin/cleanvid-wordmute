import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path


@dataclass
class Config:
    output_root: str = ""
    review_root: str = ""
    reports_root: str = ""
    work_dir: str = ""
    name_template: str = "{show} - S{season:02d}E{episode:02d}"

    clip_padding: float = 1.0
    clip_channel: str = "auto"

    engine: str = "stable-ts"
    model: str = "medium.en"
    device: str = "cpu"
    compute_type: str = "int8"
    language: str = "en"
    cpu_threads: int = 0
    min_word_probability: float = 0.1
    max_word_seconds: float = 2.0

    word_pad: float = 0.10
    window_tolerance: float = 0.5

    retry_with_mix: bool = True
    between: bool = True
    between_max_seconds: float = 1.5
    between_max_extra_words: int = 2
    relisten: bool = True
    relisten_min_probability: float = 0.5
    align: bool = True
    align_min_probability: float = 0.5

    whole_line: bool = False
    whole_line_other_words: int = 1
    whole_line_max_seconds: float = 6.0
    whole_line_pad: float = 0.25
    mute_best_guess: bool = False
    hold_unheard: bool = True

    review_clip_margin: float = 0.75

    subtitle_language: str = "eng"
    swears_file: str = ""
    extra_swears: list = field(default_factory=list)
    embed_subs: bool = True
    full_subs: bool = True
    tidy_subs: bool = True
    audio_params: str = "-c:a eac3 -b:a 640k"
    audio_stream_index: int | None = None

    extra: dict = field(default_factory=dict)

    @property
    def review_dir(self) -> Path:
        return Path(self.review_root) if self.review_root else Path(self.output_root) / "_needs-review"


# toml section -> {toml key: Config field}
_LAYOUT = {
    "paths": {k: k for k in ("output_root", "review_root", "reports_root", "work_dir", "name_template")},
    "clip": {"padding": "clip_padding", "channel": "clip_channel"},
    "transcribe": {
        k: k
        for k in (
            "engine",
            "model",
            "device",
            "compute_type",
            "language",
            "cpu_threads",
            "min_word_probability",
            "max_word_seconds",
        )
    },
    "mute": {"word_pad": "word_pad", "window_tolerance": "window_tolerance"},
    "find": {
        k: k
        for k in (
            "retry_with_mix",
            "between",
            "between_max_seconds",
            "between_max_extra_words",
            "relisten",
            "relisten_min_probability",
            "align",
            "align_min_probability",
        )
    },
    "unheard": {
        "whole_line": "whole_line",
        "whole_line_other_words": "whole_line_other_words",
        "whole_line_max_seconds": "whole_line_max_seconds",
        "whole_line_pad": "whole_line_pad",
        "mute_best_guess": "mute_best_guess",
        "hold": "hold_unheard",
    },
    "review": {"clip_margin": "review_clip_margin"},
    "cleanvid": {
        k: k
        for k in ("subtitle_language", "swears_file", "extra_swears", "embed_subs", "full_subs", "audio_params",
                  "audio_stream_index")
    },
    "subtitles": {"tidy": "tidy_subs"},
}


def load_config(path) -> Config:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    cfg = Config()
    known = {f.name for f in fields(Config)}
    for section, values in raw.items():
        mapping = _LAYOUT.get(section)
        if mapping is None:
            raise ValueError(f"{path}: unknown section [{section}]")
        for key, value in values.items():
            if key not in mapping or mapping[key] not in known:
                raise ValueError(f"{path}: unknown setting {section}.{key}")
            setattr(cfg, mapping[key], value)
    for name in ("output_root", "review_root", "reports_root", "work_dir"):
        # ${WORDMUTE_DATA} and the like, so the file need not name a machine's folders
        value = os.path.expandvars(getattr(cfg, name))
        if "$" in value:
            raise ValueError(f"{path}: paths.{name} uses an environment variable that is not set: {value}")
        setattr(cfg, name, value)
    for name in ("output_root", "reports_root", "work_dir"):
        if not getattr(cfg, name):
            raise ValueError(f"{path}: paths.{name} must be set")
    if min(cfg.clip_padding, cfg.word_pad, cfg.window_tolerance, cfg.review_clip_margin, cfg.whole_line_pad) < 0:
        raise ValueError(f"{path}: padding values cannot be negative")
    return cfg
