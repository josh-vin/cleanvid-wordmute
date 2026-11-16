from pathlib import Path

import pytest

from wordmute.config import load_config
from wordmute.naming import derive
from wordmute.report import fmt_time, load_manual, parse_time

TEMPLATE = "{show} - S{season:02d}E{episode:02d}"


def test_show_and_season_from_library_path():
    n = derive(Path("/tv/Ted Lasso/Season 4/Ted.Lasso.S04E01.Home.1080p.WEB-DL.mkv"), TEMPLATE)
    assert (n.rel, n.name) == (Path("Ted Lasso/Season 04"), "Ted Lasso - S04E01")


def test_show_from_file_name_when_folder_does_not_say():
    n = derive(Path("/downloads/Ted.Lasso.S04E02.Something.mkv"), TEMPLATE)
    assert (n.rel, n.name) == (Path("Ted Lasso/Season 04"), "Ted Lasso - S04E02")
    assert derive(Path("/downloads/x.S01E01.mkv"), TEMPLATE, show="My Show").rel == Path("My Show/Season 01")


def test_film_keeps_its_name():
    n = derive(Path("/movies/Some Film (2020)/Some Film (2020).mkv"), TEMPLATE)
    assert (n.rel, n.name) == (Path("."), "Some Film (2020)")


def test_work_key_is_shell_safe_and_unique():
    a = derive(Path('/a/Bob\'s "Show" $1/Season 1/x S01E01.mkv'), TEMPLATE).work_key
    b = derive(Path('/b/Bob\'s "Show" $1/Season 1/x S01E01.mkv'), TEMPLATE).work_key
    assert a != b and all(c.isalnum() or c in "._-" for c in a)


def test_times():
    assert fmt_time(3723.0456) == "1:02:03.046"
    assert parse_time("1:02:03.046") == pytest.approx(3723.046)
    assert parse_time("83.5") == 83.5 and parse_time("01:23,5") == 83.5
    with pytest.raises(ValueError):
        parse_time("")


def test_manual_file(tmp_path):
    f = tmp_path / "manual.csv"
    f.write_text("# help\nsub_index,word,action,start,end,note\n4,fuck,mute,8.55,0:00:08.9,x\n9,shit,skip,,,\n7,ass,,,,todo\n")
    rows = load_manual(f)
    assert [(r.sub_index, r.word, r.action, r.start, r.end) for r in rows] == [
        (4, "fuck", "mute", 8.55, 8.9), (9, "shit", "skip", None, None), (7, "ass", "", None, None),
    ]  # fmt: skip
    f.write_text("sub_index,word,action,start,end,note\n4,fuck,mute,9,8,\n")
    with pytest.raises(ValueError):
        load_manual(f)


def test_load_wav_reads_clip_samples(tmp_path):
    import wave

    from wordmute.transcribe import load_wav

    path = tmp_path / "clip.wav"
    with wave.open(str(path), "wb") as clip:
        clip.setnchannels(1)
        clip.setsampwidth(2)
        clip.setframerate(16000)
        clip.writeframes((16384).to_bytes(2, "little", signed=True) * 8000)
    samples = load_wav(path)
    assert len(samples) == 8000
    assert abs(float(samples[0]) - 0.5) < 1e-6


_PATHS = '[paths]\noutput_root = "${WORDMUTE_DATA}/out"\nreports_root = "${WORDMUTE_DATA}/reports"\nwork_dir = "/scratch"\n'


def test_paths_take_the_data_folder_from_the_environment(tmp_path, monkeypatch):
    settings = tmp_path / "config.toml"
    settings.write_text(_PATHS)
    monkeypatch.setenv("WORDMUTE_DATA", "/data/clean")
    cfg = load_config(settings)
    assert (cfg.output_root, cfg.reports_root, cfg.work_dir) == ("/data/clean/out", "/data/clean/reports", "/scratch")
    assert cfg.review_dir == Path("/data/clean/out/_needs-review")


def test_a_path_naming_an_unset_variable_is_refused(tmp_path, monkeypatch):
    settings = tmp_path / "config.toml"
    settings.write_text(_PATHS)
    monkeypatch.delenv("WORDMUTE_DATA", raising=False)
    with pytest.raises(ValueError, match="not set"):
        load_config(settings)
