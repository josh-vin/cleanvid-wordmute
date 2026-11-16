# cleanvid-wordmute

[cleanvid](https://github.com/mmguero/cleanvid) mutes the whole subtitle line that contains a flagged
word. This wraps cleanvid so that only the word itself is muted.

It is a wrapper, not a fork: cleanvid 1.7.1 is installed unmodified from PyPI and still does the
subtitle scanning, the word list, the masked subtitles and the ffmpeg mux. `WordMuteCleaner`
subclasses its `VidCleaner` and replaces one thing, the list of time spans handed to the mute step.

## How it works

For each subtitle line cleanvid flags:

1. Cut that line's audio with `clip.padding` seconds either side (1 s by default).
2. Transcribe the clip with word-level timestamps (stable-ts on faster-whisper).
3. Find the flagged word among the words heard within the line's own time span, add the clip's
   start offset, and mute that span plus `mute.word_pad` (0.10 s) either side.

The subtitles are trusted, so a flagged word that was not found that way is looked for again
(`[find]` in `config.toml`), each way giving a guess at where it is:

- `heard`: it was heard, but with too little confidence or no usable length.
- `between`: the subtitle text is lined up with what was heard; a word heard as something else
  ("ship" for "shit") or not at all lies between the words either side of it that were heard.
- `…/mix`: the same again on the full mix, when the first pass used the centre channel.
- `relisten`, `relisten/tight`: transcribed again with the subtitle text given as context, on the
  padded clip and on the line alone.
- `aligned`: the subtitle text forced onto the audio.

A guess is muted when it is sure on its own (a short gap with nothing else in it, a confident
re-listen or alignment) or when two different ways put the word in the same place. The `match` column
of `mutes.csv` says which, e.g. `between` or `relisten+relisten/tight`.

What happens to a word that is still not timed is set under `[unheard]` in `config.toml`:

- `whole_line`: a line that is nothing but flagged words ("Fuck.", "Shit. Shit. Shit."), or has at most
  `whole_line_other_words` other words ("Fuck off.", "Oh, shit."), is muted from start to end
  (`whole-line` in `mutes.csv`). Lines with more said in them are never muted whole.
- `mute_best_guess`: otherwise the best guess is muted when there is one (`guess:…` in `mutes.csv`).
- `hold`: with no guess at all the word was most likely never said aloud, or was said with a
  neighbouring line and muted there. With `hold = false` the episode is finished anyway and the word is
  only listed in `review.csv`, with a clip of its line to listen to; with `hold = true` the episode is
  marked `needs_review` and kept out of the clean folder until `manual.csv` settles it.

With all three left out a word with no guess good enough holds the episode for review, with the best
guess as a suggestion. Words that were found are muted either way.

## Setting it up on your own machine

Needs Docker and bash, about 5 GB of free memory while it runs, and no GPU. On a 12-thread CPU an
episode takes roughly 8 minutes. Videos need English text subtitles, embedded or as an `.srt` beside
the video (image subtitles such as PGS cannot be read), and episode files need `S01E02` in their name.

    git clone https://github.com/josh-vin/cleanvid-wordmute.git
    cd cleanvid-wordmute

1. Pick a data folder for the output, reports and scratch files; it is the only media path mounted
   writable. Name it in a file called `local.env` beside `run.sh` (not kept in git):
   `echo 'WORDMUTE_DATA=/path/to/that/folder' > local.env`, or export `WORDMUTE_DATA` in your shell.
   `config.toml` puts `out/`, `reports/` and `work/` inside it (`[paths]`).
2. Set `transcribe.cpu_threads` in `config.toml` to suit your CPU.
3. `./run.sh --build`, then `./run.sh --test`.
4. `./run.sh --dry-run /path/to/an/episode.mkv` to see what it would mute. The first run downloads the
   speech model (about 1.5 GB) into `~/.cache/cleanvid-wordmute/models`.

Do not make `output_root` a folder your media server scans until you are happy with the results; copy
finished files across instead.

## Running

    ./run.sh --build                      # once, and after code changes
    ./run.sh --dry-run <video or folder>  # print the planned mutes; no video, no reports
    ./run.sh <video or folder>            # write the clean video and the reports
    ./run.sh --replace-review <video>     # again, after filling in manual.csv
    ./run.sh --mute 12:34.5-12:35.2 <video>   # redo a finished episode with one more span muted
    ./run.sh --redo <video>               # redo a finished episode (new settings, edited manual.csv)
    ./run.sh --test                       # test suite

Everything runs in a container (the host needs only Docker). Inputs are mounted read-only; the only
writable media path is the data folder named by `WORDMUTE_DATA`.

Exit code: 0 all clean, 2 at least one episode needs review, 1 an error.

## Output

    <output_root>/<Show>/Season NN/<Show> - SxxEyy.mkv                 episodes with every word placed
    <output_root>/_needs-review/<Show>/Season NN/<Show> - SxxEyy.mkv   episodes with unresolved words
    <reports_root>/<Show>/Season NN/<Show> - SxxEyy/
        mutes.csv      every mute: span, word, what was heard, match type, pad, confidence, subtitle line
        review.csv     words needing a person: time, subtitle text, word, reason, suggested span, clips
        status.json    "ok" or "needs_review", counts, settings and versions used
        clips/         per review word: ..._original.wav and, with a suggestion, ..._muted-preview.wav
        clean.srt      cleanvid's masked subtitles
        manual.csv     decisions and extra spans made by hand
    <reports_root>/mutes.log    one line per mute, appended on every run

An episode whose clean file exists is skipped unless `--redo` or `--mute` is given, and nothing is
ever written into a source folder.

## Finishing an episode that needs review

Each word in `review.csv` has clips in `clips/`, named `line<number>_<time>_<word>_…`:

- `…_original.wav`: the untouched audio of the word's subtitle line (and of the suggested span plus
  `review.clip_margin`, 0.75 s, where that reaches further).
- `…_muted-preview.wav`: the same audio with the suggested span silenced, i.e. what the episode would
  sound like if the suggestion were accepted. Nothing is muted in the video for this word yet.

`manual.csv` has one row per review word with the suggested `start`/`end` already filled in. `action`
is the first column: the decision is typed at the very start of the row, before its first comma. If the
preview has the word gone and little else, type `mute`. If not, correct `start`/`end` (episode time,
seconds or `H:MM:SS.mmm`; `clips_start_at` in `review.csv` is where the clips begin) and type `mute`,
or type `skip` if the word is not spoken at all. Rows without an action are
rewritten on every run; rows with one are kept. Then rerun with `--replace-review`: transcriptions
are cached, so only the mux is repeated. Once nothing is left to review the file is written to the
clean folder and the earlier `_needs-review` copy is removed. Manual spans are muted exactly as
given, with no extra pad.

## Fixing something noticed while watching

    ./run.sh --mute 12:34.5-12:35.2 <source video>

adds a `mute` row for that span to the episode's `manual.csv` (so it is applied on every later run
too) and writes the clean file again. `--mute` can be repeated. To take a mute out or change one, edit
`manual.csv` (a `mute` row needs only `action`, `start`, `end`: `mute,,,START,END,`) and run `./run.sh --redo <source video>`.

## Word list

The list shipped with cleanvid is used unless `[cleanvid] swears_file` names another. Words it lacks
are added with `extra_swears` in `config.toml` (the shipped list has no "fuckity" or "fuckwitch", for
example). A word that is not in the list is neither muted nor masked in the subtitles.

## Subtitles

The masked subtitles in the clean file are made from a tidied copy of the source subtitles
(`[subtitles] tidy`): `{\an8}`-style positioning tags, which many players show as text, and invisible
or control characters are removed. `<i>` italics and accented letters are kept. The source file's own
subtitles are not touched.

## Settings

`config.toml` holds the paths, clip padding, word pad, model and the cleanvid options (subtitle
language, word list, audio encoding, embedded subtitles). The default is `medium.en` with `int8` on
CPU, and the image is built with CPU-only torch; for an NVIDIA GPU the Dockerfile needs the CUDA build
of torch, the container needs `--gpus all`, and the settings `device = "cuda"`,
`compute_type = "float16"` and a larger model. The transcriber sits behind a small interface (`transcribe.py`) so another engine such
as WhisperX can be added.

## Layout

    src/wordmute/cleaner.py     VidCleaner subclass; mute filters in upstream's exact format
    src/wordmute/match.py       finding flagged words in the transcription
    src/wordmute/pipeline.py    one episode end to end
    src/wordmute/clips.py       ffmpeg clip cutting
    src/wordmute/transcribe.py  transcriber interface and the stable-ts implementation
    src/wordmute/report.py      report files and manual spans
    tests/                      includes a check that the mute filters match upstream's text exactly

cleanvid is BSD-3-Clause, (c) Seth Grover. `build_mute_filters` in `cleaner.py` reproduces the
filter-formatting lines from its `CreateCleanSubAndMuteList`.
