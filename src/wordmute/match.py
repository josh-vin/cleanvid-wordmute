"""Find cleanvid's flagged words in a word-level transcription of a subtitle line's audio."""

import difflib
import re
from dataclasses import dataclass, field


@dataclass
class Word:
    text: str
    start: float
    end: float
    probability: float = 1.0


@dataclass
class Hit:
    """One heard instance of a flagged word (times are absolute episode seconds)."""

    target: str
    heard: str
    kind: str  # exact | joined | censored | variant | between | aligned
    start: float
    end: float
    probability: float | None
    why: str = ""  # for a doubtful instance: what is wrong with it


@dataclass
class Missing:
    target: str
    count: int  # how many instances the subtitle has that could not be placed
    reason: str


@dataclass
class LineResult:
    hits: list = field(default_factory=list)
    missing: list = field(default_factory=list)
    doubtful: list = field(default_factory=list)  # heard instances that were not trusted (Hit with `why`)


@dataclass
class SubToken:
    """One word of subtitle text. Words of a flagged match share `group` and carry its target_key."""

    norm: str
    target: str | None = None
    group: int | None = None
    in_line: bool = True


@dataclass
class Between:
    """Where a flagged word must be, going by the heard words either side of it."""

    target: str
    group: int
    start: float | None  # end of the nearest heard word before it, if there is one
    end: float | None  # start of the nearest heard word after it, if there is one
    before: str
    after: str
    heard: str  # what the transcript has in that gap
    extra_words: int  # other subtitle words that fall in the same gap


def normalize(text: str) -> str:
    """Lower-case and drop everything but letters, digits and '*' ("Fuckin'," -> "fuckin")."""
    return re.sub(r"[^a-z0-9*]", "", text.lower())


def target_key(matched_text: str) -> str:
    """A flagged subtitle match as one comparable string ("son of a bitch" -> "sonofabitch")."""
    return normalize(matched_text)


def clean_text(text: str) -> str:
    """Subtitle text without styling tags, sound descriptions, speaker dashes and music notes."""
    text = re.sub(r"\{[^}]*\}|<[^>]*>|\[[^\]]*\]|\([^)]*\)|[♪♫#]", " ", text)
    lines = [re.sub(r"^\s*-\s*", "", part).strip() for part in text.splitlines()]
    return " ".join(part for part in lines if part)


def tokenize(text: str, pattern, first_group=0, in_line=True) -> list:
    """clean_text(text) as SubTokens; `pattern` is cleanvid's flagged-word expression."""
    text = clean_text(text)
    flagged = [(m.start(), m.end(), target_key(m.group())) for m in pattern.finditer(text)]
    tokens = []
    for word in re.finditer(r"[^\s]+", text):
        norm = normalize(word.group())
        if not norm:
            continue
        token = SubToken(norm, in_line=in_line)
        for number, (start, end, target) in enumerate(flagged):
            if word.start() < end and start < word.end():
                token.target, token.group = target, first_group + number
                break
        tokens.append(token)
    return tokens


def locate_between(tokens, words) -> list:
    """Bound each flagged word the transcript does not have by the heard words around it.

    tokens: SubTokens of the subtitle lines covering the clip, in order.
    words:  Word list heard in the clip, episode time.
    The subtitle text and the transcript are lined up word for word; a flagged word of the line that has
    no heard counterpart (or one without usable timing) lies between its nearest matched neighbours.
    Returns a Between per such word, with None on a side that has no matched neighbour.
    """
    heard = [(normalize(w.text), w) for w in words]
    heard = [(n, w) for n, w in heard if n]
    blocks = difflib.SequenceMatcher(None, [t.norm for t in tokens], [n for n, _ in heard], autojunk=False)
    to_heard = {}
    for a, b, size in blocks.get_matching_blocks():
        for k in range(size):
            to_heard[a + k] = b + k

    def timed(i):
        return i in to_heard and heard[to_heard[i]][1].end > heard[to_heard[i]][1].start

    found = []
    groups = sorted({t.group for t in tokens if t.group is not None and t.in_line})
    for group in groups:
        members = [i for i, t in enumerate(tokens) if t.group == group]
        if all(timed(i) for i in members):
            continue  # heard with timing: resolve_line's business
        left = next((i for i in range(members[0] - 1, -1, -1) if timed(i)), None)
        right = next((i for i in range(members[-1] + 1, len(tokens)) if timed(i)), None)
        lo = to_heard[left] if left is not None else -1
        hi = to_heard[right] if right is not None else len(heard)
        inside = range(left + 1 if left is not None else 0, right if right is not None else len(tokens))
        found.append(
            Between(
                tokens[members[0]].target, group,
                heard[lo][1].end if left is not None else None,
                heard[hi][1].start if right is not None else None,
                heard[lo][1].text.strip() if left is not None else "",
                heard[hi][1].text.strip() if right is not None else "",
                " ".join(w.text.strip() for _, w in heard[lo + 1 : hi]),
                sum(1 for i in inside if tokens[i].group is None),
            )
        )  # fmt: skip
    return found


def _censored_matches(heard: str, target: str) -> bool:
    # "f***" / "f**king" as written by the transcriber stand for a word it would not spell out
    if "*" not in heard or not heard.strip("*"):
        return False
    pattern = "".join(".+" if part == "" else re.escape(part) for part in re.split(r"\*+", heard))
    pattern = re.sub(r"(\.\+)+", ".+", pattern)
    return re.fullmatch(pattern, target) is not None


def _find(target, words, used, is_swear, variants):
    """Yield (first, last, kind) index runs of `words` that are an instance of `target`."""
    max_run = 4
    i = 0
    while i < len(words):
        found = None
        if i not in used:
            norm = words[i][0]
            if not variants:
                joined = ""
                for j in range(i, min(i + max_run, len(words))):
                    if j in used:
                        break
                    joined += words[j][0]
                    if joined == target:
                        found = (i, j, "exact" if j == i else "joined")
                        break
                    if not target.startswith(joined):
                        break
                if found is None and _censored_matches(norm, target):
                    found = (i, i, "censored")
            elif norm != target and is_swear(norm) and (norm.startswith(target) or target.startswith(norm)):
                # e.g. subtitle "fuckin" heard as "fucking": a different spelling of the same flagged word
                found = (i, i, "variant")
        if found:
            yield found
            i = found[1] + 1
        else:
            i += 1


def resolve_line(targets, words, line_start, line_end, *, is_swear, tolerance, min_probability, max_word_seconds):
    """Match one subtitle line's flagged words against what was heard.

    targets: target_key() of each flagged match in the line, in order (repeats allowed).
    words:   Word list for the line's clip, times already shifted to episode time.
    Every heard instance of a flagged word inside the line's window is a hit. A target with fewer
    trusted instances than the subtitle has is reported missing; nothing is ever guessed.
    """
    lo, hi = line_start - tolerance, line_end + tolerance
    window = [(normalize(w.text), w) for w in words if lo <= (w.start + w.end) / 2.0 <= hi]
    window = [(n, w) for n, w in window if n]
    heard_text = " ".join(w.text.strip() for _, w in window) or "(nothing)"

    wanted = {}
    for t in targets:
        wanted[t] = wanted.get(t, 0) + 1

    result = LineResult()
    used = set()
    trusted = {t: 0 for t in wanted}
    untrusted = {t: [] for t in wanted}

    def take(target, variants):
        for first, last, kind in _find(target, window, used, is_swear, variants):
            run = [w for _, w in window[first : last + 1]]
            used.update(range(first, last + 1))
            start, end = run[0].start, run[-1].end
            prob = min(w.probability for w in run)
            heard = " ".join(w.text.strip() for w in run)
            length = end - start
            why = ""
            if length <= 0:
                why = f'heard "{heard}" but it has no length (start={start:.3f} end={end:.3f})'
            elif length > max_word_seconds * len(run):
                why = f'heard "{heard}" but the timing is implausible ({length:.2f}s long)'
            elif prob < min_probability:
                why = f'heard "{heard}" at {start:.3f}-{end:.3f} but confidence is low ({prob:.2f} < {min_probability})'
            if why:
                untrusted[target].append(why)
                result.doubtful.append(Hit(target, heard, kind, start, end, prob, why))
            else:
                trusted[target] += 1
                result.hits.append(Hit(target, heard, kind, start, end, prob))

    # longest targets first, so "asshole" is claimed before "ass" can take it as a variant
    order = sorted(wanted, key=len, reverse=True)
    for target in order:
        take(target, variants=False)
    for target in order:
        if trusted[target] + len(untrusted[target]) < wanted[target]:
            take(target, variants=True)

    for target in wanted:
        short = wanted[target] - trusted[target]
        if short <= 0:
            continue
        reasons = list(untrusted[target])
        not_heard = short - len(reasons)
        if not_heard > 0:
            count = f"{trusted[target]} of {wanted[target]}" if wanted[target] > 1 else "not"
            reasons.append(f'"{target}" {count} heard in the clip; transcript there: {heard_text}')
        result.missing.append(Missing(target, short, "; ".join(reasons)))
    result.hits.sort(key=lambda h: h.start)
    return result


@dataclass
class Guess:
    """A span that may hold a flagged word the transcription did not time."""

    target: str
    start: float
    end: float
    how: str  # which way of finding it this came from
    heard: str = ""
    probability: float | None = None
    strong: bool = False  # sure enough to mute on its own
    bound: bool = False  # the word is somewhere inside this span, rather than being this span
    basis: str = ""  # said to the reviewer when it is only a suggestion


def settle_guesses(guesses, wanted, taken, max_seconds):
    """Choose spans for `wanted` missing instances of one flagged word.

    guesses: Guess list for that word, best first.
    taken:   (start, end) spans already muted for it; a guess overlapping one is that same instance again.
    A guess is accepted when it is strong, or when a guess made a different way overlaps it. Guesses that
    agree are merged into one span while that stays within max_seconds; a bound that agrees only confirms.
    Returns (accepted, leftover); an accepted Guess's `how` names every way that agreed on it.
    """

    def overlap(a, b):
        return a[0] < b[1] and b[0] < a[1]

    fresh = [g for g in guesses if g.end > g.start and not any(overlap((g.start, g.end), t) for t in taken)]
    accepted, used = [], set()
    for i, guess in enumerate(fresh):
        if len(accepted) >= wanted:
            break
        if i in used or any(overlap((guess.start, guess.end), (a.start, a.end)) for a in accepted):
            continue
        agreeing = [
            j for j, other in enumerate(fresh)
            if j != i and j not in used and other.how != guess.how and not (guess.bound and other.bound)
            and overlap((guess.start, guess.end), (other.start, other.end))
        ]  # fmt: skip
        if not guess.strong and not agreeing:
            continue
        start, end, ways = guess.start, guess.end, [guess.how]
        for j in agreeing:
            other = fresh[j]
            if not other.bound and max(end, other.end) - min(start, other.start) <= max_seconds:
                start, end = min(start, other.start), max(end, other.end)
            ways.append(other.how)
        used.update([i, *agreeing])
        accepted.append(
            Guess(guess.target, start, end, "+".join(dict.fromkeys(ways)), guess.heard, guess.probability, True)
        )
    leftover = [
        g for i, g in enumerate(fresh)
        if i not in used and not any(overlap((g.start, g.end), (a.start, a.end)) for a in accepted)
    ]  # fmt: skip
    return accepted, leftover


def merge_spans(spans, gap=0.02):
    """Sort (start, end) spans and merge any that overlap or sit closer than `gap` seconds."""
    merged = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1] + gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged
