import re

from wordmute.match import (
    Guess,
    Word,
    clean_text,
    locate_between,
    merge_spans,
    normalize,
    resolve_line,
    settle_guesses,
    target_key,
    tokenize,
)

SWEARS = ["ass", "asshole", "fuck", "fuckin", "fucking", "shit", "son of a bitch", "motherfucker", "goddamn"]
_RE = re.compile(r"\b(" + "|".join(SWEARS) + r")\b", re.IGNORECASE)


def is_swear(norm):
    return any(normalize(s) == norm for s in SWEARS)


def words(*items):
    return [Word(t, s, e, p[0] if p else 0.9) for t, s, e, *p in items]


def run(targets, heard, start=10.0, end=13.0, **kw):
    opts = dict(is_swear=is_swear, tolerance=0.5, min_probability=0.3, max_word_seconds=2.0)
    opts.update(kw)
    return resolve_line([target_key(t) for t in targets], heard, start, end, **opts)


def test_normalize():
    assert normalize(" Fuckin',") == "fuckin"
    assert target_key("Son of a bitch") == "sonofabitch"


def test_exact_word_gets_its_own_span():
    r = run(["shit"], words((" Oh,", 10.2, 10.5), (" shit.", 10.6, 11.0), (" Sorry", 11.4, 11.8)))
    assert [(h.target, h.kind, h.start, h.end) for h in r.hits] == [("shit", "exact", 10.6, 11.0)]
    assert r.missing == []


def test_not_heard_is_reported_never_guessed():
    r = run(["shit"], words((" Oh,", 10.2, 10.5), (" shoot.", 10.6, 11.0)))
    assert r.hits == []
    assert len(r.missing) == 1 and "not heard" in r.missing[0].reason and "shoot" in r.missing[0].reason


def test_two_in_subtitle_one_heard():
    r = run(["fuck", "fuck"], words((" Fuck", 10.1, 10.4), (" me", 10.5, 10.7)))
    assert len(r.hits) == 1
    assert len(r.missing) == 1 and "1 of 2" in r.missing[0].reason


def test_every_heard_instance_in_the_window_is_muted():
    r = run(["shit"], words((" Shit,", 10.1, 10.4), (" shit!", 10.6, 10.9)))
    assert len(r.hits) == 2 and r.missing == []


def test_words_outside_the_line_window_belong_to_other_lines():
    r = run(["shit"], words((" shit", 8.0, 8.3), (" hello", 10.5, 10.9)))
    assert r.hits == [] and len(r.missing) == 1


def test_phrase_and_split_words():
    heard = words((" son", 10.0, 10.2), (" of", 10.2, 10.3), (" a", 10.3, 10.35), (" bitch!", 10.35, 10.8))
    r = run(["son of a bitch"], heard)
    assert [(h.kind, h.start, h.end) for h in r.hits] == [("joined", 10.0, 10.8)]
    r = run(["motherfucker"], words((" mother", 10.0, 10.3), (" fucker", 10.3, 10.7)))
    assert [(h.kind, h.start, h.end) for h in r.hits] == [("joined", 10.0, 10.7)]
    r = run(["goddamn"], words((" God", 10.0, 10.3), (" damn", 10.3, 10.7), (" it", 10.7, 10.8)))
    assert [(h.heard, h.end) for h in r.hits] == [("God damn", 10.7)]


def test_variant_spelling_and_censored():
    r = run(["fuckin"], words((" fucking", 10.0, 10.4), (" hell", 10.5, 10.8)))
    assert [(h.kind, h.heard) for h in r.hits] == [("variant", "fucking")]
    r = run(["fuck"], words((" f***", 10.0, 10.4),))
    assert [h.kind for h in r.hits] == ["censored"]


def test_longer_word_is_not_stolen_by_shorter_target():
    r = run(["ass", "asshole"], words((" asshole", 10.0, 10.6),))
    assert [h.target for h in r.hits] == ["asshole"]
    assert [m.target for m in r.missing] == ["ass"]


def test_unreliable_timing_goes_to_review():
    r = run(["shit"], words((" shit", 10.5, 10.5),))
    assert r.hits == [] and "no length" in r.missing[0].reason
    r = run(["shit"], words((" shit", 10.5, 10.9, 0.1),))
    assert r.hits == [] and "confidence is low" in r.missing[0].reason
    r = run(["shit"], words((" shit", 10.0, 12.9),))
    assert r.hits == [] and "implausible" in r.missing[0].reason


def test_merge_spans():
    assert merge_spans([(5, 6), (1, 2), (1.9, 3), (3.01, 4)]) == [(1, 4), (5, 6)]


def test_untrusted_instances_are_kept_as_doubtful():
    r = run(["shit"], words((" shit", 10.5, 10.9, 0.05)))
    assert not r.hits and r.missing[0].count == 1
    assert [(d.target, d.start, d.end) for d in r.doubtful] == [("shit", 10.5, 10.9)] and "low" in r.doubtful[0].why


def test_subtitle_text_is_cleaned_and_tokenized():
    text = "{\\an8}<i>- Oh, shit.</i>\n- [groans] You son of a bitch! ♪"
    assert clean_text(text) == "Oh, shit. You son of a bitch!"
    tokens = tokenize(text, _RE, first_group=7)
    assert [(t.norm, t.group) for t in tokens] == [
        ("oh", None), ("shit", 7), ("you", None), ("son", 8), ("of", 8), ("a", 8), ("bitch", 8),
    ]  # fmt: skip
    assert tokens[3].target == "sonofabitch"


def heard(text, start=10.0, step=0.3, length=0.25):
    return [Word(" " + t, start + i * step, start + i * step + length, 0.9) for i, t in enumerate(text.split())]


def test_misheard_word_is_bounded_by_its_neighbours():
    tokens = tokenize("He's a stupid fucking asshole, mate.", _RE)
    (gap,) = [g for g in locate_between(tokens, heard("He's a stupid fucking castle mate")) if g.target == "asshole"]
    assert (gap.before, gap.after, gap.heard, gap.extra_words) == ("fucking", "mate", "castle", 0)
    assert (gap.start, gap.end) == (10.0 + 3 * 0.3 + 0.25, 10.0 + 5 * 0.3)


def test_unheard_word_with_other_words_missing_and_one_sided_gaps():
    tokens = tokenize("Loads do it to fuck with their heads.", _RE)
    (gap,) = locate_between(tokens, heard("Loads do it but we're their heads"))
    assert (gap.before, gap.after, gap.heard, gap.extra_words) == ("it", "their", "but we're", 2)
    # nothing heard after the word: no end to the gap
    (gap,) = locate_between(tokenize("Oh, shit.", _RE), heard("Oh"))
    assert gap.start == 10.25 and gap.end is None
    # a word heard without a length is bounded too; a neighbouring line's words can be the bounds
    tokens = tokenize("You fucking ass?", _RE) + tokenize("Yeah, good one.", _RE, 100, in_line=False)
    spoken = heard("You fucking ass? Yeah, good")
    spoken[2].end = spoken[2].start
    (gap,) = locate_between(tokens, spoken)
    assert (gap.target, gap.before, gap.after) == ("ass", "fucking", "Yeah,")
    # flagged words of the neighbouring line are not this line's business
    assert locate_between(tokenize("Nice.", _RE) + tokenize("Shit.", _RE, 100, in_line=False), heard("Nice")) == []


def test_guesses_need_to_be_strong_or_to_agree():
    weak = Guess("hell", 38.75, 38.93, "relisten")
    assert settle_guesses([weak], 1, [], 1.5) == ([], [weak])
    # two different ways agreeing: merged; a bound that agrees confirms without widening
    tight = Guess("hell", 38.63, 38.91, "relisten/tight")
    bound = Guess("hell", 38.03, 39.19, "between", bound=True)
    (got,), rest = settle_guesses([weak, tight, bound], 1, [], 1.5)
    assert (got.start, got.end, got.how) == (38.63, 38.93, "relisten+relisten/tight+between") and rest == []
    # the same way twice is not agreement, and two bounds do not confirm each other
    assert settle_guesses([weak, Guess("hell", 38.7, 38.9, "relisten")], 1, [], 1.5)[0] == []
    assert settle_guesses([bound, Guess("hell", 38.0, 39.2, "between/mix", bound=True)], 1, [], 1.5)[0] == []
    # strong on its own; but not when it is an instance already muted, and never more than wanted
    strong = Guess("hell", 38.75, 38.93, "aligned", strong=True)
    assert [g.how for g in settle_guesses([strong], 1, [], 1.5)[0]] == ["aligned"]
    assert settle_guesses([strong], 1, [(38.8, 39.0)], 1.5) == ([], [])
    other = Guess("hell", 40.0, 40.3, "aligned", strong=True)
    accepted, rest = settle_guesses([strong, other], 1, [], 1.5)
    assert accepted[0].start == 38.75 and rest == [other]
