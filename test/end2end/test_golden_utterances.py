"""Golden rows in every locale route to their intent on the m2v pipeline.

For each ``golden_utterances_<lang>.jsonl`` in this directory, one MiniCroft
loads the real skill in that language on the m2v prototype pipeline, the
model2vec engine built at boot from the skill's own ``.intent`` files. Each
row's utterance goes through that pipeline's high, medium and low tiers in
order, and the row passes when the first tier to match names its
``intent_label``. The engine embeds the utterance, so a row that no template
spells out word for word still matches when it means the same thing.

Each locale must pass at least ``MIN_MATCH_RATE`` of its rows, and every
intent with rows in the locale must match at least
``MIN_MATCHED_ROWS_PER_INTENT`` of them. A locale in ``GOLDEN_LOCALE_GAPS``
that falls below the rate is an expected failure with the reason given there;
the per-intent floor still applies to it, except for the (locale, intent)
pairs in ``GOLDEN_INTENT_GAPS``. All rows run, including rows marked
``needs_manual`` or ``machine_generated``, and the test prints every row that
misses with the intent that matched instead.

Each ``negative_utterances_<lang>.jsonl`` holds requests for other skills.
On the same pipeline, no negative row may match an intent of this skill.
The only exceptions are the false claims in ``NEGATIVE_KNOWN_CLAIMS``, each
measured in two runs on the published model, and the locales in
``NEGATIVE_ENGINE_GAPS``, which the model cannot separate at all.
"""
import json
from collections import Counter
from pathlib import Path

import pytest
from ovos_bus_client.message import Message
from ovoscope import M2V_PUBLISHED_MODEL, get_m2v_minicroft
from ovoscope.golden_minicroft import warm_m2v_models

SKILL_ID = "ovos-skill-diagnostics.openvoiceos"
M2V_PROTOTYPE = "ovos-m2v-prototype-pipeline"
TIERS = ("high", "medium", "low")
# m2v gives some rows a different answer on each boot, so the test gates on
# the share of rows that match per locale, not on each row.
MIN_MATCH_RATE = 0.8
MIN_MATCHED_ROWS_PER_INTENT = 1
# A negative row that m2v claims in two runs is listed here by name, as
# {lang: {utterance: claimed_intent}}. Any other claim fails the locale.
NEGATIVE_KNOWN_CLAIMS = {
    "pl-PL": {"włącz jakąś muzykę": "query_extra_langs",
              "jak powiedzieć dzień dobry po angielsku": "query_primary_lang",
              "gdzie jest wieża eiffla": "query_kernel_version"},
    "sv-SE": {"hur blir vädret i morgon": "query_uptime"},
}
# Locales whose negatives the published m2v model cannot separate from this
# skill: the negative check becomes a non-strict xfail with this reason.
NEGATIVE_ENGINE_GAPS = {
    "ru-RU": "measured: the published model claims 5 of 6 ru-RU negatives with the "
             "head templates and with the dev templates, and the claimed intent "
             "changes when a template changes",
}
# Locales whose rate stays below MIN_MATCH_RATE on the published m2v model,
# mapped to the rate measured there (identical in three runs) and the reason.
# The rate check becomes an expected failure. It fails once the rate reaches
# MIN_MATCH_RATE (the stale entry goes) or falls more than GAP_TOLERANCE below
# the measured rate. The rows still run, the misses are still printed, and the
# per-intent floor still fails the test.
_RETRAIN = "published m2v model predates these templates; remove after retrain"
GOLDEN_LOCALE_GAPS = {
    "pl-PL": (0.795, f"{_RETRAIN}; 8 of 39 rows miss: 'na jakiej wersji jądra linuksa "
                     "pracujesz' (none), 'ile mam jeszcze wolnej pamięci masowej' "
                     "(query_memory_usage), 'czy to urządzenie ma gpu' (query_ovos_location), "
                     "'na jakim jądrze działasz' (query_uptime), 'jakimi językami się "
                     "posługujesz' (query_user_lang), 'w jakim języku zwykle mówisz' "
                     "(query_extra_langs), 'od ilu godzin jesteś włączony' (none), "
                     "'kiedy był twój ostatni restart' (none)"),
    "ru-RU": (0.795, f"{_RETRAIN}; 8 of 39 rows miss: 'протестируй процессор' (none), "
                     "'какие ещё языки ты используешь кроме основного' (none), "
                     "'есть ли у тебя графический процессор' (query_cpu_usage), "
                     "'ты используешь видеокарту' (none), 'на каком ядре ты работаешь' "
                     "(none), 'на каком языке ты обычно говоришь' (query_user_lang), "
                     "'когда тебя последний раз перезагружали' (none), "
                     "'какой язык мы сейчас используем' (query_primary_lang)"),
}
GAP_TOLERANCE = 0.10
# (locale, intent) pairs that the per-intent floor skips. Each pair keeps its
# rows in the rate; only the floor check ignores it. A missing .intent file
# still fails test/unittests/test_locale_intent_parity.py.
GOLDEN_INTENT_GAPS = {}
END2END_DIR = Path(__file__).parent


def _rows_by_lang(prefix="golden_utterances_"):
    rows = {}
    for path in sorted(END2END_DIR.glob(f"{prefix}*.jsonl")):
        lang = path.stem.removeprefix(prefix)
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                row = json.loads(line)
                assert row["lang"] == lang, f"{path.name}:{number} has lang {row['lang']!r}"
                rows.setdefault(lang, []).append(row)
    return rows


ROWS = _rows_by_lang()
NEGATIVES = _rows_by_lang("negative_utterances_")


def _matched_intent(engine, utterance, lang):
    message = Message("recognizer_loop:utterance",
                      {"utterances": [utterance], "lang": lang}, {"lang": lang})
    match = next(filter(None, (getattr(engine, f"match_{tier}")([utterance], lang, message)
                               for tier in TIERS)), None)
    return match.match_type if match else None


@pytest.mark.timeout(900)
@pytest.mark.parametrize("lang", sorted(ROWS))
def test_golden_rows_match_their_intent(lang):
    minicroft = get_m2v_minicroft([SKILL_ID], model=M2V_PUBLISHED_MODEL,
                                  lang=lang, classifier=False)
    try:
        warm_m2v_models(minicroft)
        engine = minicroft.intents.pipeline_plugins[M2V_PROTOTYPE]
        misses = []
        matched = Counter()
        for row in ROWS[lang]:
            expected = f"{SKILL_ID}:{row['intent_label']}"
            got = _matched_intent(engine, row["utterance"], lang)
            if got == expected:
                matched[row["intent_label"]] += 1
            else:
                misses.append(f"{row['utterance']!r}: expected {row['intent_label']}, got {got}")
    finally:
        minicroft.stop()
    rate = 1 - len(misses) / len(ROWS[lang])
    print(f"[{lang}] {rate:.1%} of {len(ROWS[lang])} rows match", *misses, sep="\n  ")
    starved = sorted(label for label in {row["intent_label"] for row in ROWS[lang]}
                     if matched[label] < MIN_MATCHED_ROWS_PER_INTENT
                     and (lang, label) not in GOLDEN_INTENT_GAPS)
    assert not starved, (
        f"[{lang}] intents with fewer than {MIN_MATCHED_ROWS_PER_INTENT} matched rows: {starved}"
    )
    if lang in GOLDEN_LOCALE_GAPS:
        measured, reason = GOLDEN_LOCALE_GAPS[lang]
        assert rate >= measured - GAP_TOLERANCE, (
            f"[{lang}] {rate:.1%} of rows match, more than {GAP_TOLERANCE:.0%} below "
            f"the measured {measured:.1%}:\n  " + "\n  ".join(misses)
        )
        assert rate < MIN_MATCH_RATE, (
            f"[{lang}] {rate:.1%} of rows match, at the gate: remove its GOLDEN_LOCALE_GAPS entry"
        )
        pytest.xfail(f"[{lang}] measured {rate:.1%}; known gap: {reason}")
    assert rate >= MIN_MATCH_RATE, (
        f"[{lang}] {rate:.1%} of rows match, below {MIN_MATCH_RATE:.0%}:\n  " + "\n  ".join(misses)
    )


@pytest.mark.timeout(900)
@pytest.mark.parametrize("lang", sorted(NEGATIVES))
def test_negative_rows_match_no_intent_of_this_skill(lang):
    minicroft = get_m2v_minicroft([SKILL_ID], model=M2V_PUBLISHED_MODEL,
                                  lang=lang, classifier=False)
    try:
        warm_m2v_models(minicroft)
        engine = minicroft.intents.pipeline_plugins[M2V_PROTOTYPE]
        known = NEGATIVE_KNOWN_CLAIMS.get(lang, {})
        claimed, known_claims = [], []
        for row in NEGATIVES[lang]:
            got = _matched_intent(engine, row["utterance"], lang)
            if got and got.startswith(f"{SKILL_ID}:"):
                claim = f"{row['utterance']!r}: claimed by {got}"
                if known.get(row["utterance"]) == got.removeprefix(f"{SKILL_ID}:"):
                    known_claims.append(claim)
                else:
                    claimed.append(claim)
    finally:
        minicroft.stop()
    print(f"[{lang}] {len(claimed) + len(known_claims)} of {len(NEGATIVES[lang])} negatives "
          f"claimed, {len(known_claims)} of them known", *claimed, *known_claims, sep="\n  ")
    if claimed and lang in NEGATIVE_ENGINE_GAPS:
        pytest.xfail(f"[{lang}] known engine gap: {NEGATIVE_ENGINE_GAPS[lang]}")
    assert not claimed, f"[{lang}] negatives claimed by this skill:\n  " + "\n  ".join(claimed)


def test_every_shipping_locale_has_a_golden_file():
    golden = {p.stem.split("_", 2)[2] for p in END2END_DIR.glob("golden_utterances_*.jsonl")}
    locale_root = END2END_DIR.parents[1] / "locale"
    shipping = {d.name for d in locale_root.iterdir() if d.is_dir() and any(d.rglob("*.intent"))}
    assert golden == shipping, f"golden files {sorted(golden ^ shipping)} differ from shipping locales"
