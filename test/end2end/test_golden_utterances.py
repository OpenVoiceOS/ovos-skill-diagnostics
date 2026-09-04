"""Golden-utterance end-to-end coverage for ovos-skill-diagnostics (en-US).

The golden corpus (``golden_utterances.jsonl``) is a vendored slice of the
shared ovoscope golden-utterance dataset, keyed by
``skill_id == "ovos-skill-diagnostics.openvoiceos"``. One shared
``MiniCroft`` (module-scoped fixture) is booted for the whole suite; every
row is its own parametrized test item.

Intent match is asserted directly off the ``ovos.intent.matched`` bus event's
``data.intent_name`` field rather than by scanning for a downstream handler
message type. That field is emitted as ``<skill_id>:<intent_label>`` with the
``.intent`` suffix already stripped (eg. ``query_cpu_usage`` not
``query_cpu_usage.intent``) -- see the live capture in the dev notes for this
suite. Capture ends at ``mycroft.skill.handler.start`` (right after intent
binding fires, before any handler body runs) so a row never depends on a
handler finishing.
"""
import json
from pathlib import Path

import pytest
from ovos_bus_client.message import Message
from ovos_bus_client.session import Session
from ovoscope import CaptureSession, get_minicroft

SKILL_ID = "ovos-skill-diagnostics.openvoiceos"
LANG = "en-US"

_PIPELINE = [
    "ovos-adapt-pipeline-plugin-high",
    "ovos-padatious-pipeline-plugin-high",
    "ovos-padacioso-pipeline-plugin-high",
    "ovos-adapt-pipeline-plugin-medium",
    "ovos-padacioso-pipeline-plugin-medium",
    "ovos-adapt-pipeline-plugin-low",
]

GOLDEN_PATH = Path(__file__).parent / "golden_utterances.jsonl"

# sibling-confusion negatives: each utterance is a real golden phrasing for
# one diagnostics intent that lexically overlaps a sibling diagnostics
# intent's vocabulary ("language", "location"). The sibling must not claim
# it -- these are the pairs the m2v eval flagged as easy to confuse.
SIBLING_NEGATIVES = [
    ("where am i", "query_ovos_location.intent"),
    ("where are you", "query_user_location.intent"),
    ("what language am i speaking", "query_primary_lang.intent"),
    ("what language am i speaking", "query_extra_langs.intent"),
    ("tell me your primary language", "query_user_lang.intent"),
    ("tell me your default language", "query_extra_langs.intent"),
    ("what other languages do you have", "query_primary_lang.intent"),
    ("what languages can you speak", "query_extra_langs.intent"),
    ("tell me your default location", "query_user_location.intent"),
]

# utterances lifted verbatim from OTHER skills' golden-utterance slices,
# picked for lexical overlap with diagnostics' "cpu"/"memory"/"language"/
# "location"/"kernel"/"gpu" vocabulary.
NEGATIVE_UTTERANCES = [
    ("what's the weather like today", "ovos-skill-weather.openvoiceos"),
    ("set the volume to 50 percent", "ovos-skill-volume.openvoiceos"),
    ("what's my ip address", "ovos-skill-ip.openvoiceos"),
    ("tell me a joke", "ovos-skill-icanhazdadjokes.openvoiceos"),
    ("where is the international space station", "ovos-skill-iss-location.openvoiceos"),
    ("set a timer for 5 minutes", "ovos-skill-alerts.openvoiceos"),
    ("play some music", "ovos-skill-music.openvoiceos"),
]


def _expected_names(skill_id: str, intent_label: str) -> set:
    base = intent_label[:-len(".intent")] if intent_label.endswith(".intent") else intent_label
    return {f"{skill_id}:{base}"}


def _load_golden_rows():
    rows = []
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("needs_manual"):
                continue
            rows.append(row)
    return rows


GOLDEN_ROWS = [pytest.param(r, id=r["utterance"]) for r in _load_golden_rows()]


@pytest.fixture(scope="module")
def minicroft():
    mc = get_minicroft([SKILL_ID])
    yield mc
    mc.stop()


def _matched_intent_names(mc, text, session_id):
    session = Session(session_id)
    session.lang = LANG
    session.pipeline = list(_PIPELINE)
    # blacklisted_intents defaults to None on a fresh Session, which crashes
    # the padacioso pipeline (NoneType membership test) - force an empty list.
    session.blacklisted_intents = []
    utterance = Message(
        "recognizer_loop:utterance",
        {"utterances": [text], "lang": LANG},
        {"session": session.serialize(), "source": "A", "destination": "B"},
    )
    capture = CaptureSession(
        mc,
        eof_msgs=["mycroft.skill.handler.start"],
    )
    capture.capture(utterance, timeout=30)
    msgs = capture.finish()
    return [
        m.data.get("intent_name")
        for m in msgs
        if m.msg_type == "ovos.intent.matched"
    ]


def _golden_id(row):
    return row["utterance"]


@pytest.mark.timeout(60)
@pytest.mark.parametrize("row", GOLDEN_ROWS, ids=_golden_id)
def test_golden_utterance(minicroft, row):
    expected = _expected_names(SKILL_ID, row["intent_label"])
    matched = _matched_intent_names(minicroft, row["utterance"], f"golden-{_golden_id(row)}")
    assert any(m in expected for m in matched), (
        f"{row['utterance']!r}: expected one of {sorted(expected)!r}, got {matched!r}"
    )


@pytest.mark.timeout(60)
@pytest.mark.parametrize("negative", NEGATIVE_UTTERANCES, ids=lambda n: n[0])
def test_negative_confusable_not_claimed(minicroft, negative):
    text, source_skill = negative
    matched = _matched_intent_names(minicroft, text, f"negative-{text}")
    claimed = any(m.startswith(f"{SKILL_ID}:") for m in matched if m)
    assert not claimed, f"{text!r} (from {source_skill}) was incorrectly claimed by {SKILL_ID}"


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "sibling", SIBLING_NEGATIVES, ids=lambda s: f"{s[0]}-not-{s[1]}"
)
def test_sibling_confusion_not_claimed(minicroft, sibling):
    text, forbidden_label = sibling
    forbidden = _expected_names(SKILL_ID, forbidden_label)
    matched = _matched_intent_names(minicroft, text, f"sibling-{text}-{forbidden_label}")
    assert not any(m in forbidden for m in matched), (
        f"{text!r}: incorrectly claimed by sibling intent {sorted(forbidden)!r}, got {matched!r}"
    )
