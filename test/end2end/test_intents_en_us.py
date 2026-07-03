"""End-to-end intent-routing tests for ovos-skill-diagnostics (en-US).

Each case feeds an utterance through a MiniCroft stack and asserts it routes
to the expected ``.intent`` handler. Coverage spans the CPU-usage query, the
memory-usage query, and the primary-language query.

Run: pytest test/end2end/ -v
"""
import time
from unittest import TestCase

from ovos_bus_client.message import Message
from ovos_bus_client.session import Session
from ovoscope import get_minicroft

SKILL_ID = "ovos-skill-diagnostics.openvoiceos"
LANG = "en-US"

# The intent files use padacioso sample syntax. Exact expansions score in the
# -high band while looser variants land lower, so register all three bands.
PIPELINE = [
    "ovos-padacioso-pipeline-plugin-high",
    "ovos-padacioso-pipeline-plugin-medium",
    "ovos-padacioso-pipeline-plugin-low",
]


class _IntentRoutingMixin:
    """Shared MiniCroft setup for padacioso intent routing."""

    @classmethod
    def setUpClass(cls):
        cls.minicroft = get_minicroft([SKILL_ID])

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "minicroft", None):
            cls.minicroft.stop()

    def _assert_intent(self, utterance: str, intent_file: str):
        intent_msg_type = f"{SKILL_ID}:{intent_file}"
        matched = []
        handler = lambda msg: matched.append(msg)
        self.minicroft.bus.on(intent_msg_type, handler)
        try:
            session = Session(f"e2e-en_us-{intent_file}-{hash(utterance)}")
            session.lang = LANG
            session.pipeline = PIPELINE
            self.minicroft.bus.emit(Message(
                "recognizer_loop:utterance",
                {"utterances": [utterance], "lang": LANG},
                {"session": session.serialize()},
            ))
            deadline = time.monotonic() + 15
            while not matched and time.monotonic() < deadline:
                time.sleep(0.2)
        finally:
            self.minicroft.bus.remove(intent_msg_type, handler)
        self.assertTrue(
            matched,
            f"{utterance!r} did not route to {intent_file}",
        )


class TestQueryCpuUsage(_IntentRoutingMixin, TestCase):
    """query_cpu_usage.intent"""

    def test_whats_your_cpu_usage(self):
        self._assert_intent(
            "what's your current cpu usage", "query_cpu_usage.intent")

    def test_how_much_cpu(self):
        self._assert_intent(
            "how much cpu are you using", "query_cpu_usage.intent")


class TestQueryMemoryUsage(_IntentRoutingMixin, TestCase):
    """query_memory_usage.intent"""

    def test_how_much_memory(self):
        self._assert_intent(
            "how much memory are you using", "query_memory_usage.intent")

    def test_current_memory_usage(self):
        self._assert_intent(
            "current memory usage", "query_memory_usage.intent")


class TestQueryPrimaryLang(_IntentRoutingMixin, TestCase):
    """query_primary_lang.intent"""

    def test_what_language_are_you_using(self):
        self._assert_intent(
            "what language are you using", "query_primary_lang.intent")

    def test_your_primary_language(self):
        self._assert_intent(
            "tell me your primary language", "query_primary_lang.intent")
