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


def _strip_suffix(intent_file: str) -> str:
    """The intent-routing bus event (``<skill_id>:<intent_name>``) is
    dispatched with the ``.intent`` filename suffix already stripped -- eg.
    ``query_cpu_usage`` not ``query_cpu_usage.intent``. ``ALL_INTENTS`` and
    the per-case labels below keep the ``.intent`` suffix as a readable
    filename reference; this normalizes it to the wire form actually
    emitted on the bus."""
    return intent_file[:-len(".intent")] if intent_file.endswith(".intent") else intent_file

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

    # Every ``.intent`` the skill registers; used by the negative case to prove
    # an unrelated utterance routes to none of them.
    ALL_INTENTS = (
        "query_cpu_usage.intent",
        "query_memory_usage.intent",
        "query_primary_lang.intent",
        "query_extra_langs.intent",
        "query_langs.intent",
        "query_user_lang.intent",
        "query_gpu.intent",
        "query_kernel_version.intent",
        "query_ovos_location.intent",
        "query_user_location.intent",
    )

    def _assert_no_match(self, utterance: str):
        matched = []
        handlers = {}
        for intent_file in self.ALL_INTENTS:
            intent_msg_type = f"{SKILL_ID}:{_strip_suffix(intent_file)}"
            handler = lambda msg, f=intent_file: matched.append(f)
            handlers[intent_msg_type] = handler
            self.minicroft.bus.on(intent_msg_type, handler)
        try:
            session = Session(f"e2e-en_us-nomatch-{hash(utterance)}")
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
            for intent_msg_type, handler in handlers.items():
                self.minicroft.bus.remove(intent_msg_type, handler)
        self.assertFalse(
            matched,
            f"{utterance!r} unexpectedly routed to {matched}",
        )

    def _assert_intent(self, utterance: str, intent_file: str):
        intent_msg_type = f"{SKILL_ID}:{_strip_suffix(intent_file)}"
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
            deadline = time.monotonic() + 45
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


class TestQueryLangs(_IntentRoutingMixin, TestCase):
    """query_langs.intent"""

    def test_what_languages_can_you_speak(self):
        self._assert_intent(
            "what languages can you speak", "query_langs.intent")

    def test_available_languages(self):
        self._assert_intent(
            "available languages", "query_langs.intent")


class TestQueryExtraLangs(_IntentRoutingMixin, TestCase):
    """query_extra_langs.intent"""

    def test_what_other_languages_do_you_have(self):
        self._assert_intent(
            "what other languages do you have", "query_extra_langs.intent")

    def test_tell_me_your_extra_languages(self):
        self._assert_intent(
            "tell me your extra languages", "query_extra_langs.intent")


class TestQueryUserLang(_IntentRoutingMixin, TestCase):
    """query_user_lang.intent"""

    def test_what_language_am_i_speaking(self):
        self._assert_intent(
            "what language am i speaking", "query_user_lang.intent")

    def test_what_language_am_i_using(self):
        self._assert_intent(
            "what language am i using", "query_user_lang.intent")


class TestQueryGpu(_IntentRoutingMixin, TestCase):
    """query_gpu.intent"""

    def test_do_you_have_a_gpu(self):
        self._assert_intent(
            "do you have a gpu", "query_gpu.intent")

    def test_does_your_system_have_a_gpu(self):
        self._assert_intent(
            "does your system have a gpu", "query_gpu.intent")


class TestQueryKernelVersion(_IntentRoutingMixin, TestCase):
    """query_kernel_version.intent"""

    def test_what_is_your_kernel_version(self):
        self._assert_intent(
            "what is your kernel version", "query_kernel_version.intent")

    def test_which_kernel_do_you_have(self):
        self._assert_intent(
            "which kernel do you have", "query_kernel_version.intent")


class TestQueryOvosLocation(_IntentRoutingMixin, TestCase):
    """query_ovos_location.intent"""

    def test_where_are_you(self):
        self._assert_intent(
            "where are you", "query_ovos_location.intent")

    def test_what_is_your_current_location(self):
        self._assert_intent(
            "what is your current location", "query_ovos_location.intent")


class TestQueryUserLocation(_IntentRoutingMixin, TestCase):
    """query_user_location.intent"""

    def test_where_am_i(self):
        self._assert_intent(
            "where am i", "query_user_location.intent")

    def test_what_is_my_current_location(self):
        self._assert_intent(
            "what is my current location", "query_user_location.intent")


class TestNoMatch(_IntentRoutingMixin, TestCase):
    """Unrelated utterances must not route to any diagnostics intent."""

    def test_unrelated_utterance_does_not_match(self):
        self._assert_no_match("what is the weather like tomorrow")

    def test_greeting_does_not_match(self):
        self._assert_no_match("hey how are you doing today")
