"""End-to-end intent-routing tests for ovos-skill-diagnostics (en-US).

Most cases here feed an utterance through a MiniCroft stack and assert it
routes to the expected ``.intent`` handler. Routing alone is satisfied by a
handler that speaks nothing or speaks a stale value, so
``query_kernel_version`` and ``query_primary_lang`` additionally capture the
``speak`` message and assert its exact text against the skill's own dialog
template filled with a value computed independently in the test (``platform``
directly for the kernel/OS name, ``pronounce_lang`` -- the same formatting
helper the skill calls, not the skill's own code -- for the language name).
Both handlers only read host state (``platform.system/release`` and the
minicroft's own configured language); neither is asked to write or change
anything on the host.

Run: pytest test/end2end/ -v
"""
import platform
import time
from pathlib import Path
from unittest import TestCase

from ovos_bus_client.message import Message
from ovos_bus_client.session import Session
from ovos_lang_parser import pronounce_lang
from ovoscope import get_minicroft

SKILL_ID = "ovos-skill-diagnostics.openvoiceos"
LANG = "en-US"

SKILL_ROOT = Path(__file__).parent.parent.parent


def _dialog_template(name: str) -> str:
    """Read the en-US dialog file's own text directly, rather than a copy of
    it written into the test -- a rewording of the dialog does not silently
    desync the expected text from what actually ships."""
    path = SKILL_ROOT / "locale" / "en-US" / f"{name}.dialog"
    with open(path, encoding="utf-8") as f:
        return f.readline().strip()


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

    def _assert_speaks(self, utterance: str, expected_text: str):
        """Drive the utterance and assert the exact rendered text of the
        `speak` message the handler emits, catching a handler that speaks
        the wrong dialog or a stale/empty value while still routing fine."""
        spoken = []
        handler = lambda msg: spoken.append(msg.data.get("utterance"))
        self.minicroft.bus.on("speak", handler)
        try:
            session = Session(f"e2e-en_us-speak-{hash(utterance)}")
            session.lang = LANG
            session.pipeline = PIPELINE
            self.minicroft.bus.emit(Message(
                "recognizer_loop:utterance",
                {"utterances": [utterance], "lang": LANG},
                {"session": session.serialize()},
            ))
            deadline = time.monotonic() + 45
            while not spoken and time.monotonic() < deadline:
                time.sleep(0.2)
        finally:
            self.minicroft.bus.remove("speak", handler)
        self.assertEqual(
            spoken, [expected_text],
            f"{utterance!r}: expected spoken text {expected_text!r}, got {spoken!r}",
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

    def test_speaks_primary_language_name(self):
        """The minicroft's core language is en-US with no secondary
        languages configured, so `self.core_lang == self.lang == "en-US"`;
        the expected spoken name is computed with the same `pronounce_lang`
        helper the skill calls (not the skill's own logic) and dropped into
        the dialog file's own template."""
        expected = _dialog_template("primary_lang").format(
            lang=pronounce_lang(LANG, LANG))
        self._assert_speaks("tell me your primary language", expected)


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

    def test_speaks_actual_kernel_and_os(self):
        """The handler reads `platform.system()`/`platform.release()` --
        the exact same call this test makes -- and speaks them through the
        dialog file's own template; a handler returning a hardcoded or
        stale OS string fails this even though it still routes fine. Read
        only: no host state is changed by either the handler or the test."""
        kernel = platform.release().split("-")[0]
        expected = _dialog_template("kernel_version").format(
            os_info=f"{platform.system()} {kernel}")
        self._assert_speaks("what is your kernel version", expected)


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
