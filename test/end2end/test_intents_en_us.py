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
from pathlib import Path
from unittest import TestCase

from ovos_bus_client.message import Message
from ovos_bus_client.session import Session
from ovos_lang_parser import pronounce_lang
from ovoscope import CaptureSession
from ovoscope.routing import PIPELINE, TALLY, get_prototype_minicroft

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


class _IntentRoutingMixin:
    """Shared MiniCroft setup for intent routing."""

    @classmethod
    def setUpClass(cls):
        cls.minicroft = get_prototype_minicroft([SKILL_ID])

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

    def _run(self, utterance: str, tag: str):
        session = Session(f"e2e-en_us-{tag}-{hash(utterance)}")
        session.lang = LANG
        session.pipeline = PIPELINE
        capture = CaptureSession(self.minicroft)
        capture.capture(Message(
            "recognizer_loop:utterance",
            {"utterances": [utterance], "lang": LANG},
            {"session": session.serialize()},
        ), timeout=45)
        return capture.finish()

    def _assert_no_match(self, utterance: str):
        messages = self._run(utterance, "nomatch")
        TALLY.assert_not_routed(
            utterance, [f"{SKILL_ID}:{intent}" for intent in self.ALL_INTENTS],
            [m.msg_type for m in messages])

    def _assert_intent(self, utterance: str, intent_file: str):
        messages = self._run(utterance, intent_file)
        TALLY.assert_routed(utterance, f"{SKILL_ID}:{intent_file}",
                            [m.msg_type for m in messages])

    def _assert_speaks(self, utterance: str, intent_file: str, expected_text: str):
        """Drive the utterance and assert the exact rendered text of the
        `speak` message the handler emits, catching a handler that speaks
        the wrong dialog or a stale/empty value while still routing fine."""
        messages = self._run(utterance, "speak")
        TALLY.assert_routed(utterance, f"{SKILL_ID}:{intent_file}",
                            [m.msg_type for m in messages])
        spoken = [m.data.get("utterance") for m in messages
                  if m.msg_type in ("speak", "ovos.utterance.speak")]
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
        self._assert_speaks("tell me your primary language",
                            "query_primary_lang.intent", expected)


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
        self._assert_speaks("what is your kernel version",
                            "query_kernel_version.intent", expected)


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
