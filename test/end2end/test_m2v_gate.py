"""m2v-multilingual candidate-default gate for ovos-skill-diagnostics.

Boots the skill through the candidate default intent engine -- the
model2vec multilingual classifier (``OpenVoiceOS/ovos-m2v-intents-multi-128M-v5``)
-- via ovoscope's ``get_m2v_minicroft`` and asserts, for a representative
slice of the skill's own golden utterances, the whole round trip: the
correct intent routed AND the rendered spoken text carries real diagnostic
data, not the raw dialog file name.

Most handlers here report live host state (kernel, CPU load, language
config), so the effect check can't pin an exact string; each case asserts
the routed dialog matches and the rendered text contains real evidence
specific to that dialog (the host kernel name, a "per cent" style CPU
phrase, or the configured language code) rather than merely checking that
something was spoken.
"""
import os
import shutil
import tempfile
import unittest

from ovos_bus_client.message import Message
from ovos_bus_client.session import Session

from ovoscope import get_m2v_minicroft, M2V_PIPELINE

SKILL_ID = "ovos-skill-diagnostics.openvoiceos"
LANG = "en-US"

_MC = None
_PIPE = None
_XDG = None
_ORIG_XDG = None


def setUpModule():
    global _MC, _PIPE, _XDG, _ORIG_XDG
    _ORIG_XDG = os.environ.get("XDG_DATA_HOME")
    _XDG = tempfile.mkdtemp(prefix="ovoscope-m2v-diagnostics-xdg-")
    os.environ["XDG_DATA_HOME"] = _XDG

    _MC = get_m2v_minicroft(skill_ids=[SKILL_ID], lang=LANG)
    _PIPE = _MC.intents.pipeline_plugins["ovos-m2v-pipeline"]
    _PIPE._ensure_model(background_ok=False)


def tearDownModule():
    global _MC, _XDG, _ORIG_XDG
    if _MC is not None:
        _MC.stop()
        _MC = None
    if _ORIG_XDG is None:
        os.environ.pop("XDG_DATA_HOME", None)
    else:
        os.environ["XDG_DATA_HOME"] = _ORIG_XDG
    if _XDG is not None:
        shutil.rmtree(_XDG, ignore_errors=True)
        _XDG = None


class TestM2VDiagnosticsGoldenEffect(unittest.TestCase):
    """Effect assertions: golden utterance in -> correct intent -> real host data out."""

    def _run(self, utterance: str, lang: str = LANG, timeout: float = 15.0):
        speaks = []
        failures = []

        def _on_speak(msg):
            speaks.append(msg)

        def _on_fail(msg):
            failures.append(msg)

        _MC.bus.on("speak", _on_speak)
        _MC.bus.on("complete_intent_failure", _on_fail)
        sess = Session(session_id=f"m2v-golden-{hash(utterance)}", pipeline=M2V_PIPELINE)
        sess.lang = lang
        try:
            _MC.bus.emit(Message(
                "recognizer_loop:utterance",
                data={"utterances": [utterance], "lang": lang},
                context={"session": sess.serialize(), "lang": lang},
            ))
            import time as _t
            deadline = _t.time() + timeout
            while _t.time() < deadline and not speaks and not failures:
                _t.sleep(0.05)
        finally:
            _MC.bus.remove("speak", _on_speak)
            _MC.bus.remove("complete_intent_failure", _on_fail)

        if not speaks:
            return None, None, bool(failures)
        data = speaks[0].data
        meta = data.get("meta", {}) or {}
        return meta.get("dialog"), (data.get("utterance") or ""), False

    def _assert_effect(self, utterance, expected_dialogs, expected_substrings):
        dialog, text, failed = self._run(utterance)
        self.assertFalse(failed, f"{utterance!r} did not route: complete_intent_failure")
        self.assertIsNotNone(text, f"{utterance!r} produced no spoken output")
        low = text.lower()
        self.assertIn(
            dialog, expected_dialogs,
            f"{utterance!r} routed to dialog {dialog!r}, expected one of "
            f"{expected_dialogs!r} (rendered: {text!r})")
        self.assertNotIn(
            dialog, low,
            f"{utterance!r} spoke the dialog NAME, not rendered text: {text!r}")
        for sub in expected_substrings:
            self.assertIn(
                sub.lower(), low,
                f"{utterance!r} rendered {text!r}; missing {sub!r}")

    def test_kernel_version(self):
        import platform
        kernel = platform.system().lower()
        self._assert_effect(
            "tell me your kernel version", ["kernel_version"], [kernel])

    def test_cpu_usage(self):
        self._assert_effect(
            "current cpu load", ["cpu_percent"], ["cent", "cpu"])

    def test_memory_usage(self):
        # handle_get_memory speaks current_memory then available_memory;
        # only the first speak is captured.
        self._assert_effect(
            "current memory usage", ["current_memory"], ["cent"])

    def test_gpu_query(self):
        # the dialog spells the acronym out letter-by-letter ("G P U").
        self._assert_effect(
            "are you using a gpu", ["has_gpu", "no_gpu"], ["g p u"])

    def test_primary_language(self):
        self._assert_effect(
            "what is your primary language", ["primary_lang"], ["english"])


class TestM2VRegisteredLabelRouting(unittest.TestCase):
    """The model's labels carry the skill's real runtime id and route it."""

    def test_registered_skill_id_labels_route(self):
        classes = {str(c) for c in _PIPE.model.classes_}
        registered = set(_PIPE.intents)
        self.assertIn(f"{SKILL_ID}:query_kernel_version", registered)
        self.assertTrue(
            registered & classes,
            "registered intent labels do not intersect the model's classes; "
            f"{SKILL_ID} would route nothing through this model")


if __name__ == "__main__":
    unittest.main()
