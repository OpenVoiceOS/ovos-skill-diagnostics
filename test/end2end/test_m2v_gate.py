"""Handler effects of ovos-skill-diagnostics on the m2v classifier pipeline.

The skill boots through ovoscope's ``get_m2v_minicroft`` with the m2v
classifier checkpoint ``M2V_MULTILINGUAL_MODEL``
(``OpenVoiceOS/ovos-m2v-intents-multi-128M-v5``) and the ``M2V_PIPELINE``
session pipeline. The golden runner in ``test_golden_utterances.py`` checks
routing on the prototype pipeline with ``M2V_PUBLISHED_MODEL``; this module
checks what the skill says after the classifier routes a request.

Each test computes the value it expects without the skill's code: the kernel
string from ``platform``, the CPU percentage from a fixed ``psutil`` reading,
the memory figures from ``psutil.virtual_memory`` within a tolerance. The GPU
tests inject the probe result: a fake ``torch`` whose CUDA is or is not
available, and without torch a patched ``nvidia-smi`` call. The spoken number
is parsed back and compared with the expected value.
"""
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import psutil
from ovos_bus_client.message import Message
from ovos_bus_client.session import Session
from ovos_number_parser import extract_number
from ovoscope import M2V_MULTILINGUAL_MODEL, M2V_PIPELINE, get_m2v_minicroft

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

    _MC = get_m2v_minicroft(skill_ids=[SKILL_ID], model=M2V_MULTILINGUAL_MODEL,
                            lang=LANG)
    _PIPE = _MC.intents.pipeline_plugins["ovos-m2v-pipeline"]
    _PIPE._ensure_model(background_ok=False)


def tearDownModule():
    global _MC, _XDG
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


def _fake_torch(cuda_available):
    return SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: cuda_available))


BINARY_UNITS = ["Bytes", "Kibibytes", "Mebibytes", "Gibibytes", "Tebibytes"]


def _spoken_size(size):
    exponent = 0
    while exponent < len(BINARY_UNITS) - 1 and size >= 1024 ** (exponent + 1):
        exponent += 1
    return f"{size / 1024 ** exponent:.1f} {BINARY_UNITS[exponent]}"


def _parse_spoken_size(text):
    number, unit = re.search(r"([\d.]+) (\w+)", text).groups()
    return float(number) * 1024 ** BINARY_UNITS.index(unit)


class TestM2VDiagnosticsGoldenEffect(unittest.TestCase):
    """A request routes to its intent and the skill speaks the host's real values."""

    def _run(self, utterance, speaks_expected=1, timeout=15.0):
        speaks = []
        failures = []
        _MC.bus.on("speak", speaks.append)
        _MC.bus.on("complete_intent_failure", failures.append)
        sess = Session(session_id=f"m2v-golden-{hash(utterance)}", pipeline=M2V_PIPELINE)
        sess.lang = LANG
        try:
            _MC.bus.emit(Message(
                "recognizer_loop:utterance",
                data={"utterances": [utterance], "lang": LANG},
                context={"session": sess.serialize(), "lang": LANG},
            ))
            deadline = time.time() + timeout
            while time.time() < deadline and len(speaks) < speaks_expected and not failures:
                time.sleep(0.05)
        finally:
            _MC.bus.remove("speak", speaks.append)
            _MC.bus.remove("complete_intent_failure", failures.append)

        self.assertFalse(failures, f"{utterance!r} did not route: complete_intent_failure")
        self.assertEqual(len(speaks), speaks_expected,
                         f"{utterance!r} spoke {len(speaks)} times, expected {speaks_expected}")
        spoken = [((msg.data.get("meta") or {}).get("dialog"), msg.data.get("utterance") or "")
                  for msg in speaks]
        for dialog, text in spoken:
            self.assertNotIn(dialog, text.lower(),
                             f"{utterance!r} spoke the dialog NAME, not rendered text: {text!r}")
        return spoken

    def _spoken_percent(self, text):
        value = extract_number(text, lang=LANG)
        self.assertIsNot(value, False, f"no number in {text!r}")
        self.assertGreaterEqual(value, 0, text)
        self.assertLessEqual(value, 100, text)
        return value

    def test_kernel_version(self):
        expected = f"{platform.system()} {platform.release().split('-')[0]}"
        [(dialog, text)] = self._run("tell me your kernel version")
        self.assertEqual(dialog, "kernel_version")
        self.assertIn(expected.lower(), text.lower())

    def test_cpu_usage(self):
        with mock.patch.object(psutil, "cpu_percent", autospec=True,
                               return_value=37.5) as cpu_percent:
            [(dialog, text)] = self._run("current cpu load")
        cpu_percent.assert_called_once_with()
        self.assertEqual(dialog, "cpu_percent")
        self.assertEqual(self._spoken_percent(text), 37.5, text)

    def test_memory_usage(self):
        before = psutil.virtual_memory()
        spoken = self._run("current memory usage", speaks_expected=2)
        after = psutil.virtual_memory()
        (percent_dialog, percent_text), (available_dialog, available_text) = spoken

        self.assertEqual(percent_dialog, "current_memory")
        percent = self._spoken_percent(percent_text)
        low, high = sorted((before.percent, after.percent))
        self.assertTrue(low - 2 <= percent <= high + 2,
                        f"spoke {percent}, psutil read {before.percent} and {after.percent}")

        self.assertEqual(available_dialog, "available_memory")
        self.assertTrue(available_text.endswith(f"out of {_spoken_size(before.total)}"),
                        f"{available_text!r} does not end with the total {_spoken_size(before.total)}")
        available = _parse_spoken_size(available_text)
        tolerance = 0.05 * before.total
        self.assertTrue(
            min(before.available, after.available) - tolerance
            <= available <= max(before.available, after.available) + tolerance,
            f"spoke {available_text!r}, psutil read {before.available} and {after.available} bytes")

    def test_gpu_query_with_cuda(self):
        with mock.patch.dict(sys.modules, {"torch": _fake_torch(True)}):
            [(dialog, _)] = self._run("are you using a gpu")
        self.assertEqual(dialog, "has_gpu")

    def test_gpu_query_without_cuda(self):
        with mock.patch.dict(sys.modules, {"torch": _fake_torch(False)}):
            [(dialog, _)] = self._run("are you using a gpu")
        self.assertEqual(dialog, "no_gpu")

    def test_gpu_query_nvidia_smi_fallback(self):
        cases = {"has_gpu": {"return_value": b"GPU 0: NVIDIA"},
                 "no_gpu": {"side_effect": FileNotFoundError("nvidia-smi")}}
        for expected, behaviour in cases.items():
            with self.subTest(expected=expected), \
                    mock.patch.dict(sys.modules, {"torch": None}), \
                    mock.patch.object(subprocess, "check_output", **behaviour) as check_output:
                [(dialog, _)] = self._run("are you using a gpu")
                check_output.assert_called_once_with("nvidia-smi")
                self.assertEqual(dialog, expected)

    def test_primary_language(self):
        [(dialog, text)] = self._run("what is your primary language")
        self.assertEqual(dialog, "primary_lang")
        self.assertIn("english", text.lower())


class TestM2VRegisteredLabelRouting(unittest.TestCase):
    """Every intent of the skill registers under its runtime id as a class of the model."""

    def test_every_registered_label_is_a_model_class(self):
        intent_dir = Path(__file__).parents[2] / "locale" / LANG
        expected = {f"{SKILL_ID}:{path.stem}" for path in intent_dir.glob("*.intent")}
        registered = {label for label in _PIPE.intents if label.startswith(f"{SKILL_ID}:")}
        self.assertEqual(registered, expected)
        classes = {str(c) for c in _PIPE.model.classes_}
        self.assertEqual(sorted(registered - classes), [],
                         f"registered labels that {M2V_MULTILINGUAL_MODEL} does not know")


if __name__ == "__main__":
    unittest.main()
