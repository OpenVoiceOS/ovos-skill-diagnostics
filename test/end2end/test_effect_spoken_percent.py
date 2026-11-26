"""The CPU percent is spoken through the numtag writer.

psutil reports 99.9; each row asserts the spoken number words of the
language and the word after it, with no trailing zeros after the decimal.
"""
import pytest
from ovos_bus_client.message import Message
from ovos_bus_client.session import Session
from ovoscope import CaptureSession
from ovoscope.routing import PIPELINE, get_prototype_minicroft

import ovos_skill_diagnostics

SKILL_ID = "ovos-skill-diagnostics.openvoiceos"

ROWS = [
    ("en-US", "cpu diagnostics", "ninety-nine point nine per"),
    ("pt-PT", "carga atual do cpu", "noventa e nove vírgula nove por"),
    ("pt-BR", "diagnóstico de cpu", "noventa e nove vírgula nove por"),
    ("es-ES", "cuál es la carga de la cpu", "noventa y nueve coma nueve por"),
    ("fr-FR", "diagnostic processeur", "quatre-vingt-dix-neuf virgule neuf pour"),
    ("it-IT", "che percentuale di c p u stai usando", "novantanove virgola nove percento"),
    ("de-DE", "cpu diagnose", "neunundneunzig komma neun prozent"),
    ("nl-NL", "hoe druk is je processor", "negenennegentig komma negen procent"),
    ("da-DK", "cpu diagnostik", "nioghalvfems komma ni procent"),
]


@pytest.mark.parametrize("lang,utterance,number", ROWS, ids=[r[0] for r in ROWS])
def test_cpu_percent_is_spoken_in_words(monkeypatch, lang, utterance, number):
    monkeypatch.setattr(ovos_skill_diagnostics.psutil, "cpu_percent", lambda: 99.9)
    mc = get_prototype_minicroft([SKILL_ID], lang=lang)
    try:
        session = Session(f"e2e-percent-{lang}")
        session.lang = lang
        session.pipeline = list(PIPELINE)
        session.blacklisted_intents = []
        msg = Message("recognizer_loop:utterance",
                      {"utterances": [utterance], "lang": lang},
                      {"session": session.serialize(), "source": "A", "destination": "B"})
        capture = CaptureSession(mc, eof_msgs=["mycroft.skill.handler.complete"])
        capture.capture(msg, timeout=30)
        spoken = [m.data["utterance"].lower() for m in capture.finish()
                  if m.msg_type in ("speak", "ovos.utterance.speak")]
    finally:
        mc.stop()
    assert len(spoken) == 1 and number in spoken[0], spoken
