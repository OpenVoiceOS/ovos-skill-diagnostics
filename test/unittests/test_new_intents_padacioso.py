"""Padacioso probes for the new en-US intent files (disk space, uptime, core
version), plus a cross-file negative check to guard against the disk-vs-memory
ambiguity a shared "free" vocabulary word could introduce.
"""
import os
from unittest import TestCase

from padacioso import IntentContainer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOCALE = os.path.join(ROOT, "locale", "en-US")


def _load(name):
    path = os.path.join(LOCALE, f"{name}.intent")
    with open(path, encoding="utf-8") as f:
        return [l.rstrip("\n") for l in f if l.strip()]


def _container():
    c = IntentContainer()
    for name in ("query_disk_space", "query_uptime", "query_core_version",
                 "query_memory_usage"):
        c.add_intent(name, _load(name))
    return c


class TestQueryDiskSpaceIntent(TestCase):
    def setUp(self):
        self.container = _container()

    def test_matches_expected_utterances(self):
        for utt in (
            "how much disk space is left",
            "how much storage do I have",
            "how much free disk space is there",
            "what's my free disk space",
            "am I running out of disk space",
            "am I running out of storage",
        ):
            result = self.container.calc_intent(utt)
            self.assertEqual(result["name"], "query_disk_space", utt)

    def test_does_not_match_memory_query(self):
        result = self.container.calc_intent("how much free memory are you using")
        self.assertNotEqual(result["name"], "query_disk_space")

    def test_memory_query_does_not_match_disk(self):
        result = self.container.calc_intent("what is your free ram")
        self.assertEqual(result["name"], "query_memory_usage")


class TestQueryUptimeIntent(TestCase):
    def setUp(self):
        self.container = _container()

    def test_matches_expected_utterances(self):
        for utt in (
            "what's your uptime",
            "what is your uptime",
            "how long have you been running",
            "how long are you up",
            "when did you boot",
            "when did you last start",
        ):
            result = self.container.calc_intent(utt)
            self.assertEqual(result["name"], "query_uptime", utt)


class TestQueryCoreVersionIntent(TestCase):
    def setUp(self):
        self.container = _container()

    def test_matches_expected_utterances(self):
        for utt in (
            "what ovos core version are you running",
            "which version is this",
            "what version of ovos is this",
            "what version of ovos are you running",
        ):
            result = self.container.calc_intent(utt)
            self.assertEqual(result["name"], "query_core_version", utt)
