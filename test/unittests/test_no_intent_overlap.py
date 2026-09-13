"""
query_langs.intent (installed/system languages) and query_extra_langs.intent
(secondary/additional languages) are meant to be mutually exclusive intents.
Padacioso resolves ambiguity by confidence, so any literal utterance that
fully expands identically under both templates is a routing ambiguity: the
skill can never reliably tell which the user meant.
"""
import glob
import os
import re
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def expand(line):
    """Expand padacioso ``[optional]``/``(a|b|c)`` sample syntax into the
    full set of literal utterances a template line matches."""
    def rec(s):
        m = re.search(r'\[([^\[\]]*)\]', s)
        if m:
            opts = [s[:m.start()] + s[m.end():], s[:m.start()] + m.group(1) + s[m.end():]]
            out = []
            for o in opts:
                out += rec(o)
            return out
        m = re.search(r'\(([^()]*)\)', s)
        if m:
            out = []
            for alt in m.group(1).split('|'):
                out += rec(s[:m.start()] + alt + s[m.end():])
            return out
        return [re.sub(r'\s+', ' ', s).strip()]
    return rec(line)


def expand_file(path):
    utterances = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                utterances.update(expand(line.rstrip("\n")))
    return utterances


class TestQueryLangsVsExtraLangsDisjoint(unittest.TestCase):
    def test_no_shared_utterances_per_locale(self):
        locale_dirs = sorted(glob.glob(os.path.join(ROOT, "locale", "*")))
        checked = 0
        for locale_dir in locale_dirs:
            langs_file = os.path.join(locale_dir, "query_langs.intent")
            extra_file = os.path.join(locale_dir, "query_extra_langs.intent")
            if not (os.path.isfile(langs_file) and os.path.isfile(extra_file)):
                continue
            checked += 1
            overlap = expand_file(langs_file) & expand_file(extra_file)
            self.assertFalse(
                overlap,
                f"{os.path.basename(locale_dir)}: query_langs.intent and "
                f"query_extra_langs.intent share {len(overlap)} identical "
                f"utterance(s), e.g. {sorted(overlap)[:3]!r}",
            )
        self.assertGreater(checked, 0, "no locales found to check")
