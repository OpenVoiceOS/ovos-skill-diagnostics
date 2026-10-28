"""Unit tests for the byte-formatting helper of ovos-skill-diagnostics."""
from unittest import TestCase

from ovos_skill_diagnostics import nice_bytes


class TestNiceBytes(TestCase):
    def test_spoken_binary_units(self):
        self.assertEqual(nice_bytes(512), "512.0 Bytes")
        self.assertEqual(nice_bytes(1024), "1.0 Kibibyte")
        self.assertEqual(nice_bytes(1024 ** 2), "1.0 Mebibyte")

    def test_short_binary_units(self):
        self.assertEqual(nice_bytes(1024, speech=False), "1.0 KiB")
        self.assertEqual(nice_bytes(1024 ** 3, speech=False), "1.0 GiB")

    def test_decimal_units(self):
        self.assertEqual(nice_bytes(1000, binary=False), "1.0 Kilobyte")
        self.assertEqual(nice_bytes(1000, speech=False, binary=False), "1.0 KB")

    def test_gnu_order_of_magnitude(self):
        self.assertEqual(nice_bytes(1024, gnu=True), "1.0 Kilo")
        self.assertEqual(nice_bytes(1024, speech=False, gnu=True), "1.0 K")

    def test_singular_strips_trailing_s(self):
        self.assertEqual(nice_bytes(1), "1.0 Byte")
