"""Unit tests for the disk-space, uptime and core-version handlers, with
psutil (and ovos_core.version) mocked so the assertions are deterministic
regardless of the machine running the suite.
"""
from unittest import TestCase
from unittest.mock import MagicMock, patch

from ovos_skill_diagnostics import SystemDiagnosticsSkill


def _fake_skill():
    skill = MagicMock(spec=SystemDiagnosticsSkill)
    skill.lang = "en-US"
    skill.speak_dialog = MagicMock()
    return skill


class TestHandleGetDiskSpace(TestCase):
    @patch("ovos_skill_diagnostics.psutil.disk_usage")
    def test_speaks_free_and_total(self, disk_usage):
        disk_usage.return_value = MagicMock(free=1024 ** 3, total=10 * 1024 ** 3)
        skill = _fake_skill()

        SystemDiagnosticsSkill.handle_get_disk_space(skill, MagicMock())

        disk_usage.assert_called_once_with('/')
        skill.speak_dialog.assert_called_once_with(
            "disk.space", {"free": "1.0 Gibibyte", "total": "10.0 Gibibytes"})


class TestHandleGetUptime(TestCase):
    @patch("ovos_skill_diagnostics.time.time")
    @patch("ovos_skill_diagnostics.psutil.boot_time")
    def test_speaks_elapsed_duration(self, boot_time, now):
        boot_time.return_value = 1000.0
        now.return_value = 1000.0 + 3725.0
        skill = _fake_skill()

        SystemDiagnosticsSkill.handle_get_uptime(skill, MagicMock())

        boot_time.assert_called_once()
        skill.speak_dialog.assert_called_once()
        args, _ = skill.speak_dialog.call_args
        self.assertEqual(args[0], "uptime")
        self.assertIn("uptime", args[1])
        self.assertTrue(args[1]["uptime"])


class TestHandleGetCoreVersion(TestCase):
    def test_speaks_version_string(self):
        skill = _fake_skill()

        with patch("ovos_core.version.VERSION_MAJOR", 9), \
             patch("ovos_core.version.VERSION_MINOR", 8), \
             patch("ovos_core.version.VERSION_BUILD", 7):
            SystemDiagnosticsSkill.handle_get_core_version(skill, MagicMock())

        skill.speak_dialog.assert_called_once_with(
            "core.version", {"version": "9.8.7"})
