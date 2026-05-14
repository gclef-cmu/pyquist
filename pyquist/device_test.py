import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from . import device


class _FakeSdDefault:
    """Stand-in for ``sounddevice.default``."""

    def __init__(self):
        self.device = (0, 0)


_FAKE_DEVICES = [
    {"name": "MacBook Microphone", "max_input_channels": 1, "max_output_channels": 0},
    {"name": "MacBook Speakers", "max_input_channels": 0, "max_output_channels": 2},
    {
        "name": "External Audio Interface",
        "max_input_channels": 2,
        "max_output_channels": 2,
    },
]


def _patch_sd(test, *, devices=None, default=None):
    """Helper: patches ``device.sd.query_devices`` and ``device.sd.default``."""
    devices = _FAKE_DEVICES if devices is None else devices
    default = _FakeSdDefault() if default is None else default
    patches = [
        mock.patch.object(device.sd, "query_devices", return_value=devices),
        mock.patch.object(device.sd, "default", default),
    ]
    for p in patches:
        p.start()
        test.addCleanup(p.stop)
    return default


class TestResolveDevice(unittest.TestCase):
    def setUp(self):
        _patch_sd(self)

    def test_int_id(self):
        self.assertEqual(device._resolve_device(0, "input"), (0, "MacBook Microphone"))
        self.assertEqual(device._resolve_device(1, "output"), (1, "MacBook Speakers"))

    def test_int_id_out_of_range(self):
        with self.assertRaises(ValueError):
            device._resolve_device(99, "input")
        with self.assertRaises(ValueError):
            device._resolve_device(-1, "input")

    def test_int_id_wrong_kind(self):
        # Device 0 is input-only.
        with self.assertRaises(ValueError):
            device._resolve_device(0, "output")
        # Device 1 is output-only.
        with self.assertRaises(ValueError):
            device._resolve_device(1, "input")

    def test_name_substring_match(self):
        self.assertEqual(
            device._resolve_device("Microphone", "input"), (0, "MacBook Microphone")
        )
        self.assertEqual(
            device._resolve_device("External", "output"),
            (2, "External Audio Interface"),
        )

    def test_name_no_match(self):
        with self.assertRaises(ValueError):
            device._resolve_device("Nonexistent", "input")

    def test_name_kind_filtered_out(self):
        # "Speakers" exists but only as output.
        with self.assertRaises(ValueError):
            device._resolve_device("Speakers", "input")

    def test_name_multiple_matches(self):
        ambiguous = _FAKE_DEVICES + [
            {
                "name": "USB Microphone Array",
                "max_input_channels": 4,
                "max_output_channels": 0,
            }
        ]
        with mock.patch.object(device.sd, "query_devices", return_value=ambiguous):
            with self.assertRaises(ValueError):
                device._resolve_device("Microphone", "input")

    def test_invalid_argument_type(self):
        with self.assertRaises(TypeError):
            device._resolve_device(1.5, "input")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            device._resolve_device(True, "input")  # bool subclass of int — reject


class TestPromptDevice(unittest.TestCase):
    def setUp(self):
        _patch_sd(self)
        # Silence the device-list print that _prompt_device emits.
        p = mock.patch("sys.stdout")
        p.start()
        self.addCleanup(p.stop)

    def test_prompt_with_int_input(self):
        with mock.patch("builtins.input", return_value="0"):
            self.assertEqual(device._prompt_device("input"), (0, "MacBook Microphone"))

    def test_prompt_with_name_input(self):
        with mock.patch("builtins.input", return_value="External"):
            self.assertEqual(
                device._prompt_device("output"), (2, "External Audio Interface")
            )

    def test_prompt_phrasing_session_only(self):
        with mock.patch("builtins.input", return_value="0") as mock_input:
            device._prompt_device("input", update_default=False)
        self.assertIn("Select input device", mock_input.call_args.args[0])
        self.assertNotIn("default", mock_input.call_args.args[0])

    def test_prompt_phrasing_persistent(self):
        with mock.patch("builtins.input", return_value="0") as mock_input:
            device._prompt_device("input", update_default=True)
        self.assertIn("Select default input device", mock_input.call_args.args[0])


class TestPromptDeviceAnnotations(unittest.TestCase):
    """Verify [current] and [default] markers in the prompt listing."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name) / "device_defaults.json"
        p = mock.patch.object(device, "_DEFAULTS_PATH", self.tmp_path)
        p.start()
        self.addCleanup(p.stop)
        self.addCleanup(self.tmp.cleanup)

    def _run_prompt(self, kind, fake_default, *, default_name=None):
        if default_name is not None:
            self.tmp_path.write_text(json.dumps({kind: default_name}))
        with (
            mock.patch.object(device.sd, "default", fake_default),
            mock.patch.object(device.sd, "query_devices", return_value=_FAKE_DEVICES),
            mock.patch("builtins.input", return_value="0"),
        ):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                device._prompt_device(kind)
            return buf.getvalue()

    def test_no_tags_when_neither_current_nor_default(self):
        # Set sd.default.device[0] to an out-of-range index → no "current" tag.
        fake = _FakeSdDefault()
        fake.device = (-1, -1)
        out = self._run_prompt("input", fake)
        self.assertIn("0: MacBook Microphone", out)
        self.assertNotIn("[current]", out)
        self.assertNotIn("[default]", out)

    def test_current_only(self):
        fake = _FakeSdDefault()
        fake.device = (2, 0)  # External Audio Interface as input
        out = self._run_prompt("input", fake)
        self.assertIn("2: External Audio Interface [current]", out)
        self.assertNotIn("[default]", out)

    def test_default_only(self):
        # Default is set but current is something else.
        fake = _FakeSdDefault()
        fake.device = (0, 0)
        out = self._run_prompt("input", fake, default_name="External Audio Interface")
        self.assertIn("0: MacBook Microphone [current]", out)
        self.assertIn("2: External Audio Interface [default]", out)

    def test_current_and_default_same_device(self):
        fake = _FakeSdDefault()
        fake.device = (0, 0)
        out = self._run_prompt("input", fake, default_name="MacBook Microphone")
        self.assertIn("0: MacBook Microphone [current, default]", out)

    def test_unresolvable_default_silently_skipped(self):
        # Persisted default refers to a device that no longer exists.
        fake = _FakeSdDefault()
        out = self._run_prompt("input", fake, default_name="Phantom Device")
        self.assertIn("0: MacBook Microphone", out)
        self.assertNotIn("[default]", out)


class TestNoDevicesAvailable(unittest.TestCase):
    """Behavior when the machine has zero (or zero of one kind of) devices."""

    def test_prompt_device_raises_when_no_devices(self):
        with (
            mock.patch.object(device.sd, "query_devices", return_value=[]),
            mock.patch.object(device.sd, "default", _FakeSdDefault()),
            mock.patch("sys.stdout"),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                device._prompt_device("input")
            self.assertIn("No input devices", str(ctx.exception))

    def test_prompt_device_raises_when_no_devices_of_kind(self):
        # Output-only world → asking for input prompt should fail clearly.
        output_only = [
            {"name": "Speakers", "max_input_channels": 0, "max_output_channels": 2}
        ]
        with (
            mock.patch.object(device.sd, "query_devices", return_value=output_only),
            mock.patch.object(device.sd, "default", _FakeSdDefault()),
            mock.patch("sys.stdout"),
        ):
            with self.assertRaises(RuntimeError):
                device._prompt_device("input")


class TestApplyPersistedDefaultsRobust(unittest.TestCase):
    """``_apply_persisted_defaults`` must never let module import fail."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name) / "device_defaults.json"
        p = mock.patch.object(device, "_DEFAULTS_PATH", self.tmp_path)
        p.start()
        self.addCleanup(p.stop)
        self.addCleanup(self.tmp.cleanup)

    def test_no_audio_backend_does_not_raise(self):
        # Cached default exists, but sd.query_devices itself errors out
        # (e.g. headless server with no audio backend).
        self.tmp_path.write_text(json.dumps({"input": "MacBook"}))
        with (
            mock.patch.object(
                device.sd, "query_devices", side_effect=OSError("no audio backend")
            ),
            mock.patch("sys.stderr"),
        ):
            device._apply_persisted_defaults()  # must not raise


class TestPersistence(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name) / "device_defaults.json"
        p = mock.patch.object(device, "_DEFAULTS_PATH", self.tmp_path)
        p.start()
        self.addCleanup(p.stop)
        self.addCleanup(self.tmp.cleanup)

    def test_load_missing_file(self):
        self.assertEqual(device._load_defaults(), {})

    def test_load_corrupt_file(self):
        self.tmp_path.write_text("{not json")
        with mock.patch("sys.stderr"):
            self.assertEqual(device._load_defaults(), {})

    def test_save_load_roundtrip(self):
        d = {"input": "Mic", "output": "Speakers"}
        device._save_defaults(d)
        self.assertEqual(device._load_defaults(), d)


class TestSetInputOutputDevice(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name) / "device_defaults.json"
        p = mock.patch.object(device, "_DEFAULTS_PATH", self.tmp_path)
        p.start()
        self.addCleanup(p.stop)
        self.addCleanup(self.tmp.cleanup)
        self.fake = _patch_sd(self)

    def test_set_input_by_id_session_only(self):
        # Default is session-only — must not persist.
        device.set_input_device(2)  # External Audio Interface
        self.assertEqual(self.fake.device, (2, 0))
        self.assertFalse(self.tmp_path.exists())

    def test_set_input_by_name_session_only(self):
        device.set_input_device("Microphone")
        self.assertEqual(self.fake.device, (0, 0))
        self.assertFalse(self.tmp_path.exists())

    def test_set_output_session_only(self):
        device.set_output_device(1)
        self.assertEqual(self.fake.device, (0, 1))
        self.assertFalse(self.tmp_path.exists())

    def test_set_input_persists_with_update_default(self):
        device.set_input_device(2, update_default=True)
        self.assertEqual(self.fake.device, (2, 0))
        with open(self.tmp_path) as f:
            self.assertEqual(json.load(f), {"input": "External Audio Interface"})

    def test_set_output_persists_with_update_default(self):
        device.set_output_device(1, update_default=True)
        self.assertEqual(self.fake.device, (0, 1))
        with open(self.tmp_path) as f:
            self.assertEqual(json.load(f), {"output": "MacBook Speakers"})

    def test_set_input_then_output_merges_cache(self):
        device.set_input_device("Microphone", update_default=True)
        device.set_output_device("Speakers", update_default=True)
        self.assertEqual(self.fake.device, (0, 1))
        with open(self.tmp_path) as f:
            self.assertEqual(
                json.load(f),
                {"input": "MacBook Microphone", "output": "MacBook Speakers"},
            )

    def test_invalid_choice_does_not_persist(self):
        with self.assertRaises(ValueError):
            device.set_input_device("Nonexistent", update_default=True)
        self.assertFalse(self.tmp_path.exists())
        self.assertEqual(self.fake.device, (0, 0))  # untouched

    def test_interactive_prompt_session_only_by_default(self):
        with (
            mock.patch("builtins.input", return_value="External"),
            mock.patch("sys.stdout"),
        ):
            device.set_input_device()
        self.assertEqual(self.fake.device, (2, 0))
        self.assertFalse(self.tmp_path.exists())


class TestApplyPersistedDefaults(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name) / "device_defaults.json"
        p = mock.patch.object(device, "_DEFAULTS_PATH", self.tmp_path)
        p.start()
        self.addCleanup(p.stop)
        self.addCleanup(self.tmp.cleanup)
        self.fake = _patch_sd(self)

    def test_no_cache_no_op(self):
        device._apply_persisted_defaults()
        self.assertEqual(self.fake.device, (0, 0))  # initial unchanged

    def test_applies_cached_input(self):
        self.tmp_path.write_text(json.dumps({"input": "External"}))
        device._apply_persisted_defaults()
        self.assertEqual(self.fake.device, (2, 0))

    def test_applies_cached_input_and_output(self):
        self.tmp_path.write_text(
            json.dumps({"input": "Microphone", "output": "Speakers"})
        )
        device._apply_persisted_defaults()
        self.assertEqual(self.fake.device, (0, 1))

    def test_skips_missing_device_with_warning(self):
        self.tmp_path.write_text(
            json.dumps({"input": "Nonexistent", "output": "Speakers"})
        )
        with mock.patch("sys.stderr"):
            device._apply_persisted_defaults()
        # Output still applied even though input failed
        self.assertEqual(self.fake.device, (0, 1))

    def test_stale_default_preserves_cache_across_sessions(self):
        # Simulates: user persisted "External", later unplugged it.
        self.tmp_path.write_text(
            json.dumps({"input": "External", "output": "External"})
        )
        original = self.tmp_path.read_text()

        # Only built-in devices remain.
        remaining = [
            {
                "name": "MacBook Microphone",
                "max_input_channels": 1,
                "max_output_channels": 0,
            },
            {
                "name": "MacBook Speakers",
                "max_input_channels": 0,
                "max_output_channels": 2,
            },
        ]
        with (
            mock.patch.object(device.sd, "query_devices", return_value=remaining),
            mock.patch("sys.stderr"),
        ):
            device._apply_persisted_defaults()
            # sd.default.device untouched (sounddevice's own default stands).
            self.assertEqual(self.fake.device, (0, 0))
            # Cache file is preserved verbatim — re-plugging the device next
            # session should restore it.
            self.assertEqual(self.tmp_path.read_text(), original)

    def test_stale_default_warning_includes_remediation(self):
        self.tmp_path.write_text(json.dumps({"input": "Phantom"}))
        captured = io.StringIO()
        with contextlib.redirect_stderr(captured):
            device._apply_persisted_defaults()
        msg = captured.getvalue()
        self.assertIn("Phantom", msg)
        self.assertIn("pyquist devices", msg)


if __name__ == "__main__":
    unittest.main()
