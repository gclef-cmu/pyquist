import contextlib
import io
import threading
import time
import unittest
from unittest import mock

import numpy as np

from . import stream
from .audio import Audio
from .device import SAFE_PEAK_DBFS
from .helper import db_to_amplitude
from .stream import Stream


class _FakeSdStream:
    """Stand-in for the sounddevice stream classes; records how it was opened."""

    def __init__(self, *, channels, callback, **kwargs):
        self.channels = channels
        self.callback = callback
        self.kwargs = kwargs
        self.started = self.closed = False

    def start(self):
        self.started = True

    def close(self):
        self.closed = True


def _patch_devices(test, *, num_in=2, num_out=2):
    """Patches the current default devices to have the given channel counts."""

    def query_devices(kind):
        return {
            "name": f"Fake {kind}",
            "max_input_channels": num_in,
            "max_output_channels": num_out,
        }

    for name, value in [
        ("query_devices", query_devices),
        ("Stream", _FakeSdStream),
        ("InputStream", _FakeSdStream),
        ("OutputStream", _FakeSdStream),
    ]:
        p = mock.patch.object(stream.sd, name, value)
        p.start()
        test.addCleanup(p.stop)


def _run_block(s: Stream, indata=None, num_out=None):
    """Drives one callback through the stream's sounddevice callback."""
    assert s._stream is not None
    out = None if num_out is None else np.full((s.block_size, num_out), np.nan, "f4")
    if indata is not None and out is not None:
        s._stream.callback(indata, out, s.block_size, None, None)
    elif indata is not None:
        s._stream.callback(indata, s.block_size, None, None)
    else:
        s._stream.callback(out, s.block_size, None, None)
    return out


class TestDeviceChannels(unittest.TestCase):
    def test_matching(self):
        _patch_devices(self, num_in=2, num_out=2)
        self.assertEqual(stream._device_channels(2, "input"), 2)
        self.assertEqual(stream._device_channels(2, "output"), 2)
        self.assertEqual(stream._device_channels(0, "output"), 0)

    def test_mono_output_opens_stereo(self):
        _patch_devices(self, num_out=2)
        self.assertEqual(stream._device_channels(1, "output"), 2)

    def test_mono_input_from_stereo_device(self):
        _patch_devices(self, num_in=2)
        self.assertEqual(stream._device_channels(1, "input"), 1)

    def test_mono_device(self):
        _patch_devices(self, num_in=1, num_out=1)
        self.assertEqual(stream._device_channels(2, "input"), 1)
        self.assertEqual(stream._device_channels(2, "output"), 1)
        self.assertEqual(stream._device_channels(1, "output"), 1)

    def test_fewer_channels_than_device(self):
        _patch_devices(self, num_out=8)
        self.assertEqual(stream._device_channels(4, "output"), 4)

    def test_too_many_channels(self):
        _patch_devices(self, num_out=2)
        with self.assertRaises(ValueError):
            stream._device_channels(4, "output")

    def test_no_channels(self):
        _patch_devices(self, num_in=0)
        with self.assertRaises(ValueError):
            stream._device_channels(1, "input")


class TestStream(unittest.TestCase):
    def test_invalid_channels(self):
        with self.assertRaises(ValueError):
            Stream(lambda block: None, num_output_channels=0)
        with self.assertRaises(ValueError):
            Stream(lambda block: None, num_input_channels=-1)

    def test_context_manager_opens_and_closes(self):
        _patch_devices(self)
        s = Stream(lambda block: None, block_size=64, sample_rate=22050)
        with s:
            sd_stream = s._stream
            self.assertTrue(sd_stream.started)
            self.assertEqual(sd_stream.channels, 2)
            self.assertEqual(sd_stream.kwargs["blocksize"], 64)
            self.assertEqual(sd_stream.kwargs["samplerate"], 22050)
        self.assertTrue(sd_stream.closed)
        self.assertIsNone(s._stream)

    def test_block_attributes(self):
        _patch_devices(self)
        seen = []
        s = Stream(
            seen.append, num_input_channels=1, num_output_channels=2, block_size=32
        )
        with s:
            _run_block(s, indata=np.zeros((32, 1), "f4"), num_out=2)
        (block,) = seen
        self.assertIsInstance(block, Audio)
        self.assertEqual(len(block), 32)
        self.assertEqual(block.num_channels, 2)
        self.assertEqual(block.sample_rate, 44100)

    def test_mono_output_duplicated_and_clipped(self):
        _patch_devices(self, num_out=2)

        def callback(block):
            block.samples[:] = np.linspace(-2, 2, len(block))[:, np.newaxis]

        s = Stream(callback, block_size=16, safe=False)
        with s:
            out = _run_block(s, num_out=2)
        expected = np.clip(np.linspace(-2, 2, 16), -1, 1)
        np.testing.assert_allclose(out[:, 0], expected, rtol=1e-6)
        np.testing.assert_array_equal(out[:, 0], out[:, 1])

    def test_stereo_output_averaged_on_mono_device(self):
        _patch_devices(self, num_out=1)

        def callback(block):
            block.samples[:, 0] = 0.5
            block.samples[:, 1] = -0.1

        s = Stream(callback, num_output_channels=2, block_size=16, safe=False)
        with s:
            out = _run_block(s, num_out=1)
        np.testing.assert_allclose(out, 0.2, rtol=1e-6)

    def test_passthrough(self):
        _patch_devices(self)
        s = Stream(
            lambda block: None,
            num_input_channels=2,
            num_output_channels=2,
            safe=False,
        )
        indata = np.random.uniform(-1, 1, (512, 2)).astype("f4")
        with s:
            out = _run_block(s, indata=indata, num_out=2)
        np.testing.assert_array_equal(out, indata)

    def test_mono_input_device_fills_all_channels(self):
        _patch_devices(self, num_in=1)
        seen = []
        s = Stream(
            lambda block: seen.append(block.copy()),
            num_input_channels=2,
            num_output_channels=0,
            block_size=8,
        )
        with s:
            _run_block(s, indata=np.ones((8, 1), "f4"))
        np.testing.assert_array_equal(seen[0].samples, 1.0)

    def test_extra_channels_start_silent(self):
        _patch_devices(self)
        seen = []
        s = Stream(
            lambda block: seen.append(block.copy()),
            num_input_channels=1,
            num_output_channels=2,
            block_size=8,
        )
        with s:
            _run_block(s, indata=np.ones((8, 1), "f4"), num_out=2)
            _run_block(s, indata=np.ones((8, 1), "f4"), num_out=2)
        for block in seen:
            np.testing.assert_array_equal(block.samples[:, 0], 1.0)
            np.testing.assert_array_equal(block.samples[:, 1], 0.0)

    def test_safe_scales_full_scale_to_safe_peak(self):
        _patch_devices(self, num_out=1)

        def callback(block):
            block.samples[:] = 1.0

        s = Stream(callback, block_size=16)
        with s:
            out = _run_block(s, num_out=1)
        np.testing.assert_allclose(out, db_to_amplitude(SAFE_PEAK_DBFS), rtol=1e-6)

    def test_safe_does_not_change_block(self):
        _patch_devices(self, num_out=1)
        seen = []

        def callback(block):
            block.samples[:] = 0.5
            seen.append(block)

        s = Stream(callback, block_size=16)
        with s:
            _run_block(s, num_out=1)
        np.testing.assert_array_equal(seen[0].samples, 0.5)

    def _fail_from_audio_thread(self, s: Stream, delay=0.05):
        """Runs a block on another thread, like sounddevice does, and returns
        the exception that thread saw."""
        seen = []

        def run():
            time.sleep(delay)
            try:
                _run_block(s, num_out=2)
            except Exception as e:
                seen.append(e)

        thread = threading.Thread(target=run)
        thread.start()
        return thread, seen

    def _failing_stream(self) -> Stream:
        _patch_devices(self)

        def callback(block):
            raise RuntimeError("oops")

        return Stream(callback)

    def test_callback_error_interrupts_with_block(self):
        s = self._failing_stream()
        start = time.monotonic()
        with self.assertRaisesRegex(RuntimeError, "oops"):
            with s:
                thread, seen = self._fail_from_audio_thread(s)
                time.sleep(5.0)
        thread.join()
        self.assertLess(time.monotonic() - start, 2.0)
        self.assertIsInstance(seen[0], stream.sd.CallbackAbort)
        self.assertIsNone(s._stream)

    def test_callback_error_printed_after_start(self):
        s = self._failing_stream()
        s.start()
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            thread, seen = self._fail_from_audio_thread(s, delay=0.0)
            thread.join()
        self.assertIsInstance(seen[0], stream.sd.CallbackAbort)
        self.assertIn("RuntimeError: oops", stderr.getvalue())
        s.stop()  # Already reported; doesn't raise again.

    def test_ctrl_c_still_works(self):
        _patch_devices(self)
        with self.assertRaises(KeyboardInterrupt):
            with Stream(lambda block: None):
                raise KeyboardInterrupt


if __name__ == "__main__":
    unittest.main()
