import os
import tempfile
import unittest
import warnings

import matplotlib

matplotlib.use("Agg", force=True)  # headless backend for tests
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .audio import Audio  # noqa: E402
from .plot import NFFT_MAX, plot, plot_freq, plot_spec  # noqa: E402


def _sine(freq=440.0, duration=1.0, sample_rate=44100, amplitude=0.3):
    n = int(duration * sample_rate)
    t = np.arange(n) / sample_rate
    return Audio(
        amplitude * np.sin(2 * np.pi * freq * t).astype(np.float32),
        sample_rate=sample_rate,
    )


class TestPlotWaveform(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_labels_and_xscale(self):
        ax = plot(_sine())
        self.assertEqual(ax.get_xlabel(), "Time (s)")
        self.assertEqual(ax.get_ylabel(), "Amplitude")
        # Waveform has no title.
        self.assertEqual(ax.get_title(), "")

    def test_symmetric_ylim(self):
        ax = plot(_sine(amplitude=0.3))
        ylo, yhi = ax.get_ylim()
        self.assertAlmostEqual(ylo, -yhi)
        self.assertGreater(yhi, 0.3)  # ≥ peak

    def test_silent_audio_fallback_ylim(self):
        silent = Audio(np.zeros(1000, dtype=np.float32), sample_rate=44100)
        ax = plot(silent)
        ylo, yhi = ax.get_ylim()
        self.assertEqual((ylo, yhi), (-1.0, 1.0))

    def test_no_sample_rate_uses_sample_axis(self):
        buf = Audio(np.zeros(500, dtype=np.float32))
        ax = plot(buf)
        self.assertEqual(ax.get_xlabel(), "Sample")

    def test_offset_duration_anchors_xlim_to_original_timeline(self):
        ax = plot(_sine(duration=2.0), offset=0.5, duration=0.5)
        xlo, xhi = ax.get_xlim()
        self.assertAlmostEqual(xlo, 0.5, places=3)
        # End is slightly before 1.0 because the slice is exclusive of end.
        self.assertGreater(xhi, 0.95)
        self.assertLess(xhi, 1.005)

    def test_multichannel_adds_legend(self):
        sr = 44100
        stereo_samples = np.stack(
            [np.linspace(-0.5, 0.5, sr), np.linspace(0.5, -0.5, sr)], axis=1
        ).astype(np.float32)
        ax = plot(Audio(stereo_samples, sample_rate=sr))
        self.assertIsNotNone(ax.get_legend())

    def test_offset_without_sample_rate_raises(self):
        buf = Audio(np.zeros(100, dtype=np.float32))
        with self.assertRaises(ValueError):
            plot(buf, offset=0.1)

    def test_output_file_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "waveform.png")
            plot(_sine(), output_file=path)
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 0)

    def test_uses_provided_axis(self):
        fig, ax = plt.subplots()
        returned = plot(_sine(), ax=ax)
        self.assertIs(returned, ax)


class TestPlotFreq(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_default_axes_and_labels(self):
        ax = plot_freq(_sine())
        self.assertEqual(ax.get_xlabel(), "Frequency (Hz)")
        self.assertEqual(ax.get_ylabel(), "Amplitude (dB)")
        self.assertEqual(ax.get_xscale(), "log")

    def test_linear_axes_opt_out(self):
        ax = plot_freq(_sine(), log_frequency=False, log_amplitude=False)
        self.assertEqual(ax.get_xscale(), "linear")
        self.assertEqual(ax.get_ylabel(), "Amplitude")

    def test_default_n_fft_is_next_power_of_two(self):
        # 1000 samples → next pow2 is 1024. Bin spacing = sr / 1024.
        sr = 44100
        ax = plot_freq(_sine(duration=1000 / sr, sample_rate=sr), log_frequency=False)
        # rfft of 1024 → 513 freq bins, last bin at sr/2.
        line = ax.get_lines()[0]
        x = line.get_xdata()
        self.assertEqual(len(x), 1024 // 2 + 1)

    def test_cap_at_nfft_max_emits_warning(self):
        sr = 44100
        long = Audio(np.zeros(NFFT_MAX * 2, dtype=np.float32), sample_rate=sr)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            plot_freq(long)
        self.assertEqual(len(caught), 1)
        self.assertIn("NFFT_MAX", str(caught[0].message))

    def test_explicit_n_fft_does_not_warn(self):
        sr = 44100
        long = Audio(np.zeros(NFFT_MAX * 2, dtype=np.float32), sample_rate=sr)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            plot_freq(long, n_fft=2048)
        self.assertEqual(len(caught), 0)

    def test_empty_segment_raises(self):
        with self.assertRaises(ValueError):
            plot_freq(_sine(duration=1.0), offset=10.0)

    def test_no_sample_rate_raises(self):
        buf = Audio(np.zeros(100, dtype=np.float32))
        with self.assertRaises(ValueError):
            plot_freq(buf)

    def test_output_file_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "spec.png")
            plot_freq(_sine(), output_file=path)
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 0)


class TestPlotSpec(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_default_axes_and_labels(self):
        ax = plot_spec(_sine())
        self.assertEqual(ax.get_xlabel(), "Time (s)")
        self.assertEqual(ax.get_ylabel(), "Frequency (Hz)")
        self.assertEqual(ax.get_yscale(), "log")

    def test_linear_axes_opt_out(self):
        ax = plot_spec(_sine(), log_frequency=False, log_amplitude=False)
        self.assertEqual(ax.get_yscale(), "linear")

    def test_offset_past_end_clear_error(self):
        with self.assertRaises(ValueError) as ctx:
            plot_spec(_sine(duration=1.0), offset=10.0)
        self.assertIn("empty", str(ctx.exception))

    def test_segment_shorter_than_n_fft_clear_error(self):
        with self.assertRaises(ValueError) as ctx:
            plot_spec(_sine(duration=1.0), duration=0.01)  # 441 samples < 2048
        self.assertIn("n_fft", str(ctx.exception))

    def test_no_sample_rate_raises(self):
        buf = Audio(np.zeros(10000, dtype=np.float32))
        with self.assertRaises(ValueError):
            plot_spec(buf)

    def test_output_file_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "spec.png")
            plot_spec(_sine(), output_file=path)
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 0)

    def test_offset_anchors_x_axis(self):
        # The displayed time axis should start near the requested offset.
        ax = plot_spec(_sine(duration=2.0), offset=0.5, duration=0.8)
        xlo, _ = ax.get_xlim()
        self.assertGreater(xlo, 0.4)
        self.assertLess(xlo, 0.6)


if __name__ == "__main__":
    unittest.main()
