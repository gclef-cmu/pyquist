"""Plot helpers for ``Audio`` (waveform, magnitude spectrum, spectrogram).

These work the same in a notebook and a regular script. Pass
``output_file=<path>`` to save the figure to disk in addition to (or instead
of) displaying it interactively.
"""

import warnings
from typing import Optional, Tuple

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np

from .audio import Audio
from .helper import amplitude_to_db

# Upper bound on the FFT size chosen by ``plot_freq`` when ``n_fft`` is None.
# 2**16 = 65536 samples ≈ 1.5 s at 44.1 kHz. Beyond this the plot gets dense
# and the FFT slow; the user can override by passing ``n_fft`` explicitly.
NFFT_MAX = 1 << 16


def plot(
    audio: Audio,
    *,
    offset: Optional[float] = None,
    duration: Optional[float] = None,
    figsize: Tuple[float, float] = (10, 3),
    ax: Optional[matplotlib.axes.Axes] = None,
    output_file: Optional[str] = None,
) -> matplotlib.axes.Axes:
    """Plots the waveform of an Audio.

    Channels are overlaid on a single axis. The x-axis is time in seconds if
    ``audio.sample_rate`` is set, otherwise sample index. The y-axis is
    symmetric about zero.

    Args:
        audio: The audio to plot.
        offset: Start time in seconds. Defaults to ``0.0`` (beginning of audio).
            Requires ``audio.sample_rate`` to be set.
        duration: Length to plot in seconds. Defaults to the rest of the
            audio. Requires ``audio.sample_rate`` to be set.
        figsize: Figure size passed to ``plt.subplots`` (ignored if ``ax`` is given).
        ax: An existing axis to draw into. If ``None``, a new figure is created.
        output_file: If given, the figure is saved to this path via
            ``Figure.savefig``. Extension determines format (``.png``, ``.pdf``,
            ``.svg``, ...).

    Returns:
        The matplotlib ``Axes`` the waveform was drawn on.
    """
    seg = audio.segment(offset=offset, duration=duration)
    n = seg.num_samples

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if audio.sample_rate is None:
        x = np.arange(n, dtype=float)
        xlabel = "Sample"
    else:
        # Anchor labels at the clamped start time so the x-axis reflects the
        # original audio's timeline rather than the slice's own [0, duration).
        start_time = max(0.0, offset or 0.0)
        x = start_time + np.arange(n) / audio.sample_rate
        xlabel = "Time (s)"

    for ch in range(seg.num_channels):
        label = f"Channel {ch}" if seg.num_channels > 1 else None
        ax.plot(x, seg.samples[:, ch], linewidth=0.5, label=label)

    if seg.num_channels > 1:
        ax.legend(loc="upper right")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Amplitude")
    if n > 0:
        ax.set_xlim(x[0], x[-1])
    # Symmetric y-limits around zero with 5% headroom; fall back to ±1.0
    # when the slice is silent or empty.
    peak = seg.peak_amplitude
    half = peak * 1.05 if peak > 0.0 else 1.0
    ax.set_ylim(-half, half)

    if output_file is not None:
        ax.figure.savefig(output_file)  # type: ignore[union-attr]
    return ax


def plot_freq(
    audio: Audio,
    *,
    offset: Optional[float] = None,
    duration: Optional[float] = None,
    n_fft: Optional[int] = None,
    log_frequency: bool = True,
    log_amplitude: bool = True,
    dynamic_range_db: float = 80.0,
    figsize: Tuple[float, float] = (10, 3),
    ax: Optional[matplotlib.axes.Axes] = None,
    output_file: Optional[str] = None,
) -> matplotlib.axes.Axes:
    """Plots the magnitude spectrum of an Audio via a single FFT.

    Multi-channel audio is first mixed to mono. By default ``n_fft`` is the
    smallest power of two ``>= num_samples`` (zero-padding shorter signals),
    capped at :data:`NFFT_MAX`. When the cap kicks in, a warning is issued
    and the FFT is taken over only the first ``NFFT_MAX`` samples.

    Args:
        audio: The audio to analyze. Must have a ``sample_rate``.
        offset: Start time in seconds. Defaults to ``0.0`` (beginning of audio).
        duration: Length to analyze in seconds. Defaults to the rest of the audio.
        n_fft: FFT size. ``None`` (default) picks the smallest power of two
            ``>= num_samples`` (capped at ``NFFT_MAX``). When set explicitly,
            shorter signals are zero-padded and longer signals are truncated
            to the first ``n_fft`` samples (no warning).
        log_frequency: If True (default), the x-axis uses a log scale.
        log_amplitude: If True (default), magnitudes are converted to dB.
        dynamic_range_db: Magnitudes below ``-dynamic_range_db`` dB are
            floored for numerical stability. Only used when
            ``log_amplitude=True``.
        figsize: Figure size (ignored if ``ax`` is given).
        ax: An existing axis to draw into. If ``None``, a new figure is created.
        output_file: If given, the figure is saved to this path.

    Returns:
        The matplotlib ``Axes`` the spectrum was drawn on.
    """
    if audio.sample_rate is None:
        raise ValueError("Cannot compute FFT without a sample_rate.")

    seg = audio.segment(offset=offset, duration=duration).as_mono()
    if seg.num_samples == 0:
        raise ValueError(
            f"plot_freq: selected segment is empty "
            f"(offset={offset} is past audio duration {audio.duration:.3f}s)."
        )

    samples = seg.samples[:, 0]

    if n_fft is None:
        # Smallest power of two >= num_samples.
        n_fft = 1 << (seg.num_samples - 1).bit_length()
        if n_fft > NFFT_MAX:
            warnings.warn(
                f"plot_freq: audio segment has {seg.num_samples} samples; "
                f"capping FFT size at NFFT_MAX={NFFT_MAX} and analyzing only "
                f"the first {NFFT_MAX} samples. "
                f"Pass an explicit n_fft to override.",
                stacklevel=2,
            )
            n_fft = NFFT_MAX

    # np.fft.rfft zero-pads if n > len(samples) and truncates if n < len(samples).
    spectrum = np.fft.rfft(samples, n=n_fft)
    magnitude = np.abs(spectrum)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / audio.sample_rate)

    if log_amplitude:
        floor = 10 ** (-dynamic_range_db / 20)
        y = amplitude_to_db(np.maximum(magnitude, floor))
        ylabel = "Amplitude (dB)"
    else:
        y = magnitude
        ylabel = "Amplitude"

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    ax.plot(freqs, y, linewidth=0.5)

    if log_frequency:
        ax.set_xscale("log")
        # First bin is DC; skip it so the log axis has a positive floor.
        ax.set_xlim(max(freqs[1], 20.0), audio.sample_rate / 2)
    else:
        ax.set_xlim(0, audio.sample_rate / 2)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(ylabel)

    if output_file is not None:
        ax.figure.savefig(output_file)  # type: ignore[union-attr]
    return ax


def plot_spec(
    audio: Audio,
    *,
    offset: Optional[float] = None,
    duration: Optional[float] = None,
    n_fft: int = 2048,
    hop_length: int = 512,
    log_frequency: bool = True,
    log_amplitude: bool = True,
    dynamic_range_db: float = 80.0,
    figsize: Tuple[float, float] = (10, 4),
    ax: Optional[matplotlib.axes.Axes] = None,
    output_file: Optional[str] = None,
) -> matplotlib.axes.Axes:
    """Plots a magnitude spectrogram of an Audio.

    Multi-channel audio is first mixed to mono. A Hann-windowed STFT with
    ``n_fft`` window size and ``hop_length`` frame advance is computed;
    magnitudes are optionally converted to dB and plotted on a log-frequency
    axis (both defaults).

    Args:
        audio: The audio to analyze. Must have a ``sample_rate``.
        offset: Start time in seconds. Defaults to ``0.0`` (beginning of audio).
        duration: Length to analyze in seconds. Defaults to the rest of the audio.
        n_fft: STFT window size in samples. Defaults to 2048 (~46 ms at 44.1 kHz).
        hop_length: Frame advance in samples. Defaults to 512 (75% overlap).
        log_frequency: If True (default), the y-axis uses a log scale.
        log_amplitude: If True (default), magnitudes are converted to dB
            (via :func:`pyquist.helper.amplitude_to_db`).
        dynamic_range_db: Magnitudes below ``-dynamic_range_db`` dB are floored
            for numerical stability. Only used when ``log_amplitude=True``.
        figsize: Figure size (ignored if ``ax`` is given).
        ax: An existing axis to draw into. If ``None``, a new figure is created.
        output_file: If given, the figure is saved to this path.

    Returns:
        The matplotlib ``Axes`` the spectrogram was drawn on.
    """
    if audio.sample_rate is None:
        raise ValueError("Cannot compute spectrogram without a sample_rate.")

    seg = audio.segment(offset=offset, duration=duration).as_mono()
    if seg.num_samples == 0:
        raise ValueError(
            f"plot_spec: selected segment is empty "
            f"(offset={offset} is past audio duration {audio.duration:.3f}s)."
        )
    if seg.num_samples < n_fft:
        raise ValueError(
            f"plot_spec: selected segment has {seg.num_samples} samples, "
            f"which is shorter than n_fft={n_fft}. "
            f"Use a smaller n_fft or a longer duration."
        )
    samples = seg.samples[:, 0]
    win = np.hanning(n_fft).astype(samples.dtype)
    frames = [
        np.fft.rfft(samples[i : i + n_fft] * win)
        for i in range(0, len(samples) - n_fft + 1, hop_length)
    ]
    zxx = np.stack(frames, axis=1)
    t = np.arange(len(frames)) * hop_length / audio.sample_rate + max(
        0.0, offset or 0.0
    )
    f = np.fft.rfftfreq(n_fft, d=1.0 / audio.sample_rate)

    magnitude = np.abs(zxx)
    if log_amplitude:
        floor = 10 ** (-dynamic_range_db / 20)
        spec = amplitude_to_db(np.maximum(magnitude, floor))
        cbar_label = "Amplitude (dB)"
    else:
        spec = magnitude
        cbar_label = "Amplitude"

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    mesh = ax.pcolormesh(t, f, spec, shading="auto")

    if log_frequency:
        ax.set_yscale("log")
        # First bin is DC; skip it so log scale has a positive floor.
        ax.set_ylim(max(f[1], 20.0), audio.sample_rate / 2)
    else:
        ax.set_ylim(0, audio.sample_rate / 2)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")

    cbar = plt.colorbar(mesh, ax=ax)
    cbar.set_label(cbar_label)

    if output_file is not None:
        ax.figure.savefig(output_file)  # type: ignore[union-attr]
    return ax
