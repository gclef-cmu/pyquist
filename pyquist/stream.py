"""Real-time audio streaming via :mod:`sounddevice`.

A :class:`Stream` calls your function once per block of audio, while audio
is flowing to the speakers and/or from the microphone. The function gets a
single :class:`~pyquist.Audio` ``block`` that it modifies in place: on the
way in it holds the latest input (if any), and whatever it holds on the way
out is played::

    import time

    import numpy as np
    import pyquist as pq

    def noise(block: pq.Audio) -> None:
        block.samples[:] = 0.1 * np.random.uniform(-1, 1, block.shape)

    with pq.Stream(noise):
        time.sleep(5.0)

Streams use the devices selected with :func:`~pyquist.set_input_device` /
:func:`~pyquist.set_output_device` (see :func:`~pyquist.list_devices`).
"""

import signal
import threading
import traceback
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

from .audio import Audio
from .device import SAFE_PEAK_DBFS
from .helper import db_to_amplitude


class Stream:
    """A real-time audio stream that calls ``callback`` on each block.

    Use it as a context manager (``with pq.Stream(callback): ...``), or call
    :meth:`start` and :meth:`stop` yourself to keep it running across cells.

    Each ``block`` holds ``block_size`` samples with
    ``max(num_input_channels, num_output_channels)`` channels. It arrives
    holding the input (zeros elsewhere), and its first ``num_output_channels``
    channels are played once ``callback`` returns.
    """

    def __init__(
        self,
        callback: Callable[[Audio], None],
        *,
        num_input_channels: int = 0,
        num_output_channels: int = 1,
        block_size: int = 512,
        sample_rate: int = 44100,
        safe: bool = True,
    ):
        """Configures the stream; nothing plays until :meth:`start`.

        Args:
            callback: Called as ``callback(block)`` once per block, from the
                audio thread. Must modify ``block`` in place, and return fast
                enough to keep up (``block_size / sample_rate`` seconds).
            num_input_channels: Channels to read from the input device; 0
                (default) for no input.
            num_output_channels: Channels to write to the output device; 0
                for no output. Defaults to 1 (mono).
            block_size: Samples per block. Smaller means lower latency but
                more risk of glitches.
            sample_rate: Sample rate in Hz.
            safe: If True (default), multiplies every output block by a fixed
                gain that maps full scale to ``SAFE_PEAK_DBFS`` (-20 dBFS), to
                protect ears. Unlike :func:`~pyquist.play`, it never measures
                or normalizes the signal, so the callback can write
                full-scale ``[-1, 1]`` audio. False plays output unscaled.
        """
        if num_input_channels < 0 or num_output_channels < 0:
            raise ValueError("Channel counts must be non-negative.")
        if num_input_channels == 0 and num_output_channels == 0:
            raise ValueError("Stream needs at least one input or output channel.")
        self.callback = callback
        self.num_input_channels = num_input_channels
        self.num_output_channels = num_output_channels
        self.block_size = block_size
        self.sample_rate = sample_rate
        self.safe = safe
        self._buffer = Audio.empty(
            block_size,
            max(num_input_channels, num_output_channels),
            sample_rate=sample_rate,
        )
        self._stream: Optional[sd.Stream | sd.InputStream | sd.OutputStream] = None
        self._error: Optional[Exception] = None
        # The main thread's ID while it runs a ``with`` block for this stream,
        # so a callback error can interrupt it.
        self._main_thread_id: Optional[int] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        """Opens the devices and starts calling ``callback``. No-op if running."""
        if self._stream is not None:
            return
        num_in = _device_channels(self.num_input_channels, "input")
        num_out = _device_channels(self.num_output_channels, "output")
        kwargs = dict(
            samplerate=self.sample_rate, blocksize=self.block_size, dtype="float32"
        )
        if num_in and num_out:
            self._stream = sd.Stream(
                channels=(num_in, num_out),
                callback=lambda i, o, *_: self._process(i, o),
                **kwargs,
            )
        elif num_in:
            self._stream = sd.InputStream(
                channels=num_in, callback=lambda i, *_: self._process(i, None), **kwargs
            )
        else:
            self._stream = sd.OutputStream(
                channels=num_out,
                callback=lambda o, *_: self._process(None, o),
                **kwargs,
            )
        self._stream.start()

    def stop(self) -> None:
        """Stops the stream. No-op if not running."""
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def __enter__(self) -> "Stream":
        self.start()
        if threading.current_thread() is threading.main_thread():
            self._main_thread_id = threading.get_ident()
        return self

    def __exit__(self, *exc_info) -> None:
        with self._lock:
            self._main_thread_id = None
        self.stop()
        if self._error is not None:
            # Replaces the KeyboardInterrupt that _report sent to get here.
            error, self._error = self._error, None
            raise error from None

    def _process(
        self, indata: Optional[np.ndarray], outdata: Optional[np.ndarray]
    ) -> None:
        """Runs one block: device input → ``callback`` → device output."""
        try:
            self._buffer.clear()
            if indata is not None:
                # Broadcasts a mono device across all input channels.
                self._buffer.samples[:, : self.num_input_channels] = indata
            self.callback(self._buffer)
            if outdata is not None:
                out = self._buffer.samples[:, : self.num_output_channels]
                if outdata.shape[1] < out.shape[1]:
                    out = out.mean(axis=1, keepdims=True)
                gain = db_to_amplitude(SAFE_PEAK_DBFS) if self.safe else 1.0
                # Broadcasts mono output across a stereo device.
                outdata[:] = gain * np.clip(out, -1.0, 1.0)
        except Exception as e:
            # Left alone, sounddevice would print this traceback and keep
            # calling back, once per block. Report it, and abort.
            self._report(e)
            raise sd.CallbackAbort from e

    def _report(self, error: Exception) -> None:
        """Raises ``error`` in the ``with`` block if we can, or else prints it.

        Exceptions can't cross threads, so we send the main thread a real
        SIGINT, which interrupts even a blocking ``time.sleep`` (Python's
        ``_thread.interrupt_main`` waits for the sleep to end). The resulting
        KeyboardInterrupt unwinds the ``with`` block, and ``__exit__`` swaps
        it for ``error``. We only do this while Python's default SIGINT handler
        is installed, so it's never mistaken for an unrelated Ctrl+C handler.
        """
        with self._lock:
            if (
                self._main_thread_id is not None
                and hasattr(signal, "pthread_kill")  # Not on Windows.
                and signal.getsignal(signal.SIGINT) is signal.default_int_handler
            ):
                thread_id, self._main_thread_id = self._main_thread_id, None
                self._error = error
                signal.pthread_kill(thread_id, signal.SIGINT)
                return
        traceback.print_exception(error)


def _device_channels(num_channels: int, kind: str) -> int:
    """Returns how many channels to open on the current ``kind`` device.

    Mono output opens two channels when it can, so it plays on both speakers.
    Otherwise this is ``num_channels``, capped to what the device has — a cap
    only allowed for mono devices, where :class:`Stream` duplicates or
    averages to fit.
    """
    if num_channels == 0:
        return 0
    device = sd.query_devices(kind=kind)
    available = device[f"max_{kind}_channels"]
    if available == 0 or num_channels > available > 1:
        raise ValueError(
            f"Stream has {num_channels} {kind} channels, but {kind} device "
            f"{device['name']!r} has {available}."
        )
    if kind == "output" and num_channels == 1:
        return min(2, available)
    return min(num_channels, available)
