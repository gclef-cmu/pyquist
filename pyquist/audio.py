import pathlib
from io import BytesIO
from typing import IO, Optional, Union
from urllib.request import urlopen

import numpy as np
import resampy
import soundfile as sf

from .helper import db_to_amplitude


class Audio:
    """A wrapper around a 2D float32 numpy array of audio samples.

    The two primary attributes are :attr:`samples` (a ``float32`` array
    shaped ``(num_samples, num_channels)``) and :attr:`sample_rate` (Hz, or
    ``None`` for buffers without a defined rate). By convention, sample
    values in ``[-1.0, 1.0]`` correspond to digital full-scale amplitude;
    values outside this range are valid in memory but will clip when sent
    to playback or written to most file formats.

    Example:
        >>> import numpy as np
        >>> import pyquist as pq
        >>> sr = 44100
        >>> t = np.arange(sr) / sr
        >>> audio = pq.Audio(np.sin(2 * np.pi * 440 * t), sample_rate=sr)
        >>> pq.play(audio)
    """

    def __init__(
        self,
        samples: np.ndarray,
        sample_rate: Optional[int] = None,
    ):
        """Wraps an existing numpy array as ``Audio``.

        Args:
            samples: A numpy array of samples. Accepted as 0-D, 1-D, or 2-D
                (see the ``samples`` setter for shape normalization). Must be
                ``float32`` or ``float64`` (the latter is auto-converted).
            sample_rate: Optional sample rate in Hz; ``None`` for unspecified
                (e.g. when used as a real-time block buffer).
        """
        self.samples = samples
        self.sample_rate = sample_rate

    @classmethod
    def zeros(
        cls,
        num_samples: int,
        num_channels: int,
        sample_rate: Optional[int] = None,
    ) -> "Audio":
        """Creates a silent (zero-filled) ``Audio`` of the given shape.

        Useful as a destination buffer that you fill in via ``audio.samples``
        or via in-place arithmetic.

        Args:
            num_samples: Number of samples per channel. Must be ``>= 0``.
            num_channels: Number of channels (1 for mono, 2 for stereo).
                Must be ``>= 0``.
            sample_rate: Optional sample rate in Hz.
        """
        if num_samples < 0:
            raise ValueError("num_samples must be non-negative.")
        if num_channels < 0:
            raise ValueError("num_channels must be non-negative.")
        return cls(
            np.zeros((num_samples, num_channels), dtype=np.float32),
            sample_rate=sample_rate,
        )

    @classmethod
    def from_file(cls, file: Union[str, pathlib.Path, IO]) -> "Audio":
        """Loads an ``Audio`` from a file on disk or a file-like object.

        Decoding is delegated to ``soundfile`` (libsndfile), which supports
        WAV, FLAC, OGG, MP3, and most common formats. The file's native sample
        rate is preserved; channels remain in their original order. Use
        :meth:`resample` to change the rate after loading.

        Raises :class:`FileNotFoundError` (with the offending path) when
        ``file`` is a path that doesn't exist — clearer than libsndfile's
        generic ``"System error"`` message.
        """
        # Pre-check path-like inputs so a missing file produces a useful error
        # instead of LibsndfileError: "Error opening ...: System error.".
        if isinstance(file, (str, pathlib.Path)):
            path = pathlib.Path(file)
            if not path.exists():
                raise FileNotFoundError(f"Audio file not found: {path}.")
        samples, sample_rate = sf.read(file)
        return cls(samples, sample_rate=sample_rate)

    @classmethod
    def from_url(cls, url: str) -> "Audio":
        """Downloads an audio file from a URL and loads it as ``Audio``.

        The full response is buffered in memory before decoding.
        """
        return cls.from_file(BytesIO(urlopen(url).read()))

    # --- Core attributes ----------------------------------------------------

    @property
    def samples(self) -> np.ndarray:
        """The underlying ``(num_samples, num_channels)`` ``float32`` array.

        Returned by reference: in-place mutations (``audio.samples[0] = 0``,
        ``audio.samples *= 0.5``) modify the audio directly. Reassigning the
        attribute (``audio.samples = new_array``) re-runs validation.
        """
        return self._samples

    @samples.setter
    def samples(self, value: np.ndarray) -> None:
        """Validates and stores ``value`` as the underlying sample array.

        Three conveniences are applied before validation:

        * a 0-D array becomes shape ``(1, 1)``;
        * a 1-D array of length ``n`` becomes shape ``(n, 1)`` (mono);
        * a ``float64`` array is cast to ``float32``.

        Anything else with the wrong dtype raises ``TypeError``; arrays with
        more than 2 dimensions raise ``ValueError``. When ``value`` is already
        a 2-D ``float32`` array, it is stored by reference (no copy) — this
        is what allows ``Audio`` to act as a thin view over an externally
        owned buffer (e.g. the ``outdata`` array in a real-time callback).
        """
        if not isinstance(value, np.ndarray):
            raise TypeError(
                f"samples must be a numpy.ndarray, got {type(value).__name__}."
            )
        if value.ndim == 0:
            value = value[np.newaxis, np.newaxis]
        elif value.ndim == 1:
            value = value[:, np.newaxis]
        elif value.ndim > 2:
            raise ValueError(
                f"samples must have shape (num_samples, num_channels); "
                f"got array with {value.ndim} dimensions."
            )
        if value.dtype == np.float64:
            value = value.astype(np.float32)
        if value.dtype != np.float32:
            raise TypeError(f"samples must have dtype np.float32, got {value.dtype}.")
        self._samples = value

    @property
    def sample_rate(self) -> Optional[int]:
        """The sample rate in Hz, or ``None`` if unspecified."""
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value: Optional[int]) -> None:
        """Sets the sample rate.

        Accepts a positive ``int`` or ``None``. Non-int values raise
        ``TypeError``; zero or negative values raise ``ValueError``.
        """
        if value is None:
            self._sample_rate = None
            return
        if not isinstance(value, (int, np.integer)):
            raise TypeError(
                f"sample_rate must be int or None, got {type(value).__name__}."
            )
        if value <= 0:
            raise ValueError(f"sample_rate must be positive, got {value}.")
        self._sample_rate = int(value)

    # --- Shape-derived properties ------------------------------------------

    @property
    def num_samples(self) -> int:
        """Number of samples per channel (``samples.shape[0]``)."""
        return self._samples.shape[0]

    @property
    def num_channels(self) -> int:
        """Number of channels (``samples.shape[1]``); 1 for mono, 2 for stereo."""
        return self._samples.shape[1]

    @property
    def shape(self) -> tuple:
        """Shape of the underlying array: ``(num_samples, num_channels)``."""
        return self._samples.shape

    @property
    def duration(self) -> float:
        """Duration of the audio in seconds. Requires sample_rate to be set."""
        if self._sample_rate is None:
            raise ValueError("Cannot compute duration without a sample_rate.")
        return self.num_samples / self._sample_rate

    @property
    def peak_amplitude(self) -> float:
        """Peak absolute sample value across all samples and channels.

        This is a linear amplitude (not decibels): ``1.0`` corresponds to
        digital full scale. Empty audio returns ``0.0``. Use
        :func:`pyquist.helper.amplitude_to_db` to convert to dBFS.
        """
        if self._samples.size == 0:
            return 0.0
        return float(np.abs(self._samples).max())

    # --- Mutation methods ---------------------------------------------------

    def clear(self) -> None:
        """Fills the audio with silence (zeros) in place.

        Shape, dtype, and ``sample_rate`` are unchanged.
        """
        self._samples.fill(0.0)

    def segment(
        self,
        *,
        offset: Optional[float] = None,
        duration: Optional[float] = None,
    ) -> "Audio":
        """Returns a new ``Audio`` containing a time-slice of this one.

        Both ``offset`` and ``duration`` are in seconds and require
        ``sample_rate`` to be set. Out-of-range values are clamped: a negative
        ``offset`` is treated as zero, and a ``duration`` that runs past the
        end is truncated. With both arguments ``None`` this is a no-op that
        returns ``self``.

        Args:
            offset: Start time in seconds. Defaults to the beginning.
            duration: Length in seconds. Defaults to the rest of the audio.

        Returns:
            A new ``Audio`` carrying the same ``sample_rate`` as ``self``.
        """
        if offset is None and duration is None:
            return self
        if self._sample_rate is None:
            raise ValueError("segment() requires a sample_rate.")
        start = max(0, int((offset or 0.0) * self._sample_rate))
        end = (
            self.num_samples
            if duration is None
            else start + int(duration * self._sample_rate)
        )
        start = min(start, self.num_samples)
        end = max(start, min(end, self.num_samples))
        return Audio(self._samples[start:end, :], sample_rate=self._sample_rate)

    def normalize(self, *, peak_dbfs: float = 0.0, in_place: bool = True) -> "Audio":
        """Scales the audio so its peak amplitude matches ``peak_dbfs``.

        ``peak_dbfs`` is measured in decibels relative to digital full scale
        (dBFS). ``0.0`` means full-scale (peak = 1.0); ``-6.0`` means roughly
        half full-scale (peak ≈ 0.501); positive values exceed full scale and
        will clip on playback. Silent audio (all zeros) is returned unchanged.

        Args:
            peak_dbfs: Target peak level in dBFS. Defaults to ``0.0``.
            in_place: If ``True`` (default), modifies and returns ``self``.
                If ``False``, returns a new ``Audio`` and leaves the original
                untouched.
        """
        peak = self.peak_amplitude
        if peak == 0.0:
            gain = 1.0
        else:
            gain = float(db_to_amplitude(peak_dbfs)) / peak
        if in_place:
            self._samples *= gain
            return self
        return Audio(self._samples * gain, sample_rate=self._sample_rate)

    def clip(self, *, peak_amplitude: float = 1.0, in_place: bool = True) -> "Audio":
        """Symmetrically clamps every sample to ``[-peak_amplitude, +peak_amplitude]``.

        This is a hard clip — samples beyond the threshold are truncated, not
        scaled. To rescale instead, use :meth:`normalize`.

        Args:
            peak_amplitude: Symmetric clip threshold in linear amplitude.
                Defaults to ``1.0`` (digital full scale).
            in_place: If ``True`` (default), modifies and returns ``self``.
                If ``False``, returns a new ``Audio`` and leaves the original
                untouched.
        """
        clipped = np.clip(self._samples, -peak_amplitude, peak_amplitude)
        if in_place:
            self._samples[:] = clipped
            return self
        return Audio(clipped, sample_rate=self._sample_rate)

    def as_mono(self) -> "Audio":
        """Returns a mono (1-channel) version of the audio.

        Multi-channel audio is mixed down by averaging across channels
        (mean, not sum), which preserves perceived loudness without risking
        clipping. If the audio is already mono, returns ``self`` (no copy).
        """
        if self.num_channels == 1:
            return self
        mono = self._samples.mean(axis=1, keepdims=True).astype(np.float32)
        return Audio(mono, sample_rate=self._sample_rate)

    def as_stereo(self) -> "Audio":
        """Returns a stereo (2-channel) version of the audio.

        Mono audio is duplicated across both channels (the same signal in
        L and R). Stereo audio is returned as ``self`` (no copy). Audio with
        3 or more channels raises ``ValueError`` — this method does not try
        to guess a downmix.
        """
        if self.num_channels == 2:
            return self
        if self.num_channels == 1:
            stereo = np.repeat(self._samples, 2, axis=1)
            return Audio(stereo, sample_rate=self._sample_rate)
        raise ValueError(
            f"Cannot convert audio with {self.num_channels} channels to stereo."
        )

    def resample(self, new_sample_rate: int, **kwargs) -> "Audio":
        """Returns a new ``Audio`` resampled to ``new_sample_rate``.

        Resampling is performed by ``resampy`` using a polyphase bandlimited
        filter; extra keyword arguments (e.g. ``filter='kaiser_fast'``) are
        forwarded to :func:`resampy.resample`. The number of channels is
        preserved; the number of samples scales by
        ``new_sample_rate / self.sample_rate``.

        Raises ``ValueError`` if ``self.sample_rate`` is ``None`` or
        ``new_sample_rate`` is non-positive.
        """
        if self._sample_rate is None:
            raise ValueError("Cannot resample without a sample_rate.")
        if not isinstance(new_sample_rate, (int, np.integer)):
            raise TypeError("new_sample_rate must be an int.")
        if new_sample_rate <= 0:
            raise ValueError("new_sample_rate must be positive.")
        resampled = resampy.resample(
            self._samples, self._sample_rate, new_sample_rate, axis=0, **kwargs
        )
        return Audio(resampled, sample_rate=new_sample_rate)

    def write(self, file: Union[str, IO], **kwargs) -> None:
        """Writes the audio to a file via ``soundfile``.

        The output format is inferred from the file extension (``.wav``,
        ``.flac``, ``.ogg``, ...). Extra keyword arguments are forwarded to
        :func:`soundfile.write` (e.g. ``subtype='PCM_24'``). Samples outside
        ``[-1.0, 1.0]`` will clip in fixed-point formats; consider calling
        :meth:`clip` or :meth:`normalize` first.

        Raises ``ValueError`` if ``self.sample_rate`` is ``None``.
        """
        if self._sample_rate is None:
            raise ValueError("Cannot write audio without a sample_rate.")
        sf.write(file, self._samples, self._sample_rate, **kwargs)

    # --- numpy interop ------------------------------------------------------

    def __array__(self, dtype=None) -> np.ndarray:
        if dtype is None:
            return self._samples
        return self._samples.astype(dtype)

    # --- Indexing / length --------------------------------------------------

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, key):
        return self._samples[key]

    def __setitem__(self, key, value) -> None:
        self._samples[key] = value

    # --- Arithmetic ---------------------------------------------------------

    def _check_compatible(self, other: "Audio") -> Optional[int]:
        """Validates that ``other`` can be combined with ``self`` and returns
        the sample rate the result should carry."""
        if self.shape != other.shape:
            raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}.")
        if (
            self._sample_rate is not None
            and other._sample_rate is not None
            and self._sample_rate != other._sample_rate
        ):
            raise ValueError(
                f"Sample rate mismatch: {self._sample_rate} != {other._sample_rate}."
            )
        return (
            self._sample_rate if self._sample_rate is not None else other._sample_rate
        )

    def _binary_op(self, other, op) -> "Audio":
        if isinstance(other, Audio):
            sr = self._check_compatible(other)
            return Audio(op(self._samples, other._samples), sample_rate=sr)
        return Audio(op(self._samples, other), sample_rate=self._sample_rate)

    def _ibinary_op(self, other, op) -> "Audio":
        if isinstance(other, Audio):
            self._check_compatible(other)
            op(self._samples, other._samples)
        else:
            op(self._samples, other)
        return self

    def __add__(self, other) -> "Audio":
        return self._binary_op(other, lambda a, b: a + b)

    def __radd__(self, other) -> "Audio":
        return Audio(other + self._samples, sample_rate=self._sample_rate)

    def __iadd__(self, other) -> "Audio":
        def _iadd(a, b):
            a += b

        return self._ibinary_op(other, _iadd)

    def __sub__(self, other) -> "Audio":
        return self._binary_op(other, lambda a, b: a - b)

    def __rsub__(self, other) -> "Audio":
        return Audio(other - self._samples, sample_rate=self._sample_rate)

    def __isub__(self, other) -> "Audio":
        def _isub(a, b):
            a -= b

        return self._ibinary_op(other, _isub)

    def __mul__(self, other) -> "Audio":
        return self._binary_op(other, lambda a, b: a * b)

    def __rmul__(self, other) -> "Audio":
        return Audio(other * self._samples, sample_rate=self._sample_rate)

    def __imul__(self, other) -> "Audio":
        def _imul(a, b):
            a *= b

        return self._ibinary_op(other, _imul)

    def __truediv__(self, other) -> "Audio":
        return self._binary_op(other, lambda a, b: a / b)

    def __itruediv__(self, other) -> "Audio":
        def _idiv(a, b):
            a /= b

        return self._ibinary_op(other, _idiv)

    def __neg__(self) -> "Audio":
        return Audio(-self._samples, sample_rate=self._sample_rate)

    def __repr__(self) -> str:
        return (
            f"Audio(num_samples={self.num_samples}, "
            f"num_channels={self.num_channels}, "
            f"sample_rate={self._sample_rate})"
        )
