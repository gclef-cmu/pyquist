import json
from io import BytesIO
from typing import IO, Optional
from urllib.parse import urlparse
from urllib.request import urlopen

import numpy as np

from .helper import dbfs_to_gain
from .paths import CACHE_DIR


class AudioBuffer(np.ndarray):
    """
    Represents a buffer of raw audio samples.

    Type signature inspired by JUCE AudioBuffer.

    https://docs.juce.com/master/classAudioBuffer.html
    """

    def __new__(
        cls,
        *,
        num_channels: int,
        num_samples: int,
        array: Optional[np.ndarray] = None,
    ):
        """
        Initializes a buffer of raw 32-bit float audio samples.

        Parameters:
            num_channels: The number of channels in the buffer.
            num_samples: The number of samples in the buffer.
            array: An optional numpy array to use as the buffer.
                   If None, a zero-filled buffer will be created.
                   Must be same shape as num_channels / num_samples.
        """
        if num_channels < 0:
            raise ValueError("The number of channels must be non-negative.")
        if num_samples < 0:
            raise ValueError("The number of samples must be non-negative.")
        if array is None:
            array = np.zeros((num_channels, num_samples), dtype=np.float32)
        else:
            if array.shape != (num_channels, num_samples):
                raise ValueError("Array shape must match num_channels and num_samples.")
            if array.dtype != np.float32:
                raise TypeError("Array must have dtype np.float32.")
        obj = array.view(cls)
        return obj

    def __getitem__(self, *args, **kwargs):
        """Ensure any slices return np.ndarray instead of AudioBuffer."""
        return super().__getitem__(*args, **kwargs).view(np.ndarray)

    def __array_wrap__(self, result, *args, **kwargs):
        """Ensures that output of ufuncs like sum() is np.ndarray or scalar."""
        if result.ndim == 0:
            # Scalar result: return as scalar type (float, int, etc.)
            return result[()]  # Equivalent to scalar conversion
        else:
            # For array-like results, return as np.ndarray
            return super().__array_wrap__(result, *args, **kwargs).view(np.ndarray)

    @property
    def num_channels(self) -> int:
        """Returns the number of channels in the buffer."""
        return self.shape[0]

    @property
    def num_samples(self) -> int:
        """Returns the number of samples in the buffer."""
        return self.shape[1]

    def clear(self):
        """Clears the buffer by zeroing all samples."""
        self.fill(0.0)


class Audio(AudioBuffer):

    def __new__(
        cls,
        *args,
        sample_rate: int,
        **kwargs,
    ):
        """
        Initializes an audio waveform (AudioBuffer with a sample rate).

        Parameters:
            sample_rate: The sample rate of the audio.
        """
        if sample_rate <= 0:
            raise ValueError("The number of samples must be positive.")
        obj = super().__new__(cls, *args, **kwargs)
        obj._sample_rate = sample_rate
        return obj

    @property
    def sample_rate(self) -> int:
        """Returns the sample rate of the audio."""
        return self._sample_rate

    @property
    def duration(self) -> float:
        """Returns the duration of the audio in seconds."""
        return self.num_samples / self.sample_rate

    @property
    def peak_gain(self) -> float:
        """Returns the peak amplitude of the audio."""
        return float(np.abs(self).max())

    def normalize(self, *, peak_dbfs: float = 0.0, in_place: bool = True) -> "Audio":
        """Returns a normalized version of the audio."""
        peak_gain = self.peak_gain
        if peak_gain == 0.0:
            gain = 1.0
        else:
            gain = dbfs_to_gain(peak_dbfs) / peak_gain
        if in_place:
            self[:] *= gain
            result = self
        else:
            result = Audio.from_array(self * gain, self.sample_rate)
        return result

    def clip(self, *, peak_gain: float = 1.0, in_place: bool = True) -> "Audio":
        """Clips the audio to a peak gain."""
        clipped = np.clip(self, -peak_gain, peak_gain)
        if in_place:
            self[:] = clipped
            result = self
        else:
            result = Audio.from_array(clipped, self.sample_rate)
        return result

    def resample(self, new_sample_rate: int, **kwargs) -> "Audio":
        """
        Resamples the waveform to a new sample rate.

        Parameters:
            new_sample_rate: The new sample rate to resample to.
        """
        import resampy

        resampled = resampy.resample(
            self, self.sample_rate, new_sample_rate, axis=1, **kwargs
        )
        return Audio(
            num_channels=self.num_channels,
            num_samples=resampled.shape[1],
            array=resampled,
            sample_rate=new_sample_rate,
        )

    def write(self, file: str | IO, **kwargs):
        """
        Writes the audio to a file.

        Parameters:
            file: The path to the file or a file-like object.
        """
        import soundfile as sf

        sf.write(file, self.T, self.sample_rate, **kwargs)

    @classmethod
    def from_array(cls, array: np.ndarray, sample_rate: int) -> "Audio":
        """
        Creates an AudioBuffer from a numpy array.

        Parameters:
            array: The numpy array of audio samples.
            sample_rate: The sample rate of the audio.
        """
        if array.ndim == 0:
            array = array[np.newaxis, np.newaxis]
        elif array.ndim == 1:
            array = array[np.newaxis, :]
        elif array.ndim > 2:
            raise ValueError("Array must have shape (num_channels, num_samples).")
        return Audio(
            num_channels=array.shape[0],
            num_samples=array.shape[1],
            array=array,
            sample_rate=sample_rate,
        )

    @classmethod
    def from_file(cls, file: str | IO) -> "Audio":
        """
        Creates an AudioBuffer from a file.

        Parameters:
            file: The path to the file or a file-like object.
        """
        import soundfile as sf

        array, sample_rate = sf.read(file)
        if array.dtype == np.float64:
            array = array.astype(np.float32)
        if array.ndim == 2:
            array = array.T
        return cls.from_array(array, sample_rate)

    @classmethod
    def from_url(cls, url: str) -> "Audio":
        """
        Creates an AudioBuffer from a URL.

        Parameters:
            url: The URL of the audio file.
        """

        return cls.from_file(BytesIO(urlopen(url).read()))

    @classmethod
    def from_freesound(
        cls,
        id_or_url: str | int,
        *,
        rewrite_credentials: bool = False,
    ) -> "Audio":
        # Parse URL
        # https://freesound.org/people/looplicator/sounds/759259/ -> 759259
        import re

        sound_id = id_or_url
        if isinstance(id_or_url, str):
            url = urlparse(id_or_url)
            if url.netloc == "freesound.org":
                # Check if ends in sounds/<id>
                match = re.match(r"/sounds/(\d+)", url.path)
                if not match:
                    raise ValueError("Invalid FreeSound URL.")
                sound_id = int(match.group(1))
            else:
                # Assume it's an ID
                try:
                    sound_id = int(id_or_url)
                except ValueError:
                    raise ValueError("Invalid FreeSound URL or ID.")

        # Load API from cache or prompt user
        api_key_path = CACHE_DIR / "freesound_api_key.json"
        if not api_key_path.exists() or rewrite_credentials:
            client_id = input(
                "Create and enter FreeSound client ID from https://freesound.org/apiv2/apply: "
            )
            client_secret = input("Enter FreeSound client secret: ")
            with open(api_key_path, "w") as f:
                json.dump({"client_id": client_id, "client_secret": client_secret}, f)
        else:
            with open(api_key_path, "r") as f:
                d = json.load(f)
                client_secret = d["client_secret"].strip()
                client_id = d["client_id"].strip()

        # Fetch sound
        sound_id
        raise NotImplementedError()
