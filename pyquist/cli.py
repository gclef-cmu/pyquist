import json
import time
from typing import Any, Dict

import sounddevice as sd
import tqdm

from . import Audio
from .paths import CACHE_DIR

_DEFAULTS: Dict[str, Any] = {}
_SD_DEFAULTS_PATH = CACHE_DIR / "sounddevice_defaults.json"


def _restore_sd_defaults():
    global _DEFAULTS
    if _DEFAULTS is None:
        if _SD_DEFAULTS_PATH.exists():
            with open(_SD_DEFAULTS_PATH, "r") as f:
                _DEFAULTS = json.load(f)
    for k, v in _DEFAULTS.items():
        _set_sd_default(k, v, write=False)


_restore_sd_defaults()


def _set_sd_default(name: str, value: Any, write: bool = True):
    global _DEFAULTS

    # Check if the name is a valid sounddevice default
    try:
        getattr(sd.default, name)
    except AttributeError:
        raise ValueError(f"Invalid sounddevice default: {name}")

    # Set the default device by name instead of ID
    if name == "device":
        if not (
            isinstance(value, tuple)
            and len(value) == 2
            and all(isinstance(d, str) for d in value)
        ):
            raise TypeError("device must be a tuple of two strings")
        input_name, output_name = value
        input_ids = [
            id
            for id, device in enumerate(sd.query_devices())
            if input_name in device["name"] and device["max_input_channels"] > 0
        ]
        output_ids = [
            id
            for id, device in enumerate(sd.query_devices())
            if output_name in device["name"] and device["max_output_channels"] > 0
        ]
        if len(input_ids) == 0:
            raise ValueError(f"Input device '{input_name}' not found")
        if len(input_ids) > 1:
            raise ValueError(
                f"Multiple input devices found that include '{input_name}'"
            )
        if len(output_ids) == 0:
            raise ValueError(f"Output device '{output_name}' not found")
        if len(output_ids) > 1:
            raise ValueError(
                f"Multiple output devices found that include '{output_name}'"
            )
        setattr(sd.default, name, (input_ids[0], output_ids[0]))
    else:
        setattr(sd.default, name, value)

    # Update cache
    _DEFAULTS[name] = value
    if write:
        with open(_SD_DEFAULTS_PATH, "w") as f:
            json.dump(_DEFAULTS, f)


def set_sd_defaults(**kwargs):
    for k, v in kwargs.items():
        _set_sd_default(k, v, write=True)


def play(audio: Audio, *, safe: bool = True, normalize: bool = False):
    """Plays the audio using sounddevice."""
    if normalize:
        audio = audio.normalize(in_place=False)
    audio = audio.clip(in_place=False)
    if safe:
        audio = audio.normalize(peak_dbfs=-18.0, in_place=False)
    sd.play(audio.swapaxes(0, 1), audio.sample_rate)
    sd.wait()


def record(duration: float, *, progress_bar: bool = True, **kwargs) -> Audio:
    """
    Records audio from the default input device.

    Parameters:
        duration: The duration of the recording in seconds.
        progress_bar: Whether to display a progress bar.

    Returns:
        The recorded audio.
    """

    # Start recording
    sample_rate = round(sd.query_devices(sd.default.device[0])["default_samplerate"])
    num_channels = sd.query_devices(sd.default.device[0])["max_input_channels"]
    num_samples = round(duration * sample_rate)
    audio = sd.rec(
        frames=num_samples,
        channels=num_channels,
        samplerate=sample_rate,
        dtype="float32",
        **kwargs,
    )

    # Progress bar for duration seconds using tqdm
    if progress_bar:
        with tqdm.tqdm(total=100, desc="Recording") as pbar:
            for _ in range(100):
                pbar.update(1)
                time.sleep(duration / 100)

    # Wait until stream is finished
    sd.wait()

    return Audio.from_array(audio.swapaxes(0, 1), sample_rate)


if __name__ == "__main__":
    import argparse

    # Usage: python -m pyquist.cli [mode] [arguments]
    parser = argparse.ArgumentParser(
        description="Record audio from the default input device."
    )
    parser.add_argument("mode", choices=["devices", "record"], help="The mode to run.")
    parser.add_argument("--output", "-o", help="The output file path.")
    parser.add_argument(
        "--duration",
        "-d",
        type=float,
        help="The duration of the recording in seconds.",
    )

    args = parser.parse_args()
    if args.mode == "devices":
        print("Input devices:")

        new_devices = []
        for i, t in enumerate(["input", "output"]):
            device_id = sd.default.device[i]
            device_name = sd.query_devices(device_id)["name"]
            for j, device in enumerate(sd.query_devices()):
                if device[f"max_{t}_channels"] > 0:
                    print(f"  {j}: {device['name']}")
            print(f"Current default {t} device: {device_id}: {device_name}")
            new_device = input(
                f"New {t} device ID or name (leave blank to keep the current device): "
            )
            try:
                new_device_id = int(new_device)
                # User input a device ID
                if new_device_id < 0 or new_device_id >= len(sd.query_devices()):
                    raise ValueError("Invalid device ID")
                new_device_name = sd.query_devices(new_device_id)["name"]
            except ValueError:
                if len(new_device.strip()) == 0:
                    # User input nothing
                    new_device_name = device_name
                else:
                    # User input a device name
                    new_device_name = new_device
            new_devices.append(new_device_name)

        set_sd_defaults(device=tuple(new_devices))

    elif args.mode == "record":
        args = parser.parse_args()

        audio = record(duration=args.duration)
        if args.output is not None:
            audio.write(args.output)
        else:
            play(audio)


if __name__ == "__main__":
    import argparse
