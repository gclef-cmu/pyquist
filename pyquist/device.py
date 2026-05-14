"""Sounddevice integration: device selection, ``play``, and ``record``.

The user's chosen input/output devices are persisted as JSON under
``CACHE_DIR / "device_defaults.json"`` and applied to ``sounddevice.default``
once at module import time.
"""

import json
import sys
import time
from typing import Any, Optional, Tuple, Union

import sounddevice as sd
import tqdm

from .audio import Audio
from .paths import CACHE_DIR

_DEFAULTS_PATH = CACHE_DIR / "device_defaults.json"

DeviceRef = Union[int, str]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_devices() -> None:
    """Prints all available input and output devices, grouped by kind."""
    devices = sd.query_devices()
    for kind in ("input", "output"):
        print(f"{kind.capitalize()} devices:")
        for i, dev in enumerate(devices):
            if dev[f"max_{kind}_channels"] > 0:
                print(f"  {i}: {dev['name']}")
        print()


def set_input_device(
    device_id_or_name: Optional[DeviceRef] = None,
    *,
    update_default: bool = False,
) -> None:
    """Selects the input device for the current Python session.

    Args:
        device_id_or_name: Device index, or a substring of the device name.
            If ``None``, prompts the user interactively.
        update_default: If ``False`` (default), the change applies only to
            this Python session. If ``True``, the choice is also persisted
            to the cache and reapplied next time pyquist is imported. The
            ``pyquist devices`` CLI command is the usual way to update the
            persistent default.
    """
    if device_id_or_name is None:
        device_id, device_name = _prompt_device("input", update_default=update_default)
    else:
        device_id, device_name = _resolve_device(device_id_or_name, "input")
    _set_device_slot(0, device_id)
    if update_default:
        defaults = _load_defaults()
        defaults["input"] = device_name
        _save_defaults(defaults)


def set_output_device(
    device_id_or_name: Optional[DeviceRef] = None,
    *,
    update_default: bool = False,
) -> None:
    """Selects the output device for the current Python session.

    Args:
        device_id_or_name: Device index, or a substring of the device name.
            If ``None``, prompts the user interactively.
        update_default: If ``False`` (default), the change applies only to
            this Python session. If ``True``, the choice is also persisted
            to the cache and reapplied next time pyquist is imported. The
            ``pyquist devices`` CLI command is the usual way to update the
            persistent default.
    """
    if device_id_or_name is None:
        device_id, device_name = _prompt_device("output", update_default=update_default)
    else:
        device_id, device_name = _resolve_device(device_id_or_name, "output")
    _set_device_slot(1, device_id)
    if update_default:
        defaults = _load_defaults()
        defaults["output"] = device_name
        _save_defaults(defaults)


def play(audio: Audio, *, safe: bool = True, normalize: bool = False) -> None:
    """Plays the audio from the default output device.

    Args:
        audio: The audio to play. Must have a ``sample_rate``.
        safe: If True (default), attenuates the audio to -18 dBFS before
            playback to protect ears against accidentally hot signals.
        normalize: If True, normalizes the audio to 0 dBFS before playback.
    """
    if normalize:
        audio = audio.normalize(in_place=False)
    audio = audio.clip(in_place=False)
    if safe:
        audio = audio.normalize(peak_dbfs=-18.0, in_place=False)
    sd.play(audio, audio.sample_rate)
    sd.wait()


def record(duration: float, *, progress_bar: bool = True, **kwargs: Any) -> Audio:
    """Records audio from the default input device.

    Args:
        duration: Recording length in seconds.
        progress_bar: Whether to display a tqdm progress bar.

    Returns:
        The recorded Audio at the input device's native sample rate.
    """
    device_info = sd.query_devices(sd.default.device[0])
    sample_rate = round(device_info["default_samplerate"])
    num_channels = device_info["max_input_channels"]
    samples = sd.rec(
        frames=int(duration * sample_rate),
        channels=num_channels,
        samplerate=sample_rate,
        dtype="float32",
        **kwargs,
    )
    if progress_bar:
        with tqdm.tqdm(total=100, desc="Recording") as pbar:
            for _ in range(100):
                pbar.update(1)
                time.sleep(duration / 100)
    sd.wait()
    return Audio(samples, sample_rate=sample_rate)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _resolve_device(device_id_or_name: DeviceRef, kind: str) -> Tuple[int, str]:
    """Resolves an int ID or name substring to a ``(device_id, device_name)`` pair.

    Args:
        device_id_or_name: Device index, or a substring of the device name.
        kind: ``"input"`` or ``"output"``.

    Raises:
        TypeError: if ``device_id_or_name`` is not int or str.
        ValueError: if the ID is out of range, has no channels of the given
            kind, or if the name matches zero or multiple devices.
    """
    devices = sd.query_devices()
    chan_key = f"max_{kind}_channels"

    if isinstance(device_id_or_name, bool) or not isinstance(
        device_id_or_name, (int, str)
    ):
        raise TypeError(
            f"device must be an int ID or str name, "
            f"got {type(device_id_or_name).__name__}."
        )

    if isinstance(device_id_or_name, int):
        if not 0 <= device_id_or_name < len(devices):
            raise ValueError(f"Device ID {device_id_or_name} out of range.")
        dev = devices[device_id_or_name]
        if dev[chan_key] == 0:
            raise ValueError(
                f"Device {device_id_or_name} ('{dev['name']}') has no {kind} channels."
            )
        return device_id_or_name, dev["name"]

    matches = [
        (i, dev["name"])
        for i, dev in enumerate(devices)
        if device_id_or_name in dev["name"] and dev[chan_key] > 0
    ]
    if not matches:
        raise ValueError(f"No {kind} device matches '{device_id_or_name}'.")
    if len(matches) > 1:
        names = ", ".join(name for _, name in matches)
        raise ValueError(
            f"Multiple {kind} devices match '{device_id_or_name}': {names}."
        )
    return matches[0]


def _prompt_device(kind: str, *, update_default: bool = False) -> Tuple[int, str]:
    """Interactively prompt the user to pick a device of the given kind.

    Each device in the listing is annotated with ``[current]`` if it is the
    one currently active in ``sd.default``, and ``[default]`` if it is the
    persisted default for this kind. The prompt phrasing reflects whether
    the choice will be persisted (``update_default=True``) or applied for
    this session only.
    """
    devices = sd.query_devices()
    slot = 0 if kind == "input" else 1
    current_id = sd.default.device[slot]

    # Resolve the persisted default name (if any) to its current device ID.
    default_id: Optional[int] = None
    default_name = _load_defaults().get(kind)
    if default_name is not None:
        try:
            default_id, _ = _resolve_device(default_name, kind)
        except (ValueError, TypeError):
            pass  # Cached default no longer resolvable; just don't tag it.

    print(f"Available {kind} devices:")
    for i, dev in enumerate(devices):
        if dev[f"max_{kind}_channels"] == 0:
            continue
        tags = []
        if i == current_id:
            tags.append("current")
        if i == default_id:
            tags.append("default")
        suffix = f" [{', '.join(tags)}]" if tags else ""
        print(f"  {i}: {dev['name']}{suffix}")

    label = f"default {kind}" if update_default else kind
    choice = input(f"Select {label} device (ID or name): ").strip()
    if choice.isdigit() or (choice.startswith("-") and choice[1:].isdigit()):
        return _resolve_device(int(choice), kind)
    return _resolve_device(choice, kind)


def _set_device_slot(slot: int, device_id: int) -> None:
    """Updates one half of ``sd.default.device`` without disturbing the other."""
    current = list(sd.default.device)
    current[slot] = device_id
    sd.default.device = tuple(current)


def _load_defaults() -> dict:
    if not _DEFAULTS_PATH.exists():
        return {}
    try:
        with open(_DEFAULTS_PATH, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"warning: could not read device defaults cache: {e}", file=sys.stderr)
        return {}


def _save_defaults(defaults: dict) -> None:
    _DEFAULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_DEFAULTS_PATH, "w") as f:
        json.dump(defaults, f)


def _apply_persisted_defaults() -> None:
    """Applies cached input/output device choices to ``sd.default``.

    Called once at module import. Per-key failures (e.g. a device name that
    no longer exists) log a warning to stderr and are skipped, but they do
    not mutate the on-disk cache.
    """
    defaults = _load_defaults()
    for kind, slot in [("input", 0), ("output", 1)]:
        name = defaults.get(kind)
        if name is None:
            continue
        try:
            device_id, _ = _resolve_device(name, kind)
            _set_device_slot(slot, device_id)
        except (ValueError, TypeError) as e:
            print(
                f"warning: could not restore {kind} device {name!r}: {e}",
                file=sys.stderr,
            )


# ---------------------------------------------------------------------------
# Module init
# ---------------------------------------------------------------------------

_apply_persisted_defaults()
