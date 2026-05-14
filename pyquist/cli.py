"""Command-line interface for pyquist.

This module is just the argparse dispatch layer. The actual functionality
(playback, recording, device selection) lives in :mod:`pyquist.device`,
and audio loading lives in :mod:`pyquist.audio` / :mod:`pyquist.web`.
"""

import argparse

from .audio import Audio
from .device import (
    play,
    record,
    set_input_device,
    set_output_device,
)
from .web.freesound import fetch_freesound


def _cmd_devices(args: argparse.Namespace) -> None:
    del args
    set_input_device(update_default=True)
    set_output_device(update_default=True)
    print("Saved as default for future pyquist sessions.")


def _cmd_play(args: argparse.Namespace) -> None:
    play(Audio.from_file(args.input))


def _cmd_record(args: argparse.Namespace) -> None:
    audio = record(duration=args.duration)
    if args.output:
        audio.write(args.output)
    else:
        play(audio)


def _cmd_freesound(args: argparse.Namespace) -> None:
    audio, metadata = fetch_freesound(args.url)
    for key in ("id", "name", "url", "license", "description"):
        print(f"{key}: {metadata.get(key, '')}")
    if args.output:
        audio.write(args.output)
    play(audio)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyquist", description="Pyquist command-line interface."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_devices = subparsers.add_parser(
        "devices", help="Interactively select default input and output devices."
    )
    p_devices.set_defaults(func=_cmd_devices)

    p_play = subparsers.add_parser("play", help="Play an audio file.")
    p_play.add_argument("input", help="Path to the audio file.")
    p_play.set_defaults(func=_cmd_play)

    p_record = subparsers.add_parser("record", help="Record audio.")
    p_record.add_argument(
        "output",
        nargs="?",
        help="Output file (optional; plays back instead if omitted).",
    )
    p_record.add_argument(
        "--duration", "-d", type=float, default=10.0, help="Duration in seconds."
    )
    p_record.set_defaults(func=_cmd_record)

    p_freesound = subparsers.add_parser(
        "freesound", help="Fetch a sound from FreeSound and play / save it."
    )
    p_freesound.add_argument("url", help="FreeSound URL or numeric ID.")
    p_freesound.add_argument("--output", "-o", help="Output file (optional).")
    p_freesound.set_defaults(func=_cmd_freesound)

    return parser


def main() -> None:
    args = _build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
