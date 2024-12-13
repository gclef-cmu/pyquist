import json
import pathlib
import re
from io import BytesIO
from typing import Tuple
from urllib.parse import urlparse

import requests

from ..audio import Audio
from ..paths import CACHE_DIR as _ROOT_CACHE_DIR

_CACHE_DIR = _ROOT_CACHE_DIR / "freesound"


def _get_client_credentials(reauthenticate: bool) -> Tuple[str, str]:
    # Load API from cache or prompt user
    api_key_path = _CACHE_DIR / "api_key.json"
    if not api_key_path.exists() or reauthenticate:
        print("Please go to the following URL and create or retrieve API credentials:")
        print("https://freesound.org/apiv2/apply")
        client_id = input("Create and enter FreeSound client ID from: ")
        client_secret = input("Enter FreeSound client secret: ")
        with open(api_key_path, "w") as f:
            json.dump({"client_id": client_id, "client_secret": client_secret}, f)
    else:
        with open(api_key_path, "r") as f:
            d = json.load(f)
            client_secret = d["client_secret"].strip()
            client_id = d["client_id"].strip()

    return client_id, client_secret


def _get_oauthv2_token(reauthenticate: bool) -> str:
    token_path = _CACHE_DIR / "oauthv2_token.json"
    if not token_path.exists() or reauthenticate:
        client_id, client_secret = _get_client_credentials(reauthenticate)

        # Prompt user to authorize their application in their browser
        authorization_url = "https://freesound.org/apiv2/oauth2/authorize/"
        auth_url = f"{authorization_url}?client_id={client_id}&response_type=code"
        print("Please go to the following URL to authorize the app:")
        print(auth_url)
        authorization_code = input("Enter the authorization code: ")

        # Exchange authorization code for an access token
        data = {
            "client_id": client_id,
            "client_secret": client_secret,
            "grant_type": "authorization_code",
            "code": authorization_code,
        }
        token_url = "https://freesound.org/apiv2/oauth2/access_token/"
        response = requests.post(token_url, data=data)
        if response.status_code != 200:
            raise Exception(
                f"Error retrieving access token: {response.status_code} - {response.text}"
            )
        token_data = response.json()
        if "access_token" not in token_data:
            raise Exception(f"Error retrieving access token: {token_data}")
        token_data["client_id"] = client_id
        token_data["client_secret"] = client_secret

        # Store token for future use
        with open(token_path, "w") as f:
            json.dump(token_data, f)
    else:
        with open(token_path, "r") as f:
            token_data = json.load(f)
        # TODO: implement refresh logic
    return token_data["access_token"]


def fetch_metadata(
    sound_id: int,
    *,
    reauthenticate: bool = False,
    cache_dir: pathlib.Path = _CACHE_DIR,
) -> dict:
    path = cache_dir / str(sound_id) / "metadata.json"
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    _, client_secret = _get_client_credentials(reauthenticate)
    metadata_url = f"https://freesound.org/apiv2/sounds/{sound_id}"
    response = requests.get(metadata_url, params={"token": client_secret})
    if response.status_code == 404:
        raise ValueError(f"Sound {sound_id} not found.")
    elif response.status_code != 200:
        raise Exception(
            f"Error retrieving sound info: {response.status_code} - {response.text}"
        )
    result = response.json()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f)
    return result


# TODO: Cache downloaded files
def fetch_audio(
    sound_id: int,
    *,
    preview_okay: bool = True,
    preview_tag: str = "preview-hq-ogg",
    reauthenticate: bool = False,
    cache_dir: pathlib.Path = _CACHE_DIR,
) -> Tuple[BytesIO, dict]:
    # Get metadata
    metadata = fetch_metadata(sound_id, reauthenticate=reauthenticate)

    # Set up audio request
    if preview_okay:
        if preview_tag not in metadata["previews"]:
            raise ValueError(
                f"'{preview_tag}' not found among {list(metadata['previews'].keys())}."
            )
        if not (preview_tag.endswith("-ogg") or preview_tag.endswith("-mp3")):
            raise ValueError("Preview tag must end with '-ogg' or '-mp3'.")
        path = cache_dir / str(sound_id) / (preview_tag[:-4] + "." + preview_tag[-3:])
        url = metadata["previews"][preview_tag]
        headers = None
    else:
        path = cache_dir / str(sound_id) / f"original.{metadata['type']}"
        url = f"https://freesound.org/apiv2/sounds/{sound_id}/download/"
        access_token = _get_oauthv2_token(reauthenticate=reauthenticate)
        headers = {"Authorization": f"Bearer {access_token}"}

    # Fetch audio
    if path.exists():
        with open(path, "rb") as f:
            audio_bytes = f.read()
    else:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise Exception(
                f"Error retrieving sound file: {response.status_code} - {response.text}"
            )
        audio_bytes = response.content
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(audio_bytes)

    return BytesIO(audio_bytes), metadata


def url_to_id(id_or_url: str) -> int:
    url = urlparse(id_or_url)
    if url.netloc == "freesound.org":
        # Check if ends in sounds/<id>
        match = re.match(r".*/sounds/(\d+)", url.path)
        if not match:
            raise ValueError("Invalid FreeSound URL.")
        sound_id = int(match.group(1))
    else:
        # Assume it's an ID
        try:
            sound_id = int(id_or_url)
        except ValueError:
            raise ValueError("Invalid FreeSound URL or ID.")
    return sound_id


def fetch(
    id_or_url: int | str,
    *,
    preview_okay: bool = True,
    preview_tag: str = "preview-hq-ogg",
    reauthenticate: bool = False,
    cache_dir: pathlib.Path = _CACHE_DIR,
) -> Tuple[Audio, dict]:
    # Parse URL
    # https://freesound.org/people/looplicator/sounds/759259/ -> 759259
    if isinstance(id_or_url, str):
        sound_id = url_to_id(id_or_url)
    else:
        sound_id = id_or_url

    audio_bytes, metadata = fetch_audio(
        sound_id,
        preview_okay=preview_okay,
        preview_tag=preview_tag,
        reauthenticate=reauthenticate,
        cache_dir=cache_dir,
    )
    return Audio.from_file(audio_bytes), metadata


# Alias for backwards compatibility
def fetch_from_freesound(*args, **kwargs) -> Audio:
    return fetch(*args, **kwargs)[0]


if __name__ == "__main__":
    import sys

    from ..cli import play

    audio, metadata = fetch(sys.argv[1], preview_okay=True)
    print(json.dumps(metadata, indent=2))
    play(audio)
