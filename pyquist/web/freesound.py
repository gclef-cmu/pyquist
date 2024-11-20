import json
import re
from io import BytesIO
from typing import Tuple
from urllib.parse import urlparse

import requests

from ..audio import Audio
from ..paths import CACHE_DIR


def _get_freesound_client_credentials(reauthenticate: bool) -> Tuple[str, str]:
    # Load API from cache or prompt user
    api_key_path = CACHE_DIR / "freesound_api_key.json"
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


def _get_freesound_oauthv2_token(reauthenticate: bool) -> str:
    token_path = CACHE_DIR / "freesound_oauthv2_token.json"
    if not token_path.exists() or reauthenticate:
        client_id, client_secret = _get_freesound_client_credentials(reauthenticate)

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


# TODO: Cache downloaded files
def fetch_from_freesound(
    id_or_url: str | int,
    *,
    preview_okay: bool = True,
    reauthenticate: bool = False,
) -> Audio:
    # Parse URL
    # https://freesound.org/people/looplicator/sounds/759259/ -> 759259
    sound_id = id_or_url
    if isinstance(id_or_url, str):
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

    if preview_okay:
        # Get high quality preview
        _, client_secret = _get_freesound_client_credentials(reauthenticate)
        metadata_url = f"https://freesound.org/apiv2/sounds/{sound_id}"
        response = requests.get(
            metadata_url, params={"token": client_secret, "fields": "previews"}
        )
        if response.status_code != 200:
            raise Exception(
                f"Error retrieving sound info: {response.status_code} - {response.text}"
            )
        return Audio.from_url(response.json()["previews"]["preview-hq-ogg"])
    else:
        # Get original quality using OAuthV2
        access_token = _get_freesound_oauthv2_token(reauthenticate=reauthenticate)
        headers = {"Authorization": f"Bearer {access_token}"}
        response = requests.get(
            f"https://freesound.org/apiv2/sounds/{sound_id}/download/",
            headers=headers,
        )
        return Audio.from_file(BytesIO(response.content))


if __name__ == "__main__":
    import sys

    from ..cli import play

    audio = fetch_from_freesound(sys.argv[1])
    play(audio)
