import json
import re
from urllib.parse import urlparse

from .audio import Audio
from .paths import CACHE_DIR


def fetch_from_freesound(
    id_or_url: str | int,
    *,
    rewrite_credentials: bool = False,
) -> Audio:
    # Parse URL
    # https://freesound.org/people/looplicator/sounds/759259/ -> 759259
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
