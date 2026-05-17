"""FreeSound API client.

Most users only need :func:`fetch_freesound`, which downloads a sound by URL
or numeric ID and returns it as an :class:`pyquist.Audio`. Previews download
without authentication; fetching the original uncompressed file requires the
FreeSound OAuth2 flow (walked through interactively on first use).

Credentials and downloaded files are cached under ``CACHE_DIR / "freesound"``.
"""

import json
import pathlib
import re
from io import BytesIO
from typing import Optional, Tuple, Union
from urllib.parse import urlparse

import requests

from ..audio import Audio
from ..paths import CACHE_DIR

_CACHE_DIR = CACHE_DIR / "freesound"
_API_KEY_PATH = _CACHE_DIR / "api_key.json"
_OAUTH_TOKEN_PATH = _CACHE_DIR / "oauthv2_token.json"

_API_BASE = "https://freesound.org/apiv2"
_OAUTH_AUTHORIZE_URL = f"{_API_BASE}/oauth2/authorize/"
_OAUTH_TOKEN_URL = f"{_API_BASE}/oauth2/access_token/"

# Matches paths like "/sounds/123456" or "/people/<user>/sounds/123456" with
# an optional trailing slash. Captures the numeric ID.
_URL_PATH_PATTERN = re.compile(r"^/(?:people/[^/]+/)?sounds/(\d+)/?$")

_VALID_PREVIEW_TAGS = (
    "preview-hq-mp3",
    "preview-hq-ogg",
    "preview-lq-mp3",
    "preview-lq-ogg",
)


# ---------------------------------------------------------------------------
# URL / ID parsing
# ---------------------------------------------------------------------------


def url_to_id(id_or_url: Union[int, str]) -> int:
    """Parses a FreeSound URL or numeric ID into an integer sound ID.

    Accepted forms:

    * ``123456`` — raw int
    * ``"123456"`` — numeric string
    * ``"https://freesound.org/sounds/123456/"``
    * ``"https://freesound.org/people/<user>/sounds/123456/"``
    * ``http://`` and ``www.`` variants of the above

    Raises ``ValueError`` if the input isn't a recognized FreeSound URL or
    a numeric ID.
    """
    if isinstance(id_or_url, int):
        return id_or_url
    s = id_or_url.strip()
    if s.isdigit():
        return int(s)
    parsed = urlparse(s)
    host = parsed.netloc.lower().removeprefix("www.")
    if host != "freesound.org":
        raise ValueError(f"Not a FreeSound URL or numeric ID: {id_or_url!r}.")
    match = _URL_PATH_PATTERN.match(parsed.path)
    if not match:
        raise ValueError(f"Could not extract sound ID from URL: {id_or_url!r}.")
    return int(match.group(1))


# ---------------------------------------------------------------------------
# Credential storage
# ---------------------------------------------------------------------------


def _load_json(path: pathlib.Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _save_json(path: pathlib.Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


def _credentials(*, reauthenticate: bool = False) -> Tuple[str, str]:
    """Returns ``(client_id, client_secret)`` — cached or prompted.

    On first use (or when ``reauthenticate=True``) walks the user through
    creating a FreeSound API key.
    """
    if not reauthenticate:
        cached = _load_json(_API_KEY_PATH)
        if cached:
            return cached["client_id"].strip(), cached["client_secret"].strip()

    print("FreeSound API credentials required.")
    print("Create an API key at https://freesound.org/apiv2/apply")
    client_id = input("Client ID: ").strip()
    client_secret = input("Client secret: ").strip()
    _save_json(_API_KEY_PATH, {"client_id": client_id, "client_secret": client_secret})
    return client_id, client_secret


def _oauth_token(*, reauthenticate: bool = False) -> str:
    """Returns a FreeSound OAuth2 access token — cached or freshly obtained.

    Only needed for downloading the original (uncompressed) sound files;
    fetching previews uses the API key alone.

    .. note::
        Refresh-token handling is not implemented. If the cached token
        expires, the user can pass ``reauthenticate=True`` to re-run the
        flow.
    """
    if not reauthenticate:
        cached = _load_json(_OAUTH_TOKEN_PATH)
        if cached and "access_token" in cached:
            return cached["access_token"]

    client_id, client_secret = _credentials(reauthenticate=reauthenticate)
    print("To authorize this app, visit:")
    print(f"  {_OAUTH_AUTHORIZE_URL}?client_id={client_id}&response_type=code")
    code = input("Authorization code: ").strip()

    response = requests.post(
        _OAUTH_TOKEN_URL,
        data={
            "client_id": client_id,
            "client_secret": client_secret,
            "grant_type": "authorization_code",
            "code": code,
        },
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"OAuth token request failed: {response.status_code} - {response.text}"
        )
    token_data = response.json()
    if "access_token" not in token_data:
        raise RuntimeError(f"OAuth response missing access_token: {token_data}")
    # Stash credentials alongside the token for future refresh logic.
    token_data["client_id"] = client_id
    token_data["client_secret"] = client_secret
    _save_json(_OAUTH_TOKEN_PATH, token_data)
    return token_data["access_token"]


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------


def fetch_metadata(sound_id: int, *, reauthenticate: bool = False) -> dict:
    """Fetches the FreeSound JSON metadata for a sound (cached on disk)."""
    cache_path = _CACHE_DIR / str(sound_id) / "metadata.json"
    cached = _load_json(cache_path)
    if cached is not None:
        return cached

    # The FreeSound API accepts the API key as a query-string ``token``.
    _, api_key = _credentials(reauthenticate=reauthenticate)
    response = requests.get(f"{_API_BASE}/sounds/{sound_id}", params={"token": api_key})
    if response.status_code == 404:
        raise ValueError(f"FreeSound sound {sound_id} not found.")
    if response.status_code != 200:
        raise RuntimeError(
            f"FreeSound metadata request failed for sound {sound_id}: "
            f"{response.status_code} - {response.text}"
        )
    metadata = response.json()
    _save_json(cache_path, metadata)
    return metadata


def fetch_audio_bytes(
    sound_id: int,
    *,
    preview_tag: Optional[str] = "preview-hq-ogg",
    reauthenticate: bool = False,
) -> Tuple[bytes, dict]:
    """Downloads the raw audio bytes for a sound, plus its metadata.

    Args:
        sound_id: FreeSound numeric ID.
        preview_tag: Which preview to download. One of ``"preview-hq-ogg"``
            (default), ``"preview-hq-mp3"``, ``"preview-lq-ogg"``, or
            ``"preview-lq-mp3"``. Pass ``None`` to download the original
            uncompressed file instead — this requires OAuth2 and triggers an
            interactive auth flow on first use.
        reauthenticate: Force a re-prompt of API credentials.

    Returns ``(audio_bytes, metadata)``. Both are cached on disk so repeated
    calls for the same sound are free.
    """
    metadata = fetch_metadata(sound_id, reauthenticate=reauthenticate)

    if preview_tag is not None:
        if preview_tag not in _VALID_PREVIEW_TAGS:
            raise ValueError(
                f"Invalid preview_tag {preview_tag!r}; "
                f"must be one of {list(_VALID_PREVIEW_TAGS)} or None."
            )
        url = metadata["previews"][preview_tag]
        # preview_tag has the form "preview-hq-ogg"; the last 3 chars are the extension.
        cache_path = (
            _CACHE_DIR / str(sound_id) / f"{preview_tag[:-4]}.{preview_tag[-3:]}"
        )
        headers = None
    else:
        url = f"{_API_BASE}/sounds/{sound_id}/download/"
        cache_path = _CACHE_DIR / str(sound_id) / f"original.{metadata['type']}"
        token = _oauth_token(reauthenticate=reauthenticate)
        headers = {"Authorization": f"Bearer {token}"}

    if cache_path.exists():
        return cache_path.read_bytes(), metadata

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise RuntimeError(
            f"FreeSound download failed for sound {sound_id}: "
            f"{response.status_code} - {response.text}"
        )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(response.content)
    return response.content, metadata


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def fetch_freesound(
    id_or_url: Union[int, str],
    *,
    preview_tag: Optional[str] = "preview-hq-ogg",
    reauthenticate: bool = False,
) -> Tuple[Audio, dict]:
    """Fetches an ``Audio`` (and metadata) from FreeSound by URL or numeric ID.

    Example::

        from pyquist.web import fetch_freesound
        audio, meta = fetch_freesound(
            "https://freesound.org/people/cdonahueucsd/sounds/337131/"
        )

    Args:
        id_or_url: A FreeSound URL or numeric sound ID. See :func:`url_to_id`
            for the recognized URL forms.
        preview_tag: Which preview to fetch (default high-quality OGG). Pass
            ``None`` to fetch the original uncompressed file (OAuth2 required).
        reauthenticate: Force a re-prompt of API credentials.

    Returns:
        ``(audio, metadata)`` — ``audio`` is decoded via
        :meth:`pyquist.Audio.from_file` and ``metadata`` is the raw JSON dict
        returned by the FreeSound API.
    """
    sound_id = url_to_id(id_or_url)
    audio_bytes, metadata = fetch_audio_bytes(
        sound_id, preview_tag=preview_tag, reauthenticate=reauthenticate
    )
    return Audio.from_file(BytesIO(audio_bytes)), metadata


if __name__ == "__main__":
    import sys

    from ..device import play

    audio, metadata = fetch_freesound(sys.argv[1])
    print(json.dumps(metadata, indent=2))
    play(audio)
