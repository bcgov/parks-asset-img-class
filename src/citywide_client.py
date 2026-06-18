"""Client for the PSD CityWide Asset Management REST API.

The CityWide API is used to download the raw infrastructure asset photos and
metadata that feed the capstone image-classification pipeline. Credentials are
read from ``.env`` or the shell:

- ``CITYWIDE_API_KEY``
- ``CITYWIDE_DB``
- ``CITYWIDE_USER``
- optional ``CITYWIDE_API_URL``
"""

from __future__ import annotations

import collections
import os
import random
import re
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

DEFAULT_CITYWIDE_API_URL = "https://v4.citywidesolutions.com/v4_server/external/v1"


class _RateLimiter:
    """Sliding-window rate limiter for CityWide's hourly request cap."""

    def __init__(self, max_calls: int, window: float = 3600.0) -> None:
        self.max_calls = max_calls
        self.window = window
        self._calls: collections.deque[float] = collections.deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                while self._calls and now - self._calls[0] > self.window:
                    self._calls.popleft()
                if len(self._calls) < self.max_calls:
                    self._calls.append(now)
                    return
                wait = self.window - (now - self._calls[0]) + 0.5
            time.sleep(max(0.1, wait))


class CitywideClient:
    """Small authenticated client with token refresh, pagination, and retries."""

    def __init__(
        self,
        api_key: str | None = None,
        client_db: str | None = None,
        username: str | None = None,
        url: str | None = None,
        timeout: float = 120.0,
        max_calls_per_hour: int = 900,
    ) -> None:
        self.api_key = api_key or os.getenv("CITYWIDE_API_KEY")
        self.client_db = client_db or os.getenv("CITYWIDE_DB")
        self.username = username or os.getenv("CITYWIDE_USER")
        self.url = (url or os.getenv("CITYWIDE_API_URL") or DEFAULT_CITYWIDE_API_URL).rstrip("/")

        missing = [
            name
            for name, value in [
                ("CITYWIDE_API_KEY", self.api_key),
                ("CITYWIDE_DB", self.client_db),
                ("CITYWIDE_USER", self.username),
            ]
            if not value
        ]
        if missing:
            raise RuntimeError(
                "Missing CityWide credential(s): "
                + ", ".join(missing)
                + ". Add them to .env or export them in the shell."
            )

        self._http = httpx.Client(timeout=timeout)
        self._token: str | None = None
        self._expires_at = 0.0
        self._limiter = _RateLimiter(max_calls=max_calls_per_hour, window=3600.0)

    def _authenticate(self) -> None:
        response = self._http.post(
            f"{self.url}/authenticate",
            json={
                "api_key": self.api_key,
                "client_db": self.client_db,
                "username": self.username,
            },
        )
        response.raise_for_status()
        body = response.json()
        self._token = body["access_token"]
        self._expires_at = time.time() + int(body.get("expires_in", 3600)) - 60

    def _ensure_token(self) -> str:
        if self._token is None or time.time() >= self._expires_at:
            self._authenticate()
        assert self._token is not None
        return self._token

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._ensure_token()}",
            "Accept": "application/json",
        }

    def _request_with_retry(
        self,
        url: str,
        headers: dict[str, str],
        params: dict[str, Any] | None = None,
        *,
        max_attempts: int = 8,
        max_backoff: float = 300.0,
    ) -> httpx.Response:
        """GET with exponential backoff on rate limits, 5xx, and timeouts."""
        attempt = 0
        while True:
            attempt += 1
            self._limiter.acquire()
            try:
                response = self._http.get(
                    url,
                    params=params,
                    headers=headers,
                    follow_redirects=True,
                )
            except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.RemoteProtocolError):
                if attempt >= max_attempts:
                    raise
                wait = min(max_backoff, 2**attempt) + random.random()
                time.sleep(wait)
                continue

            if response.status_code == 429 or response.status_code >= 500:
                if attempt >= max_attempts:
                    return response
                retry_after = response.headers.get("Retry-After")
                try:
                    wait = float(retry_after) if retry_after else min(max_backoff, 2**attempt)
                except ValueError:
                    wait = min(max_backoff, 2**attempt)
                wait = min(wait, max_backoff) + random.random()
                print(
                    f"    [{response.status_code}] sleeping {wait:.0f}s "
                    f"(attempt {attempt}/{max_attempts})",
                    flush=True,
                )
                time.sleep(wait)
                continue

            return response

    def get(self, path: str, params: dict[str, Any] | None = None) -> httpx.Response:
        url = f"{self.url}{path}" if path.startswith("/") else f"{self.url}/{path}"
        return self._request_with_retry(url, self._headers(), params=params)

    def get_binary(self, path: str) -> httpx.Response:
        url = f"{self.url}{path}" if path.startswith("/") else f"{self.url}/{path}"
        return self._request_with_retry(
            url,
            {"Authorization": f"Bearer {self._ensure_token()}"},
        )

    @staticmethod
    def _extract_cursor(link: str | None) -> str | None:
        if not link:
            return None
        match = re.search(r"\$cursor=([^&>]+)", link)
        return match.group(1) if match else None

    def list_all(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        limit: int = 50,
        progress: bool = True,
    ) -> Iterator[dict[str, Any]]:
        """Yield every record from a cursor-paginated CityWide list endpoint."""
        base_params = dict(params or {})
        base_params["$limit"] = limit

        response = self.get(path, params=base_params)
        response.raise_for_status()
        batch = response.json()
        yield from batch

        total = int(response.headers.get("X-Total", "0"))
        cursor = self._extract_cursor(response.headers.get("Link"))
        if progress:
            print(f"    listing {path}: {len(batch)}/{total}", flush=True)
        if not cursor or not total or len(batch) >= total:
            return

        seen = len(batch)
        page = 2
        while seen < total:
            response = self.get(path, params={**base_params, "$cursor": cursor, "$page": page})
            if response.status_code != 200:
                break
            batch = response.json()
            if not batch:
                break
            yield from batch
            seen += len(batch)
            if progress and (page % 5 == 0 or seen >= total):
                print(f"    listing {path}: {seen}/{total}", flush=True)
            page += 1

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "CitywideClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
