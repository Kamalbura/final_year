from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import requests
from requests import RequestException

DEFAULT_THINGSPEAK_URL = "https://api.thingspeak.com/update"


@dataclass(frozen=True)
class ThingSpeakPublishResult:
    city: str
    status: str
    entry_id: int | None
    success: bool
    message: str


class ThingSpeakClient:
    def __init__(self, timeout: int = 45, retries: int = 3, base_url: str = DEFAULT_THINGSPEAK_URL) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if retries <= 0:
            raise ValueError("retries must be positive")
        self._timeout = timeout
        self._retries = retries
        self._base_url = base_url.rstrip("/")
        self._session = requests.Session()

    def close(self) -> None:
        self._session.close()

    def publish(self, write_key: str, fields: Mapping[str, object], status: str) -> ThingSpeakPublishResult:
        if not write_key:
            raise ValueError("write_key is required")

        payload = {"api_key": write_key, "status": status}
        for field_name, field_value in fields.items():
            payload[field_name] = field_value

        last_error: Exception | None = None
        for attempt in range(1, self._retries + 1):
            try:
                response = self._session.post(self._base_url, data=payload, timeout=self._timeout)
                response.raise_for_status()
                text = response.text.strip()
                entry_id = int(text) if text.isdigit() and int(text) > 0 else None
                success = entry_id is not None
                message = "published" if success else f"ThingSpeak rejected update: {text or 'empty response'}"
                return ThingSpeakPublishResult(city="", status=status, entry_id=entry_id, success=success, message=message)
            except RequestException as exc:
                last_error = exc
                if attempt == self._retries:
                    break

        return ThingSpeakPublishResult(city="", status=status, entry_id=None, success=False, message=str(last_error))
