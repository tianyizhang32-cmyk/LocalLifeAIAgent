from __future__ import annotations

import os
from dataclasses import dataclass
from getpass import getpass
from typing import Optional


@dataclass(frozen=True)
class Settings:
    """
    Centralized config. In production, prefer environment variables or a secret manager.
    In notebooks/local dev, call Settings.load(interactive=True) to prompt for keys.
    """
    openai_api_key: str
    google_places_api_key: str
    openai_model: str = "gpt-5.2"

    @staticmethod
    def load(interactive: bool = False, openai_model: Optional[str] = None) -> "Settings":
        oa = os.environ.get("OPENAI_API_KEY")
        gp = os.environ.get("GOOGLE_PLACES_API_KEY")

        if interactive:
            if not oa:
                oa = getpass("OpenAI API Key (will not echo): ").strip()
            if not gp:
                gp = getpass("Google Places API Key (will not echo): ").strip()

        if not oa:
            raise ValueError("Missing OPENAI_API_KEY (set env var or use Settings.load(interactive=True))")
        if not gp:
            raise ValueError("Missing GOOGLE_PLACES_API_KEY (set env var or use Settings.load(interactive=True))")

        return Settings(
            openai_api_key=oa,
            google_places_api_key=gp,
            openai_model=openai_model or os.environ.get("OPENAI_MODEL", "gpt-5.2"),
        )
