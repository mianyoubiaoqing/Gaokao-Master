"""Persistent local settings for Gaokao-Master WebUI."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_SETTINGS_PATH = Path(".gaokao_master") / "settings.json"


@dataclass(frozen=True)
class AppSettings:
    """User-editable settings persisted from the WebUI."""

    kb_root: str = "Gaokao_KB"
    embedding_model_name: str = "local-hash"
    use_llm: bool = False
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    openai_api_key: str = ""
    openai_temperature: float = 0.2
    tavily_api_key: str = ""
    use_ocr: bool = False
    ocr_base_url: str = "https://api.openai.com/v1"
    ocr_model: str = "gpt-4o-mini"
    ocr_api_key: str = ""
    ocr_temperature: float = 0.0
    ocr_dpi: int = 180
    ocr_max_pages: int = 20
    ocr_extract_media: bool = True


def load_app_settings(path: str | Path = DEFAULT_SETTINGS_PATH) -> AppSettings:
    """Load WebUI settings from disk, returning defaults when missing."""

    settings_path = Path(path)
    if not settings_path.exists():
        return AppSettings()

    try:
        data = json.loads(settings_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return AppSettings()

    defaults = asdict(AppSettings())
    merged: dict[str, Any] = {**defaults, **data}
    return AppSettings(**{key: merged[key] for key in defaults})


def save_app_settings(
    settings: AppSettings,
    path: str | Path = DEFAULT_SETTINGS_PATH,
) -> Path:
    """Persist WebUI settings to disk as UTF-8 JSON."""

    settings_path = Path(path)
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(
        json.dumps(asdict(settings), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return settings_path
