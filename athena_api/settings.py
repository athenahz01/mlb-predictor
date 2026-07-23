from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Athena Baseball"
    environment: str = "development"
    api_prefix: str = "/api/v1"
    database_url: str = Field(
        default=f"sqlite:///{(ROOT / 'data' / 'athena_dev.db').as_posix()}"
    )
    cors_origins: str = "http://localhost:3000"
    supabase_url: str | None = None
    supabase_anon_key: str | None = None
    supabase_jwt_secret: str | None = None
    auth_required: bool = False
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    ai_provider: str = "auto"
    ai_model: str | None = None
    ai_timeout_seconds: float = 15.0
    ai_max_output_tokens: int = 800
    ai_max_tool_calls: int = 4
    prediction_cache_seconds: int = 30
    legacy_ledger_path: Path = ROOT / "ledger" / "ledger.json"
    dashboard_ledger_path: Path = ROOT / "dashboard" / "ledger.json"
    promoted_snapshot_pointer: Path = ROOT / "data" / "processed" / "promoted.json"

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
