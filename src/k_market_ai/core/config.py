from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="KMARKET_AI_",
        extra="ignore",
    )

    app_name: str = "K-Market-Navigator AI"
    environment: Literal["local", "test", "production"] = "local"
    allowed_hosts: list[str] = Field(default_factory=lambda: ["localhost", "127.0.0.1"])
    database_url: SecretStr | None = None
    service_token: SecretStr | None = None
    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    news_model: str = "gpt-5-nano"
    news_prompt_version: str = "news-intelligence-v2"
    term_prompt_version: str = "financial-term-v1"
    translation_model: str = "gpt-5-nano"
    title_translation_prompt_version: str = "financial-title-translation-v5"
    news_narrative_prompt_version: str = "news-narrative-v12"
    disclosure_section_prompt_version: str = "disclosure-section-translation-v5"
    title_translation_timeout_seconds: float = Field(default=90.0, ge=10.0, le=180.0)
    news_narrative_timeout_seconds: float = Field(default=180.0, ge=10.0, le=180.0)
    disclosure_section_timeout_seconds: float = Field(default=90.0, ge=10.0, le=180.0)
    filing_summary_prompt_version: str = "filing-summary-v2"
    agent_model: str = "gpt-5-nano"
    agent_prompt_version: str = "market-agent-v1"
    tax_document_prompt_version: str = "kmarket-tax-ocr-e2e-v1"
    peer_model: str = "gpt-5-nano"
    peer_prompt_version: str = "global-peer-narrative-v2"
    model_bundle_root: Path | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "KMARKET_AI_MODEL_BUNDLE_ROOT",
            "KMARKET_AI_HANA_PROJECT_ROOT",
        ),
    )
    model_bundle_commit: str = Field(
        default="ab82ccc51cb096872f9a110a85c027a4158a147f",
        validation_alias=AliasChoices(
            "KMARKET_AI_MODEL_BUNDLE_COMMIT",
            "KMARKET_AI_HANA_EXPECTED_COMMIT",
        ),
    )

    @property
    def docs_enabled(self) -> bool:
        return self.environment != "production"

    @property
    def api_rag_configured(self) -> bool:
        return all((self.database_url, self.service_token, self.openai_api_key))

    @property
    def openai_configured(self) -> bool:
        return all((self.service_token, self.openai_api_key))

    @property
    def news_configured(self) -> bool:
        return self.openai_configured and self.model_bundle_root is not None


@lru_cache
def get_settings() -> Settings:
    return Settings()
