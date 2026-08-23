from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
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
    news_model: str = "gpt-5-mini"
    news_prompt_version: str = "news-intelligence-v1"
    term_prompt_version: str = "financial-term-v1"
    filing_summary_prompt_version: str = "filing-summary-v1"
    agent_model: str = "gpt-5-mini"
    agent_prompt_version: str = "market-agent-v1"
    tax_document_model: str = "gpt-5-mini"
    tax_document_prompt_version: str = "tax-document-v1"

    @property
    def docs_enabled(self) -> bool:
        return self.environment != "production"

    @property
    def api_rag_configured(self) -> bool:
        return all((self.database_url, self.service_token, self.openai_api_key))

    @property
    def news_configured(self) -> bool:
        return all((self.service_token, self.openai_api_key))


@lru_cache
def get_settings() -> Settings:
    return Settings()
