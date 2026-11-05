from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelEndpoint(BaseModel):
    """An OpenAI-compatible model service."""

    model_config = ConfigDict(frozen=True)

    base_url: HttpUrl
    model: str = Field(min_length=1)
    # Local model servers often need no key.
    api_key: SecretStr | None = None
    # The model's context window, which sizes what one request may hold.
    context_tokens: int = Field(default=128_000, ge=8_000)


class Observability(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    # `metadata` records each step without code, prompts or model output.
    content: Literal["metadata", "full"] = "metadata"
    exporter: Literal["langfuse", "langsmith", "phoenix", "otlp"] = "otlp"
    endpoint: HttpUrl | None = None
    headers: dict[str, SecretStr] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _endpoint_when_enabled(self) -> Self:
        if self.enabled and self.endpoint is None:
            raise ValueError("observability is enabled but has no endpoint")
        return self


class Settings(BaseSettings):
    """Service settings, read once from `REVIEW_AGENTS_*` environment variables.

    Secrets come only from the environment. Nested values use `__`, for example
    `REVIEW_AGENTS_FINDER_MODEL__BASE_URL`; `headers` takes a JSON object.
    """

    model_config = SettingsConfigDict(
        env_prefix="REVIEW_AGENTS_", env_nested_delimiter="__", frozen=True
    )

    data_dir: Path = Path("data")
    host: str = "127.0.0.1"
    port: int = Field(default=8080, ge=1, le=65535)
    worker_poll_seconds: float = Field(default=2.0, gt=0)

    # Required once reviews run (#9) and requests are authenticated (#13).
    finder_model: ModelEndpoint | None = None
    verifier_model: ModelEndpoint | None = None
    shared_secret: SecretStr | None = Field(default=None, min_length=32)

    observability: Observability = Field(default_factory=Observability)

    # Safety nets against runaway cost, not design limits (decisions D-009, D-016); to be
    # tuned from measurements on repositories of different sizes (#17). A review's
    # budget counts cached input at a tenth; a repository can only lower it.
    tokens_per_review: int = Field(default=1_500_000, gt=0)
    # A safety net against runaway cost when summarising a repository's modules, not a
    # design limit; to be tuned from measurements on repositories of different sizes (#17).
    summary_tokens_per_build: int = Field(default=3_000_000, gt=0)

    # Model calls one review makes at a time (decision D-022), so a large PR cannot set off
    # the model service's rate limit; local models may want 1.
    model_concurrency: int = Field(default=5, ge=1)
    # Of a review's time, what is kept after the dimension phase for merging, verifying and
    # writing the findings (decisions D-022, D-028); to be tuned from measurements (#17).
    wrap_up_seconds: float = Field(default=600.0, ge=0)

    # Preload the callers, tests and types of changed code into the PR layer. Off: the
    # agents look them up with their tools (decision D-015); kept for comparison in #17
    # and set by whoever runs the service, never by a reviewed repository (D-017).
    surrounding_code: bool = False

    @property
    def db_path(self) -> Path:
        return self.data_dir / "review-agents.sqlite3"

    @property
    def repos_dir(self) -> Path:
        """Git mirrors of reviewed repositories, on the same volume as the database."""
        return self.data_dir / "repos"
