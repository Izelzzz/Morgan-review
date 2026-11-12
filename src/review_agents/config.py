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
    # Replies stream, and a stream with no data at all (heartbeats count) for
    # `idle_seconds` is stalled; turned off, a whole reply must arrive within two minutes,
    # for services that stream tool calls badly (decisions D-029, D-030).
    stream: bool = True
    idle_seconds: float = Field(default=300.0, gt=0)
    # Retries of a rate limit, server error, stall, broken connection or empty reply.
    max_retries: int = Field(default=4, ge=0)


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

    # Who writes the findings and the walkthrough (D-038): an agent per dimension, one
    # general agent, or one general agent that writes the walkthrough first and carries
    # on from it. Verification, merging and the report are the same for all three.
    strategy: Literal["dimensions", "general", "walkthrough_first"] = "dimensions"

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
    # The dimension agents hand in their findings at the latest this long before the
    # review's end, so the slowest one's findings can still be verified; the others are
    # verified as soon as they hand in (decision D-031). To be tuned from measurements (#17).
    after_submit_seconds: float = Field(default=600.0, ge=0)
    # An agent's tool conversation stays inside the model's window (decision D-030): one
    # tool result keeps at most this many tokens, head and tail; when the conversation
    # would not fit, older results are cut to a head and tail this long each, keeping
    # room for the reply.
    tool_result_tokens: int = Field(default=10_000, ge=1)
    shortened_result_tokens: int = Field(default=500, ge=1)
    output_reserve_tokens: int = Field(default=4_000, ge=0)

    # Preload the callers, tests and types of changed code into the PR layer. Off: the
    # agents look them up with their tools (decision D-015); kept for comparison in #17
    # and set by whoever runs the service, never by a reviewed repository (D-017).
    surrounding_code: bool = False

    @model_validator(mode="after")
    def _verifier_fits_the_context(self) -> Self:
        # A verifier reads the same context pack as the dimension agents, which is sized
        # for the finder model's window (decision D-032).
        finder, verifier = self.finder_model, self.verifier_model
        if finder and verifier and verifier.context_tokens < finder.context_tokens:
            raise ValueError(
                f"the verifier model's window ({verifier.context_tokens} tokens) is smaller"
                f" than the finder model's ({finder.context_tokens}); a verifier reads the"
                " same context, so give it a model with at least as large a window"
            )
        return self

    @property
    def db_path(self) -> Path:
        return self.data_dir / "review-agents.sqlite3"

    @property
    def repos_dir(self) -> Path:
        """Git mirrors of reviewed repositories, on the same volume as the database."""
        return self.data_dir / "repos"
