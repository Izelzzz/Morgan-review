import pytest
from pydantic import ValidationError

from review_agents.config import Settings


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    for name in list(os.environ):
        if name.startswith("REVIEW_AGENTS_"):
            monkeypatch.delenv(name)


def test_models_secret_and_observability_come_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REVIEW_AGENTS_FINDER_MODEL__BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("REVIEW_AGENTS_FINDER_MODEL__MODEL", "small")
    monkeypatch.setenv("REVIEW_AGENTS_VERIFIER_MODEL__BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("REVIEW_AGENTS_VERIFIER_MODEL__MODEL", "strong")
    monkeypatch.setenv("REVIEW_AGENTS_VERIFIER_MODEL__API_KEY", "sk-verifier")
    monkeypatch.setenv("REVIEW_AGENTS_SHARED_SECRET", "s" * 32)
    monkeypatch.setenv("REVIEW_AGENTS_OBSERVABILITY__ENABLED", "true")
    monkeypatch.setenv("REVIEW_AGENTS_OBSERVABILITY__EXPORTER", "langfuse")
    monkeypatch.setenv(
        "REVIEW_AGENTS_OBSERVABILITY__ENDPOINT", "https://cloud.langfuse.com/api/public/otel"
    )
    monkeypatch.setenv("REVIEW_AGENTS_OBSERVABILITY__HEADERS", '{"Authorization": "Basic abc"}')

    settings = Settings()

    assert settings.finder_model is not None and settings.finder_model.api_key is None
    assert settings.finder_model.context_tokens == 128_000
    assert settings.verifier_model is not None
    assert settings.verifier_model.api_key is not None
    assert settings.verifier_model.api_key.get_secret_value() == "sk-verifier"
    assert settings.observability.headers["Authorization"].get_secret_value() == "Basic abc"


def test_secrets_never_appear_in_the_settings_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REVIEW_AGENTS_FINDER_MODEL__BASE_URL", "http://localhost/v1")
    monkeypatch.setenv("REVIEW_AGENTS_FINDER_MODEL__MODEL", "m")
    monkeypatch.setenv("REVIEW_AGENTS_FINDER_MODEL__API_KEY", "sk-finder-secret")
    monkeypatch.setenv("REVIEW_AGENTS_SHARED_SECRET", "hmac-secret-" + "x" * 32)
    monkeypatch.setenv("REVIEW_AGENTS_OBSERVABILITY__HEADERS", '{"Authorization": "Basic abc"}')

    text = repr(Settings())

    for secret in ("sk-finder-secret", "hmac-secret-", "Basic abc"):
        assert secret not in text


def test_a_short_shared_secret_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REVIEW_AGENTS_SHARED_SECRET", "short")
    with pytest.raises(ValidationError, match="shared_secret"):
        Settings()


def test_observability_needs_an_endpoint_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REVIEW_AGENTS_OBSERVABILITY__ENABLED", "true")
    with pytest.raises(ValidationError, match="no endpoint"):
        Settings()


def test_everything_optional_is_off_by_default() -> None:
    settings = Settings()
    assert settings.finder_model is None
    assert settings.shared_secret is None
    assert settings.observability.enabled is False
    assert settings.observability.content == "metadata"
    assert settings.surrounding_code is False


def test_the_review_token_budget_is_a_service_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    assert Settings().tokens_per_review == 1_500_000

    monkeypatch.setenv("REVIEW_AGENTS_TOKENS_PER_REVIEW", "800000")

    assert Settings().tokens_per_review == 800_000


def test_tool_results_have_the_decided_limits_by_default() -> None:
    settings = Settings()  # decision D-030

    assert (
        settings.tool_result_tokens,
        settings.shortened_result_tokens,
        settings.output_reserve_tokens,
    ) == (10_000, 500, 4_000)


def test_dimension_agents_submit_ten_minutes_before_the_end_by_default() -> None:
    assert Settings().after_submit_seconds == 600  # decision D-031: by minute 20 of 30


def endpoint(context_tokens: int, model: str) -> dict[str, object]:
    return {"base_url": "http://127.0.0.1:9/v1", "model": model, "context_tokens": context_tokens}


def test_a_verifier_model_with_a_smaller_window_is_refused() -> None:
    # The verifier reads the dimension agents' whole context pack (decision D-032).
    with pytest.raises(ValidationError, match="verifier model's window"):
        Settings(
            finder_model=endpoint(128_000, "finder"),  # type: ignore[arg-type]
            verifier_model=endpoint(32_000, "verifier"),  # type: ignore[arg-type]
        )

    same = Settings(
        finder_model=endpoint(128_000, "finder"),  # type: ignore[arg-type]
        verifier_model=endpoint(128_000, "verifier"),  # type: ignore[arg-type]
    )
    assert same.verifier_model is not None
