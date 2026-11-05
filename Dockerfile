FROM ghcr.io/astral-sh/uv:0.11.32@sha256:df4cae8f3a96d175e2e5f992e597550000edbe78fdc2594d5cd8de1a217f504c AS uv

FROM python:3.12-slim@sha256:aa943ce411423c7582288f18707ff66eddcbaeb510502d747a72aeb34f30d159 AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project
COPY src ./src
RUN uv sync --locked --no-dev --no-editable
# Token counting needs tiktoken's encoding file; fetched here (tiktoken checks its hash)
# so the running service never downloads it (decision D-013).
# DL3064 flags "TOKEN" in the name; this is tiktoken's cache directory, not a secret.
# hadolint ignore=DL3064
ENV TIKTOKEN_CACHE_DIR=/app/tiktoken
RUN /app/.venv/bin/python -c "from review_agents.context.tokens import ENCODING; import tiktoken; tiktoken.get_encoding(ENCODING)"

FROM python:3.12-slim@sha256:aa943ce411423c7582288f18707ff66eddcbaeb510502d747a72aeb34f30d159
# The review reads repositories through git. Package versions follow the pinned base image's
# Debian release; pinning each one would break the build on every security update. apt is
# named too so a security fix released after the base image (CVE-2026-109518) is installed.
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install --yes --no-install-recommends apt git \
    && rm -rf /var/lib/apt/lists/*
RUN useradd --system --uid 10001 --home-dir /nonexistent --shell /usr/sbin/nologin review \
    && mkdir /data && chown review /data
COPY --from=build /app/.venv /app/.venv
COPY --from=build /app/tiktoken /app/tiktoken
# The baked encoding (see the build stage); DL3064 misreads the name as a secret.
# hadolint ignore=DL3064
ENV TIKTOKEN_CACHE_DIR=/app/tiktoken
ENV PATH="/app/.venv/bin:$PATH" \
    REVIEW_AGENTS_DATA_DIR=/data \
    REVIEW_AGENTS_HOST=0.0.0.0 \
    REVIEW_AGENTS_PORT=8080
USER 10001
VOLUME ["/data"]
EXPOSE 8080
CMD ["python", "-m", "review_agents"]
