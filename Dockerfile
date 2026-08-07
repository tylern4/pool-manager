# syntax=docker/dockerfile:1

FROM ghcr.io/astral-sh/uv:0.8-python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_EDITABLE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# git is required to fetch the htcondor-rest dependency from its git source.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first so their layer can be cached independently
# of the application source.
COPY pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project --extra rest --extra sfapi

# Install the project itself.
COPY pool_manager ./pool_manager
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra rest --extra sfapi

FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

COPY pool-manager.yaml ./pool-manager.yaml

EXPOSE 9090

ENTRYPOINT ["pool-manager"]
CMD ["run"]
