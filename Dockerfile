# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.12

# Base: interpreter and project sources
FROM python:${PYTHON_VERSION}-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
COPY constraints.txt pyproject.toml README.md LICENSE ./
COPY src ./src

# Test: dev tools + test suite (used by CI for lint and pytest)
FROM base AS test
COPY tests ./tests
RUN pip install -c constraints.txt -e ".[dev]"
CMD ["pytest", "--cov", "--cov-fail-under=90"]

# Runtime: minimal image that runs the stackxai CLI
# Build with --build-arg EXTRAS="[explain]" to include SHAP (slower, larger image).
FROM base AS runtime
ARG EXTRAS=""
RUN pip install -c constraints.txt ".${EXTRAS}" \
 && useradd --create-home --uid 10001 appuser \
 && mkdir /work && chown 10001:10001 /work
LABEL org.opencontainers.image.title="stackxai" \
      org.opencontainers.image.description="Reproducible stacking-ensemble evaluation with explainability" \
      org.opencontainers.image.licenses="MIT"
USER 10001:10001
WORKDIR /work
ENTRYPOINT ["stackxai"]
CMD ["--help"]
