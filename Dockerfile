FROM python:3.14-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN groupadd --system --gid 10001 kmarket \
    && useradd --system --uid 10001 --gid kmarket --create-home kmarket \
    && apt-get update \
    && apt-get install --no-install-recommends -y poppler-utils tesseract-ocr tesseract-ocr-eng tesseract-ocr-kor \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv==0.11.12

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

COPY src src
COPY README.md ./
RUN uv sync --locked --no-dev

USER 10001:10001
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
CMD ["uvicorn", "k_market_ai.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
