# syntax=docker/dockerfile:1.7

# ============================================================
# Stage 1: Builder
# ============================================================
FROM python:3.10-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /usr/local/bin/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /build

COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
        --prefix /install \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        -r requirements.txt


# ============================================================
# Stage 2: Runtime
# ============================================================
FROM python:3.10-slim AS runtime

LABEL org.opencontainers.image.title="fashion-recommender" \
      org.opencontainers.image.description="Fashion product similarity API (local k8s)"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NAME=ProductSimilarityApp \
    HOME=/home/appuser \
    HF_HOME=/home/appuser/.cache/huggingface

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

COPY app.py config.py sim_search.py utils.py /app/
COPY static /app/static

COPY image_hnsw_all.bin text_hnsw_all.bin /app/
COPY product_metadata_all.json /app/
COPY marketing_sample_for_amazon_com-amazon_fashion_products__20200201_20200430__30k_data.ldjson /app/

RUN useradd --create-home --home-dir /home/appuser \
        --shell /bin/false appuser && \
    mkdir -p /home/appuser/.cache/huggingface && \
    chown -R appuser:appuser /app /home/appuser

USER appuser

EXPOSE 9000

CMD ["uvicorn", "app:app", \
     "--host", "0.0.0.0", \
    "--port", "9000", \
     "--workers", "1", \
     "--timeout-graceful-shutdown", "10"]