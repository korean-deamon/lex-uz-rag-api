# The /v1 API, its index builder and the evaluation harness in one image.
#
# Built so that `docker compose up` is the only command a reviewer has to run:
# the entrypoint waits for Qdrant, builds the index if it is missing, and then
# serves. See docker/entrypoint.py.

FROM python:3.11-slim

# PYTHONUTF8: the corpus is full of oʻ / gʻ / ʼ and the logs print them.
# HF_HOME: the embedding model (~1.1 GB) is cached here, on a volume, so it is
# downloaded once rather than on every container rebuild.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUTF8=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models

WORKDIR /app

# torch comes from the CPU index on purpose and before everything else: pulled
# as a plain dependency of sentence-transformers it brings several GB of CUDA
# wheels that nothing here will ever use.
RUN pip install --no-cache-dir \
        torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ backend/
COPY ingest/ ingest/
COPY scripts/ scripts/
COPY tests/ tests/
COPY docker/ docker/
COPY data/corpus/ data/corpus/
COPY data/eval/ data/eval/
COPY data/section_titles.json data/

EXPOSE 8000

ENTRYPOINT ["python", "docker/entrypoint.py"]
