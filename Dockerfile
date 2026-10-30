# Reproduces the submission from a mounted bundle: docker run -v /path/to/mtah2027:/data mtah
FROM python:3.11.14-slim
RUN apt-get update && apt-get install -y --no-install-recommends make libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /repo
COPY requirements.txt pyproject.toml ./
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python -m pip install --no-cache-dir -e . --no-deps
ENV MTAH_DATA_DIR=/data MTAH_CACHE_DIR=/tmp/mtah_cache PYTHONHASHSEED=0 PY=python
CMD ["make", "PY=python", "cache", "submission"]
