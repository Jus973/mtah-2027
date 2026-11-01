# Reproduces the submission from a mounted bundle (PLAN.md M4). CPU only, pinned deps.
#   docker build -t mtah .
#   docker run -t -v /path/to/mtah2027:/data -v "$PWD/out":/out mtah   [-e TEAM="<name>"]
# -t is required: the real bundle opens only at a terminal (src/mtah/registry.py gate). The run
# parses the ATD, writes out/<team>.csv (default: the team in configs/base.yaml) and checks its
# sha256 against the committed submissions/final/<team>.sha256.
FROM python:3.11.14-slim
RUN apt-get update && apt-get install -y --no-install-recommends make libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /repo
COPY requirements.txt pyproject.toml ./
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python -m pip install --no-cache-dir -e . --no-deps
ENV MTAH_DATA_DIR=/data MTAH_CACHE_DIR=/tmp/mtah_cache PYTHONHASHSEED=0 PY=python
CMD ["sh", "scripts/docker_run.sh"]
