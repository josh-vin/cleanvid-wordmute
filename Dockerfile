FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    HF_HOME=/models \
    XDG_CACHE_HOME=/models/cache \
    NUMBA_CACHE_DIR=/tmp/numba

# CPU-only torch first, so stable-ts does not pull the CUDA build
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchaudio
RUN pip install "cleanvid==1.7.1" "stable-ts[fw]==2.19.1" pytest

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-deps .
COPY tests ./tests
COPY config.toml ./

ENTRYPOINT ["wordmute"]
