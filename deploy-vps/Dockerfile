FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/home/arborscan

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /tmp/requirements.txt

# CPU-only PyTorch: на нашем VPS нет NVIDIA GPU.
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision \
    && python -m pip install -r /tmp/requirements.txt

# COCO-модель нужна текущему server.py при реальном анализе.
# Загружаем ее ОДИН РАЗ на этапе сборки, а не во время запроса пользователя.
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" \
    && test -s /app/yolov8n.pt

RUN groupadd --gid 1000 arborscan \
    && useradd \
        --uid 1000 \
        --gid 1000 \
        --create-home \
        --shell /usr/sbin/nologin \
        arborscan

COPY --chown=arborscan:arborscan server.py config.py tree_dynamics.py /app/
COPY --chown=arborscan:arborscan services /app/services

RUN chown arborscan:arborscan /app/yolov8n.pt \
    && mkdir -p \
        /tmp/arborscan-models \
        /home/arborscan/.u2net \
        /home/arborscan/.cache \
        /home/arborscan/.config \
    && chown -R arborscan:arborscan \
        /tmp/arborscan-models \
        /home/arborscan

USER arborscan

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=8).read()" || exit 1

CMD ["python", "-m", "uvicorn", "server:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--proxy-headers", \
     "--forwarded-allow-ips=*"]
