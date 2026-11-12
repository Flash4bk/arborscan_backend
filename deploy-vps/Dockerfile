# Full rebuild is a distinct, gated candidate. Prefer runtime_candidate.py for
# a source-only security rollout on the exact recorded runtime.
ARG PYTHON_BASE_IMAGE
FROM ${PYTHON_BASE_IMAGE}
ARG PYTHON_BASE_IMAGE
ARG COCO_MODEL_SHA256

RUN python -c "import os,re,sys; assert re.fullmatch(r'python:3\.11\.16-slim-bookworm@sha256:[a-f0-9]{64}',os.environ['PYTHON_BASE_IMAGE']); assert sys.version_info[:3]==(3,11,16); assert re.fullmatch(r'[a-f0-9]{64}',os.environ['COCO_MODEL_SHA256'])"

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
COPY deploy-vps/requirements.runtime-v3.lock.txt /tmp/deploy-vps/requirements.runtime-v3.lock.txt

# CPU-only PyTorch: на нашем VPS нет NVIDIA GPU.
RUN python -m pip install --upgrade pip==26.2.1 setuptools==84.0.0 wheel==0.48.0 \
    && python -m pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        --no-deps torch==2.14.0+cpu torchvision==0.29.0+cpu \
    && python -m pip install --no-deps -r /tmp/requirements.txt \
    && python -m pip check \
    && python -c "import importlib.metadata as m,cv2,numpy,torch,torchvision,ultralytics,fastapi,PIL,rembg; assert torch.version.cuda is None; assert not any(d.metadata['Name'].lower()=='opencv-python-headless' for d in m.distributions())"

# COCO-модель нужна текущему server.py при реальном анализе.
# Загружаем ее ОДИН РАЗ на этапе сборки, а не во время запроса пользователя.
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" \
    && python -c "import os,hashlib,pathlib; assert hashlib.sha256(pathlib.Path('/app/yolov8n.pt').read_bytes()).hexdigest()==os.environ['COCO_MODEL_SHA256']"

RUN groupadd --gid 1000 arborscan \
    && useradd \
        --uid 1000 \
        --gid 1000 \
        --create-home \
        --shell /usr/sbin/nologin \
        arborscan

COPY --chown=arborscan:arborscan server.py auth_google_identity.py config.py tree_dynamics.py /app/
COPY --chown=arborscan:arborscan services /app/services
# Legacy owner feedback delegates to the private v4 moderation adapter.
# Importing this package does not start arborscan_v4.api or load its models.
COPY --chown=arborscan:arborscan arborscan_v4 /app/arborscan_v4

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
