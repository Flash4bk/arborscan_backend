FROM python:3.11-slim

# Установим системные библиотеки, нужные для OpenCV в Linux-контейнере Railway
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgtk2.0-dev \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV TREE_MODEL_PATH=models/tree_model.pt
ENV STICK_MODEL_PATH=models/stick_model.pt
ENV CLASSIFIER_PATH=models/classifier.pth

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
