FROM python:3.11-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    CLEAN_SHORE_SERVER=1 CLEAN_SHORE_MODELS_DIR=/models CLEAN_SHORE_DATA_DIR=/data \
    HOME=/data/home XDG_CACHE_HOME=/data/cache HF_HOME=/data/huggingface \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 YOLO_AUTOINSTALL=false YOLO_CONFIG_DIR=/data/yolo \
    MPLCONFIGDIR=/data/matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt ./requirements.txt
# CPU wheels keep CUDA libraries out of the image; existing app pins are preserved.
RUN python -m pip install --upgrade pip \
    && python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install -r requirements.txt \
    && python -m pip check
RUN groupadd --gid 10001 shore && useradd --uid 10001 --gid shore --no-create-home shore \
    && mkdir -p /data /models && chown shore:shore /data
COPY *.py *.svg *.css ./
COPY .streamlit/ ./.streamlit/
COPY photo_component/ ./photo_component/
COPY licenses/ ./licenses/
COPY THIRD_PARTY_NOTICES.txt ./
USER shore
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.getenv('PORT','8501')+'/_stcore/health',timeout=3).read()"
CMD ["python", "serve.py"]
