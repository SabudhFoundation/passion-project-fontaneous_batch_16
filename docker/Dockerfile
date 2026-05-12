FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:${PATH}" \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    TESSERACT_CMD=/usr/bin/tesseract

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    fontforge \
    python3-fontforge \
    tesseract-ocr \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY require.txt ./require.txt

COPY . .

EXPOSE 8501

CMD ["sh", "-lc", "if [ ! -x .venv/bin/python ]; then sh docker/setup_linux_venv.sh; fi && exec .venv/bin/streamlit run src/main.py --server.address=0.0.0.0 --server.port=8501"]
