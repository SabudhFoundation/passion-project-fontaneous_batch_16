#!/usr/bin/env sh
set -eu

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_DIR}"

if ! command -v fontforge >/dev/null 2>&1 || ! command -v tesseract >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y \
        fontforge \
        python3-fontforge \
        tesseract-ocr \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender1 \
        python3-pip \
        python3-venv
fi

rm -rf .venv
python3 -m venv .venv
. .venv/bin/activate

pip install --upgrade pip
grep -vE '^(easyocr|torch|torchvision)==' require.txt > /tmp/fontaneous_require_linux.txt
pip install -r /tmp/fontaneous_require_linux.txt
pip install easyocr==1.7.2 --no-deps
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 torchvision==0.20.1
