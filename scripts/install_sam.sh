#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY ftp_proxy FTP_PROXY
.venv/bin/pip install -i https://mirrors.aliyun.com/pypi/simple 'torch==2.5.1' 'torchvision==0.20.1' 'hydra-core==1.3.2' 'iopath==0.1.10' 'tqdm==4.67.1' 'wheel==0.45.1'
SAM2_BUILD_CUDA=0 .venv/bin/pip install --no-build-isolation --no-deps ./third_party/sam2
python3 scripts/download_sam.py
