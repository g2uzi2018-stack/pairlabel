#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY ftp_proxy FTP_PROXY
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
mkdir -p data logs models
printf '%s\n' '安装完成。运行 scripts/start.sh；模型下载运行 python3 scripts/download_models.py。'
