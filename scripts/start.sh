#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# This project never uses the user's proxy, including for loopback inference.
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY ftp_proxy FTP_PROXY
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
exec "${LABELER_PYTHON:-.venv/bin/python}" -m uvicorn app.main:app --host "${LABELER_HOST:-127.0.0.1}" --port "${LABELER_PORT:-7860}" --workers 1 --no-access-log
