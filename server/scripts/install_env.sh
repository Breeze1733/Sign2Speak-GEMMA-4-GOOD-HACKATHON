#!/usr/bin/env bash
# Idempotent bootstrap for the Sign2Speak cloud node (Ubuntu 22.04 x86_64).
# Re-runnable: every step checks before acting.
set -euo pipefail

PIP_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"
ROOT=/opt/sign2speak
UV="$HOME/.local/bin/uv"

echo "==> base packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git curl wget ca-certificates rsync unzip htop jq tmux \
  build-essential iproute2 sysstat aria2 python3-pip

echo "==> timezone / clock"
timedatectl set-timezone Asia/Shanghai
systemctl enable --now chrony

echo "==> uv"
if [[ ! -x "$UV" ]]; then
  python3 -m pip install --user -q -i "$PIP_MIRROR" uv
fi
"$UV" --version

echo "==> layout"
mkdir -p "$ROOT"/{app,models,logs,scripts,results,cache}

echo "==> venv + deps"
if [[ ! -x "$ROOT/venv/bin/python" ]]; then
  # Use the system interpreter: the litert-lm-api wheel is py3-none, so any
  # Python 3 works and we avoid pulling python-build-standalone from GitHub.
  "$UV" venv --python /usr/bin/python3.10 "$ROOT/venv"
fi
UV_INDEX_URL="$PIP_MIRROR" "$UV" pip install --python "$ROOT/venv/bin/python" \
  -r "$ROOT/app/p1/requirements.txt"
"$UV" pip freeze --python "$ROOT/venv/bin/python" > "$ROOT/requirements.lock"

echo "==> done"
"$ROOT/venv/bin/python" - <<'PY'
import litert_lm
from importlib.metadata import version
print("litert-lm-api", version("litert-lm-api"))
print("litert_lm module:", litert_lm.__file__)
PY
