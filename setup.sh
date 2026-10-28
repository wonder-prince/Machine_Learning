#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

python_bin="${PYTHON_BIN:-python3.11}"
if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "找不到 $python_bin。请先安装 Python 3.11，或用 PYTHON_BIN 指定解释器。" >&2
  exit 1
fi

if [[ ! -x .venv/bin/python ]]; then
  "$python_bin" -m venv .venv
fi

if command -v uv >/dev/null 2>&1; then
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$project_dir/.cache/uv}"
  uv pip install --python .venv/bin/python -r requirements.txt
  uv pip check --python .venv/bin/python
else
  if ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
    .venv/bin/python -m ensurepip --upgrade
  fi
  .venv/bin/python -m pip install -r requirements.txt
  .venv/bin/python -m pip check
fi

.venv/bin/python src/download_data.py
echo "环境已就绪。快速检查：.venv/bin/python src/train.py --quick --cv-folds 2"
