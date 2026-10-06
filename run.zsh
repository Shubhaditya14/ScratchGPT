#!/usr/bin/env zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
PROJECT_DIR="${SCRIPT_DIR}"
VENV_DIR="${PROJECT_DIR}/.venv"

if ! command -v python3 >/dev/null 2>&1; then
  print "python3 not found in PATH. Please install Python 3."
  exit 1
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

python3 -m pip install --upgrade pip >/dev/null
python3 -m pip install -r "${PROJECT_DIR}/requirements.txt"

python3 "${PROJECT_DIR}/src/scripts/tui.py" "$@"
