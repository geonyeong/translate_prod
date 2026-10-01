#!/usr/bin/env bash
# Launch the desktop GUI.
# Usage:
#   ./gui.sh            # prepare venv if needed, then open the GUI
#   ./gui.sh --setup    # reinstall packages first
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

DO_SETUP=0
case "${1:-}" in
  --setup|-s) DO_SETUP=1 ;;
  "") ;;
  -h|--help)
    echo "Usage: ./gui.sh [--setup]"
    exit 0
    ;;
  *)
    echo "알 수 없는 옵션: $1" >&2
    exit 1
    ;;
esac

# shellcheck source=lib/venv.sh
source "$ROOT/lib/venv.sh"
ensure_venv "$ROOT" "$DO_SETUP"
ensure_modules "PySide6.QtWidgets" PySide6-Essentials
ensure_modules "deep_translator, requests" deep-translator requests

exec "$PY" "$ROOT/gui.py"
