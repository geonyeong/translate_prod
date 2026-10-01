# shellcheck shell=bash
# Shared venv bootstrap for run.sh / gui.sh. Source this file, then call ensure_venv.
#
# venv/bin/activate hardcodes the path the venv was created at, so it breaks after the
# project folder is moved. Callers use "$PY" (the venv interpreter) directly instead.

ensure_venv() {
  local root="$1"
  local force_setup="${2:-0}"
  PY="$root/venv/bin/python"

  if [[ ! -x "$PY" ]] || ! "$PY" -c "import sys" >/dev/null 2>&1; then
    if [[ -d "$root/venv" ]]; then
      echo "venv가 손상되어 다시 생성합니다 (python3.11)..."
      rm -rf "$root/venv"
    else
      echo "venv가 없어 생성합니다 (python3.11)..."
    fi
    if ! command -v python3.11 >/dev/null 2>&1; then
      echo "오류: python3.11이 없습니다. 'brew install python@3.11'로 설치하세요." >&2
      exit 1
    fi
    python3.11 -m venv "$root/venv"
    force_setup=1
  fi

  if [[ "$force_setup" -eq 1 ]] || ! "$PY" -c "import mlx_qwen3_asr" >/dev/null 2>&1; then
    echo "패키지 설치 중..."
    "$PY" -m pip install --upgrade pip
    "$PY" -m pip install -r "$root/requirements.txt"
  fi
}

# Install individual packages only when their import fails (for venvs created before
# a dependency was added to requirements.txt).
ensure_modules() {
  local import_names="$1"
  shift
  if ! "$PY" -c "import $import_names" >/dev/null 2>&1; then
    echo "추가 패키지 설치 중: $*"
    "$PY" -m pip install "$@"
  fi
}
