#!/usr/bin/env bash
# One-shot pipeline: video -> SRT (Qwen3-ASR) -> Korean SRT (NLLB / Google / Ollama)
# Usage:
#   ./run.sh video.mp4
#   ./run.sh                          # prompts for filename
#   ./run.sh video.mp4 --setup        # also (re)install packages
#   ./run.sh video.mp4 --engine google
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

ASR_MODEL="${ASR_MODEL:-Qwen/Qwen3-ASR-1.7B}"
OUTPUT_DIR="${OUTPUT_DIR:-./result}"
ENGINE="${TRANSLATE_ENGINE:-}"
DO_SETUP=0
VIDEO=""

usage() {
  echo "Usage: ./run.sh [video_file] [--engine nllb|google|ollama] [--setup]"
  echo "  video_file      동영상 경로 (생략 시 입력 요청)"
  echo "  --engine, -e    번역 엔진 (생략 시 선택 메뉴 표시)"
  echo "                    nllb   : 로컬 NLLB 모델 (오프라인)"
  echo "                    google : 구글 번역 (GOOGLE_API_KEY 있으면 공식 API, 없으면 무료 웹)"
  echo "                    ollama : 로컬 Ollama qwen3"
  echo "  --setup, -s     venv 생성 및 requirements 설치"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --setup|-s) DO_SETUP=1 ;;
    --engine|-e)
      if [[ $# -lt 2 ]]; then
        echo "--engine 뒤에 엔진 이름이 필요합니다." >&2
        exit 1
      fi
      ENGINE="$2"
      shift
      ;;
    --engine=*) ENGINE="${1#*=}" ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "알 수 없는 옵션: $1" >&2
      usage >&2
      exit 1
      ;;
    *)
      if [[ -z "$VIDEO" ]]; then
        VIDEO="$1"
      else
        echo "동영상은 하나만 지정할 수 있습니다." >&2
        exit 1
      fi
      ;;
  esac
  shift
done

if [[ -z "$VIDEO" ]]; then
  read -r -p "동영상 파일명(또는 경로)을 입력하세요: " VIDEO
fi

if [[ -z "$ENGINE" ]]; then
  echo "========================================"
  echo " 번역 엔진을 선택하세요."
  echo " [1] 로컬 NLLB (오프라인, 기본값)"
  echo " [2] 구글 번역"
  echo " [3] 로컬 Ollama (qwen3)"
  echo "========================================"
  read -r -p "👉 선택 (Enter = 1): " ENGINE_CHOICE
  case "${ENGINE_CHOICE:-1}" in
    1) ENGINE="nllb" ;;
    2) ENGINE="google" ;;
    3) ENGINE="ollama" ;;
    *)
      echo "잘못된 선택입니다: $ENGINE_CHOICE" >&2
      exit 1
      ;;
  esac
fi

ENGINE="$(echo "$ENGINE" | tr '[:upper:]' '[:lower:]')"
case "$ENGINE" in
  nllb|google|ollama) ;;
  *)
    echo "지원하지 않는 번역 엔진: $ENGINE (nllb | google | ollama)" >&2
    exit 1
    ;;
esac

VIDEO="${VIDEO/#\~/$HOME}"
if [[ ! -f "$VIDEO" ]]; then
  echo "오류: 파일을 찾을 수 없습니다 → $VIDEO" >&2
  exit 1
fi

# --- venv ---
# venv/bin/activate hardcodes the path the venv was created at, so it breaks after the
# project folder is moved. Call the venv interpreter directly instead of activating.
PY="$ROOT/venv/bin/python"

if [[ ! -x "$PY" ]] || ! "$PY" -c "import sys" >/dev/null 2>&1; then
  if [[ -d "$ROOT/venv" ]]; then
    echo "venv가 손상되어 다시 생성합니다 (python3.11)..."
    rm -rf "$ROOT/venv"
  else
    echo "venv가 없어 생성합니다 (python3.11)..."
  fi
  if ! command -v python3.11 >/dev/null 2>&1; then
    echo "오류: python3.11이 없습니다. 'brew install python@3.11'로 설치하세요." >&2
    exit 1
  fi
  python3.11 -m venv "$ROOT/venv"
  DO_SETUP=1
fi

if [[ "$DO_SETUP" -eq 1 ]] || ! "$PY" -c "import mlx_qwen3_asr" >/dev/null 2>&1; then
  echo "패키지 설치 중..."
  "$PY" -m pip install --upgrade pip
  "$PY" -m pip install -r "$ROOT/requirements.txt"
elif [[ "$ENGINE" == "google" ]] && ! "$PY" -c "import deep_translator, requests" >/dev/null 2>&1; then
  echo "구글 번역 패키지 설치 중..."
  "$PY" -m pip install deep-translator requests
fi

mkdir -p "$OUTPUT_DIR"

STEM="$(basename "$VIDEO")"
STEM="${STEM%.*}"
INPUT_SRT="$OUTPUT_DIR/${STEM}.srt"
OUTPUT_SRT="$OUTPUT_DIR/${STEM}_KOR.srt"

echo ""
echo "========================================"
echo " 1) STT (mlx-qwen3-asr)"
echo "    입력: $VIDEO"
echo "    모델: $ASR_MODEL"
echo "    출력: $INPUT_SRT"
echo "========================================"
# python -m 으로 호출해 venv 경로 이전 후에도 깨진 shebang에 의존하지 않음
"$PY" -m mlx_qwen3_asr \
  --model "$ASR_MODEL" \
  --output-format srt \
  --output-dir "$OUTPUT_DIR" \
  "$VIDEO"

if [[ ! -f "$INPUT_SRT" ]]; then
  # 혹시 stem이 다를 경우 최신 srt 탐색
  NEWEST_SRT="$(ls -t "$OUTPUT_DIR"/*.srt 2>/dev/null | head -n 1 || true)"
  if [[ -n "$NEWEST_SRT" && -f "$NEWEST_SRT" ]]; then
    INPUT_SRT="$NEWEST_SRT"
    STEM="$(basename "$INPUT_SRT" .srt)"
    OUTPUT_SRT="$OUTPUT_DIR/${STEM}_KOR.srt"
    echo "알림: 예상과 다른 SRT 이름을 감지해 사용합니다 → $INPUT_SRT"
  else
    echo "오류: SRT가 생성되지 않았습니다 ($OUTPUT_DIR)." >&2
    exit 1
  fi
fi

case "$ENGINE" in
  nllb)   ENGINE_LABEL="로컬 NLLB" ;;
  google) ENGINE_LABEL="구글 번역" ;;
  ollama) ENGINE_LABEL="로컬 Ollama" ;;
esac

echo ""
echo "========================================"
echo " 2) 번역 ($ENGINE_LABEL → 한국어)"
echo "    입력: $INPUT_SRT"
echo "    출력: $OUTPUT_SRT"
echo "========================================"
case "$ENGINE" in
  nllb)   "$PY" "$ROOT/import_srt.py" --input "$INPUT_SRT" --output "$OUTPUT_SRT" ;;
  google) "$PY" "$ROOT/google_translate_srt.py" --input "$INPUT_SRT" --output "$OUTPUT_SRT" ;;
  ollama) "$PY" "$ROOT/translate_srt.py" --input "$INPUT_SRT" --output "$OUTPUT_SRT" ;;
esac

echo ""
echo "완료!"
echo "  원본 자막: $INPUT_SRT"
echo "  한글 자막: $OUTPUT_SRT"
