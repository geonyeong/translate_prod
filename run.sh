#!/usr/bin/env bash
# Subtitle pipeline: video -> SRT (Qwen3-ASR) -> Korean SRT (NLLB / Google / Ollama)
# Usage:
#   ./run.sh                                   # menus for mode, file, engine, language
#   ./run.sh video.mp4                         # pick mode/engine/language from menus
#   ./run.sh video.mp4 --mode full --engine ollama --lang ja
#   ./run.sh video.mp4 --mode stt
#   ./run.sh result/video.srt --mode translate --engine nllb
#   ./run.sh video.mp4 --setup                 # also (re)install packages
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

ASR_MODEL="${ASR_MODEL:-mlx-community/Qwen3-ASR-1.7B-8bit}"
OUTPUT_DIR="${OUTPUT_DIR:-./result}"
ENGINE="${TRANSLATE_ENGINE:-}"
MODE="${RUN_MODE:-}"
SRC_LANG="${SRC_LANG:-}"
CLEAN="${CLEAN:-1}"
DO_SETUP=0
INPUT=""

usage() {
  echo "Usage: ./run.sh [file] [--mode full|stt|translate] [--engine nllb|google|ollama] [--lang ja|en|auto] [--no-clean] [--setup]"
  echo "  file            동영상 경로 (translate 모드는 .srt 경로). 생략 시 입력 요청"
  echo "  --mode, -m      실행 모드 (생략 시 선택 메뉴 표시)"
  echo "                    full      : 동영상 → 자막 → 번역 자막 (한 번에)"
  echo "                    stt       : 동영상 → 자막만 생성"
  echo "                    translate : 기존 자막(.srt) → 번역 자막만 생성"
  echo "  --engine, -e    번역 엔진 (생략 시 선택 메뉴 표시, stt 모드에서는 무시)"
  echo "                    nllb   : 로컬 NLLB 모델 (오프라인, 빠름)"
  echo "                    google : 구글 번역 (GOOGLE_API_KEY 있으면 공식 API, 없으면 무료 웹)"
  echo "                    ollama : 로컬 Ollama qwen3 (앞뒤 문맥을 보고 번역, 품질 우선)"
  echo "  --lang, -l      원본 언어 ja | en | auto (생략 시 선택 메뉴). 음성 인식과 번역에 모두 사용"
  echo "  --no-clean      자막 정리(반복·잡음 제거, 조각 병합) 끄기"
  echo "  --setup, -s     venv 생성 및 requirements 설치"
}

require_value() {
  if [[ $# -lt 2 || -z "$2" ]]; then
    echo "$1 뒤에 값이 필요합니다." >&2
    exit 1
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --setup|-s) DO_SETUP=1 ;;
    --engine|-e)
      require_value "$@"
      ENGINE="$2"
      shift
      ;;
    --engine=*) ENGINE="${1#*=}" ;;
    --mode|-m)
      require_value "$@"
      MODE="$2"
      shift
      ;;
    --mode=*) MODE="${1#*=}" ;;
    --lang|-l)
      require_value "$@"
      SRC_LANG="$2"
      shift
      ;;
    --lang=*) SRC_LANG="${1#*=}" ;;
    --no-clean) CLEAN=0 ;;
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
      if [[ -z "$INPUT" ]]; then
        INPUT="$1"
      else
        echo "파일은 하나만 지정할 수 있습니다." >&2
        exit 1
      fi
      ;;
  esac
  shift
done

is_srt() {
  local lower
  lower="$(echo "$1" | tr '[:upper:]' '[:lower:]')"
  [[ "$lower" == *.srt ]]
}

# --- 실행 모드 ---
if [[ -z "$MODE" ]]; then
  DEFAULT_MODE=1
  if [[ -n "$INPUT" ]] && is_srt "$INPUT"; then
    DEFAULT_MODE=3
  fi
  echo "========================================"
  echo " 실행 모드를 선택하세요."
  echo " [1] 자막과 번역 한 번에 (동영상 → 자막 → 번역 자막)"
  echo " [2] 자막만 생성 (동영상 → 자막)"
  echo " [3] 자막으로 번역 자막만 생성 (.srt → 번역 자막)"
  echo "========================================"
  read -r -p "👉 선택 (Enter = $DEFAULT_MODE): " MODE_CHOICE
  case "${MODE_CHOICE:-$DEFAULT_MODE}" in
    1) MODE="full" ;;
    2) MODE="stt" ;;
    3) MODE="translate" ;;
    *)
      echo "잘못된 선택입니다: $MODE_CHOICE" >&2
      exit 1
      ;;
  esac
fi

MODE="$(echo "$MODE" | tr '[:upper:]' '[:lower:]')"
case "$MODE" in
  full|stt|translate) ;;
  *)
    echo "지원하지 않는 실행 모드: $MODE (full | stt | translate)" >&2
    exit 1
    ;;
esac

# --- 입력 파일 ---
if [[ -z "$INPUT" ]]; then
  if [[ "$MODE" == "translate" ]]; then
    read -r -p "번역할 자막 파일(.srt) 경로를 입력하세요: " INPUT
  else
    read -r -p "동영상 파일명(또는 경로)을 입력하세요: " INPUT
  fi
fi

INPUT="${INPUT/#\~/$HOME}"
if [[ ! -f "$INPUT" ]]; then
  echo "오류: 파일을 찾을 수 없습니다 → $INPUT" >&2
  exit 1
fi

if [[ "$MODE" == "translate" ]] && ! is_srt "$INPUT"; then
  echo "오류: 번역 자막만 생성 모드는 .srt 파일이 필요합니다 → $INPUT" >&2
  exit 1
fi
if [[ "$MODE" != "translate" ]] && is_srt "$INPUT"; then
  echo "오류: 자막 파일이 입력되었습니다. 기존 자막을 번역하려면 [3] (--mode translate)을 선택하세요." >&2
  exit 1
fi

# --- 번역 엔진 (번역이 포함된 모드만) ---
if [[ "$MODE" != "stt" ]]; then
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
fi

# --- 원본 언어 (음성 인식 언어 고정 + 번역 원문 언어) ---
ALLOW_AUTO=1
if [[ "$MODE" != "stt" && "$ENGINE" == "nllb" ]]; then
  ALLOW_AUTO=0
fi
if [[ -z "$SRC_LANG" ]]; then
  echo "========================================"
  echo " 원본 언어를 선택하세요. (음성 인식·번역에 사용)"
  echo " [1] 일본어 (기본값)"
  echo " [2] 영어"
  if [[ "$ALLOW_AUTO" == "1" ]]; then
    echo " [3] 자동 감지"
  fi
  echo "========================================"
  read -r -p "👉 선택 (Enter = 1): " LANG_CHOICE
  case "${LANG_CHOICE:-1}" in
    1) SRC_LANG="ja" ;;
    2) SRC_LANG="en" ;;
    3) SRC_LANG="auto" ;;
    *)
      echo "잘못된 선택입니다: $LANG_CHOICE" >&2
      exit 1
      ;;
  esac
fi
SRC_LANG="$(echo "$SRC_LANG" | tr '[:upper:]' '[:lower:]')"
case "$SRC_LANG" in
  ja|en) ;;
  auto)
    if [[ "$ALLOW_AUTO" == "0" ]]; then
      echo "NLLB 번역은 원본 언어를 직접 지정해야 합니다 (--lang ja | en)." >&2
      exit 1
    fi
    ;;
  *)
    echo "지원하지 않는 언어: $SRC_LANG (ja | en | auto)" >&2
    exit 1
    ;;
esac

CLEAN_ARGS=()
if [[ "$CLEAN" == "0" ]]; then
  CLEAN_ARGS=(--no-clean)
fi

# --- venv ---
# shellcheck source=lib/venv.sh
source "$ROOT/lib/venv.sh"
ensure_venv "$ROOT" "$DO_SETUP"
if [[ "$ENGINE" == "google" ]]; then
  ensure_modules "deep_translator, requests" deep-translator requests
fi

# --- 1) STT ---
run_stt() {
  mkdir -p "$OUTPUT_DIR"

  local stem
  stem="$(basename "$INPUT")"
  stem="${stem%.*}"
  SRT_FILE="$OUTPUT_DIR/${stem}.srt"

  local lang_args=()
  case "$SRC_LANG" in
    ja) lang_args=(--language Japanese) ;;
    en) lang_args=(--language English) ;;
  esac

  echo ""
  echo "========================================"
  echo " STT (mlx-qwen3-asr)"
  echo "    입력: $INPUT"
  echo "    모델: $ASR_MODEL"
  echo "    언어: $SRC_LANG"
  echo "    출력: $SRT_FILE"
  echo "========================================"
  # 기본 SRT 출력은 문장부호가 빠지고 10단어마다 끊기므로, 문장부호가 살아있는 JSON(+단어 타임스탬프)을
  # 임시 폴더에 받은 뒤 srt_cleanup.py가 문장 단위 자막으로 다시 만든다.
  local tmp_dir json_file
  tmp_dir="$(mktemp -d)"
  # python -m 으로 호출해 venv 경로 이전 후에도 깨진 shebang에 의존하지 않음
  "$PY" -m mlx_qwen3_asr \
    --model "$ASR_MODEL" \
    --output-format json \
    --timestamps \
    --output-dir "$tmp_dir" \
    ${lang_args[@]+"${lang_args[@]}"} \
    "$INPUT"

  json_file="$(ls "$tmp_dir"/*.json 2>/dev/null | head -n 1 || true)"
  if [[ -z "$json_file" ]]; then
    rm -rf "$tmp_dir"
    echo "오류: 음성 인식 결과가 생성되지 않았습니다." >&2
    exit 1
  fi
  "$PY" "$ROOT/srt_cleanup.py" --input "$json_file" --output "$SRT_FILE" --lang "$SRC_LANG" \
    ${CLEAN_ARGS[@]+"${CLEAN_ARGS[@]}"}
  rm -rf "$tmp_dir"
}

# --- 2) 번역 ---
run_translate() {
  local src="$1"
  local dir stem label
  dir="$(dirname "$src")"
  stem="$(basename "$src")"
  stem="${stem%.*}"
  KOR_FILE="$dir/${stem}_KOR.srt"

  case "$ENGINE" in
    nllb)   label="로컬 NLLB" ;;
    google) label="구글 번역" ;;
    ollama) label="로컬 Ollama" ;;
  esac

  echo ""
  echo "========================================"
  echo " 번역 ($label → 한국어)"
  echo "    입력: $src"
  echo "    출력: $KOR_FILE"
  echo "========================================"
  local script
  case "$ENGINE" in
    nllb)   script="import_srt.py" ;;
    google) script="google_translate_srt.py" ;;
    ollama) script="translate_srt.py" ;;
  esac
  "$PY" "$ROOT/$script" --input "$src" --output "$KOR_FILE" --lang "$SRC_LANG" \
    ${CLEAN_ARGS[@]+"${CLEAN_ARGS[@]}"}
}

case "$MODE" in
  full)
    run_stt
    run_translate "$SRT_FILE"
    echo ""
    echo "완료!"
    echo "  원본 자막: $SRT_FILE"
    echo "  한글 자막: $KOR_FILE"
    ;;
  stt)
    run_stt
    echo ""
    echo "완료!"
    echo "  원본 자막: $SRT_FILE"
    ;;
  translate)
    run_translate "$INPUT"
    echo ""
    echo "완료!"
    echo "  한글 자막: $KOR_FILE"
    ;;
esac
