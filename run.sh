#!/usr/bin/env bash
# One-shot pipeline: video -> SRT (Qwen3-ASR) -> Korean SRT (NLLB)
# Usage:
#   ./run.sh video.mp4
#   ./run.sh                  # prompts for filename
#   ./run.sh video.mp4 --setup  # also (re)install packages
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

ASR_MODEL="${ASR_MODEL:-Qwen/Qwen3-ASR-1.7B}"
OUTPUT_DIR="${OUTPUT_DIR:-./result}"
DO_SETUP=0
VIDEO=""

for arg in "$@"; do
  case "$arg" in
    --setup|-s) DO_SETUP=1 ;;
    -h|--help)
      echo "Usage: ./run.sh [video_file] [--setup]"
      echo "  video_file  동영상 경로 (생략 시 입력 요청)"
      echo "  --setup     venv 생성 및 requirements 설치"
      exit 0
      ;;
    -*)
      echo "알 수 없는 옵션: $arg" >&2
      exit 1
      ;;
    *)
      if [[ -z "$VIDEO" ]]; then
        VIDEO="$arg"
      else
        echo "동영상은 하나만 지정할 수 있습니다." >&2
        exit 1
      fi
      ;;
  esac
done

if [[ -z "$VIDEO" ]]; then
  read -r -p "동영상 파일명(또는 경로)을 입력하세요: " VIDEO
fi

VIDEO="${VIDEO/#\~/$HOME}"
if [[ ! -f "$VIDEO" ]]; then
  echo "오류: 파일을 찾을 수 없습니다 → $VIDEO" >&2
  exit 1
fi

# --- venv ---
if [[ ! -d "$ROOT/venv" ]]; then
  echo "venv가 없어 생성합니다 (python3.11)..."
  python3.11 -m venv "$ROOT/venv"
  DO_SETUP=1
fi

# shellcheck disable=SC1091
source "$ROOT/venv/bin/activate"

if [[ "$DO_SETUP" -eq 1 ]] || ! python -c "import mlx_qwen3_asr" >/dev/null 2>&1; then
  echo "패키지 설치 중..."
  python -m pip install --upgrade pip
  python -m pip install -r "$ROOT/requirements.txt"
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
python -m mlx_qwen3_asr \
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

echo ""
echo "========================================"
echo " 2) 번역 (NLLB → 한국어)"
echo "    입력: $INPUT_SRT"
echo "    출력: $OUTPUT_SRT"
echo "========================================"
python "$ROOT/import_srt.py" --input "$INPUT_SRT" --output "$OUTPUT_SRT"

echo ""
echo "완료!"
echo "  원본 자막: $INPUT_SRT"
echo "  한글 자막: $OUTPUT_SRT"
