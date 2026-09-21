# translate_prod

로컬 Mac에서 동영상을 **STT(음성→자막) → 한국어 번역**까지 한 번에 처리하는 파이프라인입니다.  
클라우드 API 없이 Apple Silicon(MLX) + 로컬 NLLB 모델로 동작합니다.

## 사용 환경

| 항목 | 권장 사양 |
|------|-----------|
| 기기 | Mac mini M2 (또는 Apple Silicon Mac) |
| 메모리 | 16GB 이상 |
| 저장공간 | 512GB 이상 (모델·캐시 여유 필요) |
| OS | macOS |
| Python | **3.11** |
| GPU | Apple Metal (MPS / MLX) |

### 사용 모델

- **STT**: `Qwen/Qwen3-ASR-1.7B` (`mlx-qwen3-asr`)
- **번역(기본)**: `facebook/nllb-200-distilled-600M` (Transformers + MPS)
- **번역(선택)**: Ollama `qwen3:8b` (`translate_srt.py`)

> 최초 실행 시 모델 다운로드로 수 분~수십 분이 걸릴 수 있으며, 디스크 공간을 수 GB 이상 사용합니다.

## 빠른 시작

```bash
# 1) 저장소 클론 후 이동
cd translate_prod

# 2) 동영상 파일을 프로젝트 폴더(또는 임의 경로)에 준비

# 3) 최초 1회: venv 생성 + 패키지 설치 + 파이프라인 실행
./run.sh 파일명.mp4 --setup

# 4) 이후 실행
./run.sh 파일명.mp4
```

실행 중 원본 자막 언어를 선택합니다.

- `[1]` 일본어 → 한국어  
- `[2]` 영어 → 한국어  

### 결과물

| 파일 | 설명 |
|------|------|
| `result/파일명.srt` | STT로 추출한 원본 자막 |
| `result/파일명_KOR.srt` | 한국어로 번역된 자막 |

파일명 인자 없이 `./run.sh`만 실행하면 터미널에서 동영상 경로를 입력받습니다.

## 수동 실행 (단계별)

통합 스크립트 대신 단계별로 돌릴 수도 있습니다.

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# STT
python -m mlx_qwen3_asr \
  --model Qwen/Qwen3-ASR-1.7B \
  --output-format srt \
  --output-dir ./result \
  파일명.mp4

# 번역 (NLLB)
python import_srt.py \
  --input ./result/파일명.srt \
  --output ./result/파일명_KOR.srt

# (선택) Ollama 배치 번역
python translate_srt.py \
  --input ./result/파일명.srt \
  --output ./result/파일명_KOR.srt
```

## 프로젝트 구조

```
translate_prod/
├── run.sh              # STT → 번역 통합 실행
├── import_srt.py       # NLLB 로컬 번역
├── translate_srt.py    # Ollama(qwen3) 배치 번역 (선택)
├── requirements.txt
├── .gitignore
└── result/             # 자막 출력 (git 제외)
```

## 환경 변수 (선택)

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `ASR_MODEL` | `Qwen/Qwen3-ASR-1.7B` | STT 모델 |
| `OUTPUT_DIR` | `./result` | 자막 출력 폴더 |

예:

```bash
ASR_MODEL=Qwen/Qwen3-ASR-0.6B ./run.sh video.mp4
```

## 추가 목표 (Roadmap)

1. **GUI**  
   파일 선택, 언어 선택, 진행률 표시, 결과 미리보기를 제공하는 데스크톱 UI 추가

2. **외부 번역 API 연동**  
   로컬 NLLB/Ollama 외에 DeepL, Google Translate, OpenAI 등 외부 API를 선택해 번역할 수 있는 옵션 추가  
   (API 키는 환경 변수 / `.env`로 관리, 로컬 전용 모드와 병행)

## 주의사항

- `venv/`, 동영상·자막 결과물, Hugging Face/MLX 캐시는 `.gitignore`에 포함되어 저장소에 올라가지 않습니다.
- 프로젝트 경로를 옮긴 뒤에는 `./run.sh 파일명.mp4 --setup`으로 venv를 다시 맞추는 것을 권장합니다.
- Ollama 경로를 쓰려면 별도로 [Ollama](https://ollama.com) 설치 및 `ollama pull qwen3:8b`가 필요합니다.
