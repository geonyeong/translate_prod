# translate_prod

로컬 Mac에서 동영상을 **STT(음성→자막) → 한국어 번역**까지 한 번에 처리하는 파이프라인입니다.  
STT는 Apple Silicon(MLX)에서 로컬로 동작하고, 번역은 로컬 모델(NLLB / Ollama) 또는 구글 번역 중에서 선택할 수 있습니다.

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
- **번역(기본)**: `facebook/nllb-200-distilled-600M` (Transformers + MPS, `import_srt.py`)
- **번역(선택)**: 구글 번역 (`google_translate_srt.py`)
- **번역(선택)**: Ollama `qwen3:8b` (`translate_srt.py`)

> 최초 실행 시 모델 다운로드로 수 분~수십 분이 걸릴 수 있으며, 디스크 공간을 수 GB 이상 사용합니다.

## 빠른 시작

```bash
# 1) 저장소 클론 후 이동
cd translate_prod

# 2) 동영상 파일을 프로젝트 폴더(또는 임의 경로)에 준비

# 3) 최초 1회: venv 생성 + 패키지 설치 + 실행
./run.sh 파일명.mp4 --setup

# 4) 이후 실행
./run.sh 파일명.mp4
```

### 1단계: 실행 모드 선택

실행하면 먼저 실행 모드를 고르는 메뉴가 나옵니다. Enter를 누르면 기본값이 선택됩니다(동영상이면 `[1]`, `.srt` 파일이면 `[3]`).

| 번호 | 모드 | 입력 | 결과 |
|------|------|------|------|
| `[1]` | 자막과 번역 한 번에 | 동영상 | 원본 자막 + 한글 자막 |
| `[2]` | 자막만 생성 | 동영상 | 원본 자막 |
| `[3]` | 자막으로 번역 자막만 생성 | 기존 `.srt` 자막 | 한글 자막 |

```bash
./run.sh 파일명.mp4               # 메뉴에서 [1] 또는 [2] 선택
./run.sh result/파일명.srt        # 메뉴에서 [3] 선택 (Enter)
./run.sh                          # 모드 선택 후 파일 경로를 입력받음
```

`[2]`로 자막만 먼저 뽑아 두고, 자막을 확인·수정한 뒤 `[3]`으로 번역하는 식으로 나눠서 쓸 수 있습니다.

### 2단계: 번역 엔진 선택 (`[1]`, `[3]` 모드만)

번역이 포함된 모드에서는 번역 엔진을 고르는 메뉴가 이어서 나옵니다(Enter를 누르면 NLLB).

- `[1]` 로컬 NLLB: 오프라인으로 동작하며, 원본 언어(일본어/영어)를 추가로 선택합니다.
- `[2]` 구글 번역: 원본 언어를 자동 감지하며, 인터넷 연결이 필요합니다.
- `[3]` 로컬 Ollama(qwen3): Ollama 설치가 필요합니다.

### 메뉴 없이 바로 실행

모드와 엔진을 옵션으로 지정하면 해당 메뉴를 건너뜁니다.

```bash
./run.sh 파일명.mp4 --mode full --engine google           # 자막 + 번역
./run.sh 파일명.mp4 --mode stt                            # 자막만
./run.sh result/파일명.srt --mode translate --engine nllb # 번역 자막만
```

| 옵션 | 값 |
|------|----|
| `--mode`, `-m` | `full` (자막+번역) / `stt` (자막만) / `translate` (번역 자막만) |
| `--engine`, `-e` | `nllb` / `google` / `ollama` |
| `--setup`, `-s` | venv 생성 및 패키지 재설치 |

### 구글 번역 옵션

| 방식 | 조건 | 특징 |
|------|------|------|
| 무료 웹 번역 | `GOOGLE_API_KEY` 미설정 | API 키 없이 사용 가능. 요청이 많거나 VPN/공용 IP에서는 일시 차단(HTTP 429)될 수 있음 |
| 공식 Cloud Translation API | `GOOGLE_API_KEY` 설정 | 안정적. [Google Cloud](https://console.cloud.google.com/apis/library/translate.googleapis.com)에서 API 활성화 및 키 발급 필요 (월 50만 자 무료, 이후 유료) |

```bash
# 공식 API 사용
export GOOGLE_API_KEY="발급받은_API_키"
./run.sh 파일명.mp4 --engine google
```

### 결과물

| 파일 | 생성 모드 | 설명 |
|------|-----------|------|
| `result/파일명.srt` | `[1]`, `[2]` | STT로 추출한 원본 자막 |
| `result/파일명_KOR.srt` | `[1]` | 한국어로 번역된 자막 |
| `입력자막과_같은_폴더/파일명_KOR.srt` | `[3]` | 입력한 자막 파일 옆에 한글 자막 생성 |

> `run.sh`는 내부에서 `venv/bin/python`을 직접 실행하므로 `source venv/bin/activate` 없이 바로 실행하면 됩니다.  
> (가상환경은 그대로 사용하며, 프로젝트 폴더를 옮겨도 동작합니다.)

## 수동 실행 (단계별)

통합 스크립트 대신 단계별로 돌릴 수도 있습니다.  
`source venv/bin/activate`는 venv를 만든 경로가 고정되어 폴더를 옮기면 동작하지 않으므로, 아래처럼 `venv/bin/python`을 직접 호출하는 방식을 권장합니다.

```bash
python3.11 -m venv venv
venv/bin/python -m pip install --upgrade pip
venv/bin/python -m pip install -r requirements.txt

# STT
venv/bin/python -m mlx_qwen3_asr \
  --model Qwen/Qwen3-ASR-1.7B \
  --output-format srt \
  --output-dir ./result \
  파일명.mp4

# 번역 (NLLB)
venv/bin/python import_srt.py \
  --input ./result/파일명.srt \
  --output ./result/파일명_KOR.srt

# (선택) 구글 번역 (--lang auto|ja|en, 기본 auto)
venv/bin/python google_translate_srt.py \
  --input ./result/파일명.srt \
  --output ./result/파일명_KOR.srt

# (선택) Ollama 배치 번역
venv/bin/python translate_srt.py \
  --input ./result/파일명.srt \
  --output ./result/파일명_KOR.srt
```

## 프로젝트 구조

```
translate_prod/
├── run.sh                  # 실행 모드 선택 (자막+번역 / 자막만 / 번역만)
├── import_srt.py           # NLLB 로컬 번역
├── google_translate_srt.py # 구글 번역 (무료 웹 / 공식 API)
├── translate_srt.py        # Ollama(qwen3) 배치 번역 (선택)
├── requirements.txt
├── .gitignore
└── result/                 # 자막 출력 (git 제외)
```

## 환경 변수 (선택)

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `ASR_MODEL` | `Qwen/Qwen3-ASR-1.7B` | STT 모델 |
| `OUTPUT_DIR` | `./result` | 자막 출력 폴더 |
| `RUN_MODE` | (선택 메뉴) | 실행 모드 기본값: `full` / `stt` / `translate` |
| `TRANSLATE_ENGINE` | (선택 메뉴) | 번역 엔진 기본값: `nllb` / `google` / `ollama` |
| `GOOGLE_API_KEY` | (없음) | 설정 시 구글 공식 Cloud Translation API 사용 |

예:

```bash
ASR_MODEL=Qwen/Qwen3-ASR-0.6B ./run.sh video.mp4
```

## 추가 목표 (Roadmap)

1. **GUI**  
   파일 선택, 언어 선택, 진행률 표시, 결과 미리보기를 제공하는 데스크톱 UI 추가

2. **외부 번역 API 연동**  
   - [x] 구글 번역 (무료 웹 / 공식 Cloud Translation API)
   - [ ] DeepL, OpenAI 등 추가 외부 API
   - [ ] API 키를 `.env` 파일로 관리

## 주의사항

- `venv/`, 동영상·자막 결과물, Hugging Face/MLX 캐시는 `.gitignore`에 포함되어 저장소에 올라가지 않습니다.
- 프로젝트 경로를 옮긴 뒤에는 `./run.sh 파일명.mp4 --setup`으로 venv를 다시 맞추는 것을 권장합니다.
- Ollama 경로를 쓰려면 별도로 [Ollama](https://ollama.com) 설치 및 `ollama pull qwen3:8b`가 필요합니다.
