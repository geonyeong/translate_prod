#!/usr/bin/env python3
"""Translate an SRT file to Korean with a local NLLB model (Apple MPS when available).

Each subtitle cue is translated as one sentence (not line by line); identical lines
are translated once, and cues are batched by length for GPU throughput.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from srt_cleanup import prepare_for_translation, save_cues, wrap_lines

NLLB_MODELS = {
    "600m": "facebook/nllb-200-distilled-600M",
    "1.3b": "facebook/nllb-200-distilled-1.3B",
}
DEFAULT_MODEL = os.environ.get("NLLB_MODEL", "600m")
DEFAULT_BATCH = 32
TGT_LANG = "kor_Hang"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Translate SRT subtitles to Korean (NLLB-200).")
    parser.add_argument(
        "--input",
        "-i",
        default=os.environ.get("INPUT_SRT", "INPUT.srt"),
        help="Source SRT path (default: INPUT.srt or $INPUT_SRT)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=os.environ.get("OUTPUT_SRT", "OUTPUT.srt"),
        help="Translated SRT path (default: OUTPUT.srt or $OUTPUT_SRT)",
    )
    parser.add_argument(
        "--lang",
        "-l",
        choices=("1", "2", "ja", "en", "jpn", "eng"),
        default=None,
        help="Source language: 1/ja/jpn=Japanese, 2/en/eng=English. Prompt if omitted.",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=DEFAULT_MODEL,
        help="600m (default, fast), 1.3b (better quality, slower), or a Hugging Face model id ($NLLB_MODEL)",
    )
    parser.add_argument("--batch-size", "-b", type=int, default=DEFAULT_BATCH, help=f"Cues per batch (default: {DEFAULT_BATCH})")
    parser.add_argument("--no-clean", action="store_true", help="Skip subtitle cleanup before translating")
    return parser.parse_args()


def resolve_src_lang(choice: str | None) -> tuple[str, str, str]:
    if choice is None:
        print("=" * 40)
        print(" 번역할 원본 자막의 언어를 선택하세요.")
        print(" [1] 일본어 (Japanese)")
        print(" [2] 영어 (English)")
        print("=" * 40)
        choice = input("👉 선택 (1 또는 2 입력): ").strip()

    normalized = choice.lower()
    if normalized in {"1", "ja", "jpn", "japanese"}:
        return "jpn_Jpan", "일본어", "ja"
    if normalized in {"2", "en", "eng", "english"}:
        return "eng_Latn", "영어", "en"

    print("❌ 잘못된 선택입니다. 프로그램을 종료합니다.")
    sys.exit(1)


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        print("\n🍏 M2 Mac Metal(MPS) GPU 가속 엔진을 활성화합니다.")
        return torch.device("mps")
    print("\n⚠️ MPS를 사용할 수 없어 CPU 모드로 구동합니다. (속도가 느릴 수 있음)")
    return torch.device("cpu")


def translate_unique(texts: list[str], tokenizer, model, device, batch_size: int) -> dict[str, str]:
    unique = sorted(set(texts), key=len)
    bos = tokenizer.convert_tokens_to_ids(TGT_LANG)
    results: dict[str, str] = {}
    done = 0
    for start in range(0, len(unique), batch_size):
        batch = unique[start : start + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        max_new = min(256, int(inputs["input_ids"].shape[1] * 2) + 16)
        with torch.inference_mode():
            tokens = model.generate(**inputs, forced_bos_token_id=bos, max_new_tokens=max_new)
        for src, out in zip(batch, tokenizer.batch_decode(tokens, skip_special_tokens=True)):
            results[src] = out.strip()
        done += len(batch)
        print(f" 진행 중: [{done}/{len(unique)}] 완료", end="\r", flush=True)
    return results


def main() -> None:
    args = parse_args()
    input_srt = args.input
    output_srt = args.output

    if not os.path.exists(input_srt):
        print(f"❌ 오류: 파일 '{input_srt}'이 없습니다. 경로를 확인해주세요.")
        sys.exit(1)

    src_lang, lang_name, lang_code = resolve_src_lang(args.lang)
    model_name = NLLB_MODELS.get(args.model.lower(), args.model)
    device = pick_device()

    print(f"⏳ 로컬 AI 번역 모델({model_name})을 로드하고 있습니다... (최초 실행 시 다운로드로 인해 수 분 소요)")
    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=src_lang)
    dtype = torch.float16 if device.type == "mps" else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, dtype=dtype).to(device).eval()
    model.generation_config.max_length = None

    print(f"\n📂 '{input_srt}' 자막 파일을 읽고 있습니다...")
    cues = prepare_for_translation(input_srt, lang_code, clean=not args.no_clean)
    texts = [" ".join(c.text.split()) for c in cues]

    unique_count = len(set(texts))
    print(f"🚀 로컬 번역 시작 (원문: {lang_name} ➡️ 타겟: 한국어) · 자막 {len(texts)}개 (고유 {unique_count}개)")
    started = time.monotonic()
    translations = translate_unique(texts, tokenizer, model, device, max(1, args.batch_size))

    for cue, text in zip(cues, texts):
        cue.text = wrap_lines(translations.get(text, text), 22)

    print(f"\n💾 번역된 자막 파일을 저장하고 있습니다... ({time.monotonic() - started:.1f}초)")
    save_cues(output_srt, cues)
    print(f"\n🎉 로컬 AI 자막 번역이 완료되었습니다! '{output_srt}' 파일이 생성되었습니다.")


if __name__ == "__main__":
    main()
