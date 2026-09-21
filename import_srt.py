#!/usr/bin/env python3
"""Translate an SRT file to Korean with a local NLLB model (Apple MPS when available)."""

from __future__ import annotations

import argparse
import os
import sys

import srt
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


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
    return parser.parse_args()


def resolve_src_lang(choice: str | None) -> tuple[str, str]:
    if choice is None:
        print("=" * 40)
        print(" 번역할 원본 자막의 언어를 선택하세요.")
        print(" [1] 일본어 (Japanese)")
        print(" [2] 영어 (English)")
        print("=" * 40)
        choice = input("👉 선택 (1 또는 2 입력): ").strip()

    normalized = choice.lower()
    if normalized in {"1", "ja", "jpn", "japanese"}:
        return "jpn_Jpan", "일본어"
    if normalized in {"2", "en", "eng", "english"}:
        return "eng_Latn", "영어"

    print("❌ 잘못된 선택입니다. 프로그램을 종료합니다.")
    sys.exit(1)


def main() -> None:
    args = parse_args()
    input_srt = args.input
    output_srt = args.output

    if not os.path.exists(input_srt):
        print(f"❌ 오류: 파일 '{input_srt}'이 없습니다. 경로를 확인해주세요.")
        sys.exit(1)

    src_lang, lang_name = resolve_src_lang(args.lang)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("\n🍏 M2 Mac Metal(MPS) GPU 가속 엔진을 활성화합니다.")
    else:
        device = torch.device("cpu")
        print("\n⚠️ MPS를 사용할 수 없어 CPU 모드로 구동합니다. (속도가 느릴 수 있음)")

    print("⏳ 로컬 AI 번역 모델(NLLB-200)을 로드하고 있습니다... (최초 실행 시 다운로드로 인해 수 분 소요)")
    model_name = "facebook/nllb-200-distilled-600M"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    print(f"\n📂 '{input_srt}' 자막 파일을 읽고 있습니다...")
    with open(input_srt, "r", encoding="utf-8-sig", errors="ignore") as f:
        content = f.read()

    subtitles = list(srt.parse(content))
    total = len(subtitles)

    print(f"🚀 로컬 번역 시작 (원문: {lang_name} ➡️ 타겟: 한국어)")
    tgt_lang = "kor_Hang"

    for idx, sub in enumerate(subtitles):
        text = sub.content.strip()
        if not text:
            continue

        lines = text.split("\n")
        translated_lines = []

        for line in lines:
            if not line.strip():
                translated_lines.append("")
                continue

            inputs = tokenizer(line, return_tensors="pt").to(device)
            translated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
                max_length=128,
            )
            translated_line = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            translated_lines.append(translated_line)

        sub.content = "\n".join(translated_lines).strip()

        if (idx + 1) % 10 == 0 or (idx + 1) == total:
            print(f" 진행 중: [{idx + 1}/{total}] 완료", end="\r")

    print("\n💾 번역된 자막 파일을 저장하고 있습니다...")
    os.makedirs(os.path.dirname(os.path.abspath(output_srt)) or ".", exist_ok=True)
    translated_content = srt.compose(subtitles)
    with open(output_srt, "w", encoding="utf-8") as f:
        f.write(translated_content)

    print(f"\n🎉 로컬 AI 자막 번역이 완료되었습니다! '{output_srt}' 파일이 생성되었습니다.")


if __name__ == "__main__":
    main()
