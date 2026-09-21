#!/usr/bin/env python3
"""Optional Ollama (qwen3) batch translator for SRT files."""

from __future__ import annotations

import argparse
import os
import re
import sys

import ollama

OLLAMA_MODEL = "qwen3:8b"
BATCH_SIZE = 15  # M2 16GB optimized batch of subtitle blocks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Translate SRT with local Ollama qwen3.")
    parser.add_argument("--input", "-i", required=True, help="Source SRT path")
    parser.add_argument("--output", "-o", required=True, help="Translated SRT path")
    parser.add_argument("--model", default=OLLAMA_MODEL, help=f"Ollama model (default: {OLLAMA_MODEL})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help=f"Blocks per request (default: {BATCH_SIZE})")
    return parser.parse_args()


def translate_batch(batch_blocks: list[str], model: str) -> str:
    batch_text = "\n\n".join(batch_blocks)
    prompt = (
        "You are an expert subtitle translator. Translate the following Japanese SRT subtitle batch into natural, contextual Korean.\n"
        "CRITICAL RULES:\n"
        "1. Maintain the exact same SRT format (Index, Timestamp, and Line breaks).\n"
        "2. Do NOT change or translate any numbers, arrow symbols (-->), or timestamps.\n"
        "3. Translate ONLY the Japanese text into Korean fluidly.\n"
        "4. Do not add any introduction, explanations, or notes. Output ONLY the translated SRT blocks.\n\n"
        f"{batch_text}"
    )

    try:
        response = ollama.generate(model=model, prompt=prompt)
        return response["response"].strip()
    except Exception as e:
        print(f"\n[오류 발생] 배치 번역 중 에러가 발생했습니다: {e}")
        return batch_text


def process_srt(input_srt: str, output_srt: str, model: str, batch_size: int) -> None:
    if not os.path.exists(input_srt):
        print(f"Error: 원본 자막 파일을 찾을 수 없습니다: {input_srt}")
        sys.exit(1)

    print(f"로컬 배치 번역을 시작합니다 (배치 크기: {batch_size}). 100% 오프라인으로 진행됩니다...")

    with open(input_srt, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = re.split(r"\n\s*\n", content.strip())
    translated_content = []

    for i in range(0, len(blocks), batch_size):
        batch = blocks[i : i + batch_size]
        print(f"⚡ 진행 중: {i} ~ {min(i + batch_size, len(blocks))} / 총 {len(blocks)} 단락 처리 중...")
        translated_content.append(translate_batch(batch, model))

    final_srt = "\n\n".join(translated_content)
    os.makedirs(os.path.dirname(os.path.abspath(output_srt)) or ".", exist_ok=True)
    with open(output_srt, "w", encoding="utf-8", newline="\r\n") as f:
        f.write(final_srt)

    print(f"\n🎉 초고속 배치 번역 완료! 저장 위치: {output_srt}")


if __name__ == "__main__":
    args = parse_args()
    process_srt(args.input, args.output, args.model, args.batch_size)
