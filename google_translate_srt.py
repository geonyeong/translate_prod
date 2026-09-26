#!/usr/bin/env python3
"""Translate an SRT file to Korean with Google Translate.

- GOOGLE_API_KEY (or --api-key) set: official Cloud Translation API v2 (paid, stable)
- otherwise: free Google Translate web endpoint via deep-translator (no key, rate limited)
"""

from __future__ import annotations

import argparse
import html
import os
import sys
import time

import requests
import srt

CLOUD_ENDPOINT = "https://translation.googleapis.com/language/translate/v2"
# Free endpoint rejects requests over 5000 chars; Cloud API recommends <= 128 segments per call.
FREE_CHUNK_CHARS = 4500
CLOUD_CHUNK_SEGMENTS = 100
MAX_RETRIES = 3

LANG_ALIASES = {
    "auto": "auto",
    "1": "ja",
    "ja": "ja",
    "jpn": "ja",
    "2": "en",
    "en": "en",
    "eng": "en",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Translate SRT subtitles to Korean with Google Translate.")
    parser.add_argument("--input", "-i", required=True, help="Source SRT path")
    parser.add_argument("--output", "-o", required=True, help="Translated SRT path")
    parser.add_argument(
        "--lang",
        "-l",
        choices=sorted(LANG_ALIASES),
        default="auto",
        help="Source language: auto (default), 1/ja/jpn=Japanese, 2/en/eng=English",
    )
    parser.add_argument("--target", "-t", default="ko", help="Target language code (default: ko)")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("GOOGLE_API_KEY"),
        help="Google Cloud Translation API key (default: $GOOGLE_API_KEY). Omit to use the free web endpoint.",
    )
    return parser.parse_args()


def with_retry(func, *args):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return func(*args)
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            wait = 2 * attempt
            print(f"\n⚠️ 요청 실패 ({e}). {wait}초 후 재시도 ({attempt}/{MAX_RETRIES - 1})...")
            time.sleep(wait)


class CloudTranslator:
    def __init__(self, api_key: str, source: str, target: str):
        self.api_key = api_key
        self.source = source
        self.target = target

    def translate(self, lines: list[str]) -> list[str]:
        results: list[str] = []
        for start in range(0, len(lines), CLOUD_CHUNK_SEGMENTS):
            chunk = lines[start : start + CLOUD_CHUNK_SEGMENTS]
            results.extend(with_retry(self._request, chunk))
            print(f" 진행 중: [{len(results)}/{len(lines)}] 줄 완료", end="\r")
        return results

    def _request(self, chunk: list[str]) -> list[str]:
        payload = {"q": chunk, "target": self.target, "format": "text"}
        if self.source != "auto":
            payload["source"] = self.source
        resp = requests.post(CLOUD_ENDPOINT, params={"key": self.api_key}, json=payload, timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        return [html.unescape(t["translatedText"]) for t in resp.json()["data"]["translations"]]


class FreeTranslator:
    def __init__(self, source: str, target: str):
        try:
            from deep_translator import GoogleTranslator
        except ImportError:
            print("❌ deep-translator가 설치되어 있지 않습니다: pip install deep-translator")
            sys.exit(1)
        self.client = GoogleTranslator(source=source, target=target)

    def translate(self, lines: list[str]) -> list[str]:
        results: list[str] = []
        for chunk in self._chunks(lines):
            results.extend(self._translate_chunk(chunk))
            print(f" 진행 중: [{len(results)}/{len(lines)}] 줄 완료", end="\r")
        return results

    @staticmethod
    def _chunks(lines: list[str]):
        chunk: list[str] = []
        size = 0
        for line in lines:
            if chunk and size + len(line) + 1 > FREE_CHUNK_CHARS:
                yield chunk
                chunk, size = [], 0
            chunk.append(line)
            size += len(line) + 1
        if chunk:
            yield chunk

    def _translate_chunk(self, chunk: list[str]) -> list[str]:
        # Send many lines in one request; fall back to per-line if Google merges/splits lines.
        translated = with_retry(self.client.translate, "\n".join(chunk)) or ""
        parts = translated.split("\n")
        if len(parts) == len(chunk):
            return [p.strip() for p in parts]
        return [(with_retry(self.client.translate, line) or line).strip() for line in chunk]


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.input):
        print(f"❌ 오류: 파일 '{args.input}'이 없습니다. 경로를 확인해주세요.")
        sys.exit(1)

    source = LANG_ALIASES[args.lang]

    if args.api_key:
        print("☁️ Google Cloud Translation API (공식, API 키 사용)로 번역합니다.")
        translator = CloudTranslator(args.api_key, source, args.target)
    else:
        print("🌐 Google 번역 (무료 웹, API 키 없음)으로 번역합니다. 요청이 많으면 일시 차단될 수 있습니다.")
        translator = FreeTranslator(source, args.target)

    with open(args.input, "r", encoding="utf-8-sig", errors="ignore") as f:
        subtitles = list(srt.parse(f.read()))

    # Flatten non-empty lines so they can be batched, then map results back.
    positions: list[tuple[int, int]] = []
    lines: list[str] = []
    split_contents = [sub.content.strip().split("\n") for sub in subtitles]
    for sub_idx, sub_lines in enumerate(split_contents):
        for line_idx, line in enumerate(sub_lines):
            if line.strip():
                positions.append((sub_idx, line_idx))
                lines.append(line.strip())

    print(f"🚀 번역 시작: 자막 {len(subtitles)}개 / {len(lines)}줄 ({source} ➡️ {args.target})")
    try:
        translated = translator.translate(lines)
    except Exception as e:
        print(f"\n❌ 번역 실패: {e}")
        if not args.api_key and "too many requests" in str(e).lower():
            print("💡 무료 구글 번역이 현재 네트워크(IP)를 일시 차단했습니다.")
            print("   잠시 후 다시 시도하거나, VPN을 끄거나, GOOGLE_API_KEY를 설정해 공식 API를 사용하세요.")
        sys.exit(1)

    for (sub_idx, line_idx), text in zip(positions, translated):
        split_contents[sub_idx][line_idx] = text
    for sub, sub_lines in zip(subtitles, split_contents):
        sub.content = "\n".join(sub_lines).strip()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(srt.compose(subtitles))

    print(f"\n🎉 구글 번역 완료! '{args.output}' 파일이 생성되었습니다.")


if __name__ == "__main__":
    main()
