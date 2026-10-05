#!/usr/bin/env python3
"""Context-aware SRT translator using a local Ollama model (default: qwen3:8b).

Timestamps never go through the LLM: only numbered subtitle texts are sent, together
with the previous translations and the upcoming lines as read-only context, and the
reply is a JSON object validated against the requested line numbers.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import ollama

from srt_cleanup import Cue, prepare_for_translation, save_cues, wrap_lines

ROOT = Path(__file__).resolve().parent
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:8b")
BATCH_SIZE = 20
CONTEXT_BEFORE = 6
CONTEXT_AFTER = 3
DEFAULT_GLOSSARY = ROOT / "glossary.txt"
ERROR_MARK = "[오류 발생]"

LANG_NAMES = {"ja": "Japanese", "en": "English", "auto": "the source language"}
KANA_RE = re.compile(r"[\u3040-\u30ff]")
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Translate SRT to Korean with a local Ollama LLM (context-aware).")
    parser.add_argument("--input", "-i", required=True, help="Source SRT path")
    parser.add_argument("--output", "-o", required=True, help="Translated SRT path")
    parser.add_argument("--model", default=OLLAMA_MODEL, help=f"Ollama model (default: {OLLAMA_MODEL})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help=f"Lines per request (default: {BATCH_SIZE})")
    parser.add_argument(
        "--lang",
        "-l",
        choices=sorted(LANG_NAMES),
        default="ja",
        help="Source language: ja (default), en, auto",
    )
    parser.add_argument(
        "--glossary",
        "-g",
        default=None,
        help=f"Glossary file ('원문 = 번역' per line). Default: {DEFAULT_GLOSSARY.name} if it exists",
    )
    parser.add_argument("--no-clean", action="store_true", help="Skip subtitle cleanup before translating")
    return parser.parse_args()


def load_glossary(path: str | None) -> dict[str, str]:
    file = Path(path) if path else DEFAULT_GLOSSARY
    if not file.is_file():
        if path:
            print(f"⚠️ 용어집을 찾을 수 없습니다: {path}")
        return {}
    glossary: dict[str, str] = {}
    for raw in file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        sep = "=" if "=" in line else "\t" if "\t" in line else None
        if sep is None:
            continue
        src, dst = (part.strip() for part in line.split(sep, 1))
        if src and dst:
            glossary[src] = dst
    if glossary:
        print(f"📖 용어집 {len(glossary)}개 항목을 사용합니다 ({file.name})")
    return glossary


def system_prompt(lang: str) -> str:
    name = LANG_NAMES[lang]
    return (
        f"You are a professional subtitle translator. Translate {name} subtitles into natural, "
        "colloquial Korean that fits spoken dialogue.\n"
        "Rules:\n"
        "- Translate every numbered line in [TRANSLATE]. Keep each line separate: never merge, split, or skip lines.\n"
        "- Use [PREVIOUS] and [NEXT] only to understand the context (who is speaking, omitted subjects, "
        "sentences that continue across lines). Do not translate them.\n"
        "- Keep the speech level (반말/존댓말) consistent with the previous translations.\n"
        "- Keep subtitles short and readable. Do not add explanations, notes, or romanization.\n"
        "- The output must be 100% Korean (Hangul): never leave Japanese endings such as ましょう, ます, です, "
        "ください, 必要があります, and write names in Hangul.\n"
        "- If a line is only an interjection or a sound, render it as a short Korean interjection.\n"
        "- Follow the glossary exactly when a term appears.\n"
        'Reply with JSON only: {"1": "번역", "2": "번역", ...} using exactly the line numbers in [TRANSLATE].'
    )


def build_user_prompt(
    batch: list[str],
    before: list[tuple[str, str]],
    after: list[str],
    glossary: dict[str, str],
) -> str:
    parts: list[str] = []
    joined = "\n".join(batch)
    terms = {src: dst for src, dst in glossary.items() if src in joined}
    if terms:
        parts.append("[GLOSSARY]\n" + "\n".join(f"{src} = {dst}" for src, dst in terms.items()))
    if before:
        parts.append("[PREVIOUS]\n" + "\n".join(f"{src}  =>  {dst}" for src, dst in before))
    parts.append("[TRANSLATE]\n" + "\n".join(f"{i}: {text}" for i, text in enumerate(batch, 1)))
    if after:
        parts.append("[NEXT]\n" + "\n".join(after))
    return "\n\n".join(parts)


FEW_SHOT = {
    "ja": (
        ["一緒に行きましょう。", "明日までに提出する必要があります。", "ちょっと待ってください。"],
        ["같이 가요.", "내일까지 제출해야 해요.", "잠깐만 기다려 주세요."],
    ),
    "en": (
        ["Let's go together.", "You need to submit it by tomorrow.", "Wait a second, please."],
        ["같이 가요.", "내일까지 제출해야 해요.", "잠깐만 기다려 주세요."],
    ),
}

# qwen3 sometimes keeps a Japanese verb ending verbatim ("만나ましょう") even after retries.
KANA_ENDING_FIXES = [
    ("必要があります", " 필요가 있어요"),
    ("必要がある", " 필요가 있어"),
    ("ましょう", "요"),
    ("ください", " 주세요"),
    ("でしょう", "겠죠"),
    ("です", "요"),
    ("ます", "요"),
]


def few_shot_messages(lang: str) -> list[dict]:
    sources, targets = FEW_SHOT.get(lang, FEW_SHOT["ja"])
    return [
        {"role": "user", "content": "[TRANSLATE]\n" + "\n".join(f"{i}: {t}" for i, t in enumerate(sources, 1))},
        {
            "role": "assistant",
            "content": json.dumps({str(i): t for i, t in enumerate(targets, 1)}, ensure_ascii=False),
        },
    ]


def repair_kana(text: str) -> str:
    for src, dst in KANA_ENDING_FIXES:
        text = text.replace(src, dst)
    return re.sub(r" {2,}", " ", text).strip()


def response_schema(count: int) -> dict:
    keys = [str(i) for i in range(1, count + 1)]
    return {
        "type": "object",
        "properties": {key: {"type": "string"} for key in keys},
        "required": keys,
    }


def request_batch(
    batch: list[str],
    before: list[tuple[str, str]],
    after: list[str],
    *,
    model: str,
    lang: str,
    glossary: dict[str, str],
    strict: bool = True,
    temperature: float = 0.2,
) -> list[str | None]:
    """Translate one batch; returns None for lines the model failed to translate.

    strict: also reject lines that still contain kana (left untranslated).
    """
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt(lang)},
            *few_shot_messages(lang),
            {"role": "user", "content": build_user_prompt(batch, before, after, glossary)},
        ],
        format=response_schema(len(batch)),
        think=False,
        options={"temperature": temperature, "num_ctx": 8192},
        # Unload soon after the run: a resident 5GB model makes the next STT run ~3x slower on 16GB Macs.
        keep_alive="1m",
    )
    content = THINK_RE.sub("", response["message"]["content"]).strip()
    data = json.loads(content)
    results: list[str | None] = []
    for i, src in enumerate(batch, 1):
        text = str(data.get(str(i), "")).strip()
        untranslated = strict and lang != "en" and KANA_RE.search(text) and KANA_RE.search(src)
        results.append(text if text and not untranslated else None)
    return results


def translate_lines(
    texts: list[str],
    *,
    model: str,
    lang: str,
    batch_size: int,
    glossary: dict[str, str],
) -> tuple[list[str], int]:
    translated: list[str | None] = [None] * len(texts)
    failed_batches = 0

    def context_before(start: int) -> list[tuple[str, str]]:
        lo = max(0, start - CONTEXT_BEFORE)
        return [(texts[j], translated[j]) for j in range(lo, start) if translated[j]]

    def attempt(start: int, end: int, strict: bool, temperature: float = 0.2) -> None:
        try:
            results = request_batch(
                texts[start:end],
                context_before(start),
                texts[end : end + CONTEXT_AFTER],
                model=model,
                lang=lang,
                glossary=glossary,
                strict=strict,
                temperature=temperature,
            )
        except Exception as e:
            print(f"\n⚠️ 요청 실패 ({e})")
            return
        for offset, text in enumerate(results):
            if text:
                translated[start + offset] = text

    total = len(texts)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        attempt(start, end, strict=True)
        missing = [j for j in range(start, end) if translated[j] is None]
        for j in missing:
            attempt(j, j + 1, strict=False, temperature=0.5)
        still = sum(1 for j in missing if translated[j] is None)
        if still:
            failed_batches += 1
            print(f"\n{ERROR_MARK} {still}줄은 번역에 실패해 원문을 유지합니다.")
        print(f" 진행 중: [{end}/{total}] 완료", end="\r", flush=True)

    results: list[str] = []
    for text, src in zip(translated, texts):
        if text is None:
            results.append(src)
            continue
        if lang != "en" and KANA_RE.search(text) and KANA_RE.search(src):
            text = repair_kana(text)
        results.append(text)
    return results, failed_batches


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.input):
        print(f"❌ 오류: 원본 자막 파일을 찾을 수 없습니다: {args.input}")
        sys.exit(1)

    try:
        ollama.show(args.model)
    except Exception as e:
        print(f"❌ Ollama 모델을 사용할 수 없습니다 ({args.model}): {e}")
        print("💡 Ollama 앱을 실행하고 'ollama pull qwen3:8b'로 모델을 받아 주세요.")
        sys.exit(1)

    cues: list[Cue] = prepare_for_translation(args.input, args.lang, clean=not args.no_clean)
    glossary = load_glossary(args.glossary)
    texts = [c.text.replace("\n", " ") for c in cues]

    print(f"🚀 문맥 번역 시작: 자막 {len(texts)}개 · 모델 {args.model} · 묶음 {args.batch_size}줄")
    translated, failed = translate_lines(
        texts, model=args.model, lang=args.lang, batch_size=args.batch_size, glossary=glossary
    )

    for cue, text in zip(cues, translated):
        cue.text = wrap_lines(text, 22)
    save_cues(args.output, cues)

    print()
    if failed:
        print(f"⚠️ 일부 줄은 번역에 실패해 원문이 유지되었습니다 ({failed}곳).")
    print(f"🎉 문맥 번역 완료! 저장 위치: {args.output}")


if __name__ == "__main__":
    main()
