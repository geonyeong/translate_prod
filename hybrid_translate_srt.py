#!/usr/bin/env python3
"""NLLB draft, then Ollama correction of only the lines that look wrong.

NLLB runs in a child process that exits before Ollama starts, so the draft
model’s GPU memory is returned to the OS. Ollama rewrites a
line only when the draft keeps source-language text, misses a glossary term, is
far too short for a long source, or repeats itself. A failed or timed-out
correction keeps the NLLB draft.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter

import httpx

from ollama_boot import ensure_ollama
from srt_cleanup import Cue, ends_sentence, prepare_for_translation, save_cues, wrap_lines
from translate_srt import (
    DEFAULT_GLOSSARY,
    ERROR_MARK,
    KANA_RE,
    OLLAMA_MODEL,
    load_glossary,
    repair_kana,
    request_batch,
)

NLLB_BATCH = 32
POLISH_BATCH = 8
POLISH_BEFORE = 1
POLISH_AFTER = 1
POLISH_NUM_CTX = 2048
BATCH_TIMEOUT = 60
LINE_TIMEOUT = 30
LATIN_RE = re.compile(r"[A-Za-z]{3,}")
HANGUL_RE = re.compile(r"[가-힣]")
SRC_CHAR_RE = re.compile(r"[A-Za-z0-9\u3040-\u30ff\u3400-\u9fff]")
REPEAT_RE = re.compile(r"(.{2,12}?)\1{2,}")
UNFINISHED_RE = re.compile(r"(?:은|는|이|가|을|를|의|와|과|에서|에게|한테|부터|까지)$")
LATIN_ALLOW = {"ok", "tv", "dvd", "cd", "bgm", "ng", "cm", "pv"}
# Short cues (はい, moans, cut-off fragments) stay as the NLLB draft.
MIN_CONTEXT_CHARS = 18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NLLB draft plus Ollama correction of weak lines.")
    parser.add_argument("--input", "-i", help="Source SRT path")
    parser.add_argument("--output", "-o", required=True, help="Translated SRT path")
    parser.add_argument(
        "--lang",
        "-l",
        choices=("1", "2", "ja", "en", "jpn", "eng"),
        default="ja",
        help="Source language: 1/ja/jpn=Japanese, 2/en/eng=English (default: ja)",
    )
    parser.add_argument(
        "--nllb-model",
        default=os.environ.get("NLLB_MODEL", "600m"),
        help="NLLB size: 600m (default), 1.3b, or a Hugging Face model id ($NLLB_MODEL)",
    )
    parser.add_argument("--nllb-batch", type=int, default=NLLB_BATCH, help=f"NLLB cues per batch (default: {NLLB_BATCH})")
    parser.add_argument("--model", default=OLLAMA_MODEL, help=f"Ollama model used for corrections (default: {OLLAMA_MODEL})")
    parser.add_argument("--batch-size", type=int, default=POLISH_BATCH, help=f"Lines per correction request (default: {POLISH_BATCH})")
    parser.add_argument(
        "--glossary",
        "-g",
        default=None,
        help=f"Glossary file ('원문 = 번역' per line). Default: {DEFAULT_GLOSSARY.name} if it exists",
    )
    parser.add_argument("--no-clean", action="store_true", help="Skip subtitle cleanup before translating")
    parser.add_argument("--draft-file", help=argparse.SUPPRESS)
    parser.add_argument("--write-draft", help=argparse.SUPPRESS)
    return parser.parse_args()


def core_len(text: str) -> int:
    return len(SRC_CHAR_RE.findall(text))


def revision_reasons(index: int, src: str, dst: str, texts: list[str], glossary: dict[str, str], lang: str) -> list[str]:
    """Reasons this draft should be rewritten. Empty means keep the NLLB line."""
    reasons: list[str] = []
    src_s, dst_s = src.strip(), dst.strip()
    dst_flat = " ".join(dst_s.split())

    if not dst_flat or dst_flat == src_s:
        reasons.append("미번역")

    if lang != "en" and KANA_RE.search(dst_s):
        reasons.append("일본어 잔존")
    if lang != "en":
        leftover = [w for w in LATIN_RE.findall(dst_s) if w.lower() not in LATIN_ALLOW]
        if leftover:
            reasons.append("외국어 잔존")

    for term, translated in glossary.items():
        if term and term in src_s and translated not in dst_s:
            reasons.append("용어 불일치")
            break

    src_len = core_len(src_s)
    if src_len >= 12 and len(HANGUL_RE.findall(dst_s)) < src_len * 0.3:
        reasons.append("과짧음")

    if src_len >= 8 and REPEAT_RE.search(dst_flat):
        reasons.append("반복")

    tail = dst_flat.rstrip(" .?!…。！？~")
    if src_len >= 12 and tail and UNFINISHED_RE.search(tail):
        reasons.append("미완결")

    if len(src_s) >= MIN_CONTEXT_CHARS and not ends_sentence(src_s):
        reasons.append("문장 단절")

    return reasons


def line_note(index: int, texts: list[str], drafts: list[str]) -> str:
    lines = [f"   draft: {drafts[index]}"]
    before = [
        f"{texts[j]} => {drafts[j]}"
        for j in range(max(0, index - POLISH_BEFORE), index)
    ]
    after = texts[index + 1 : index + 1 + POLISH_AFTER]
    if before:
        lines.append("   before: " + " / ".join(before))
    if after:
        lines.append("   after: " + " / ".join(after))
    return "\n".join(lines)


def _accept(text: str | None, src: str, lang: str) -> str | None:
    if not text:
        return None
    if lang != "en" and KANA_RE.search(text) and KANA_RE.search(src):
        text = repair_kana(text)
    return text


def polish(
    texts: list[str],
    drafts: list[str],
    indices: list[int],
    *,
    model: str,
    lang: str,
    glossary: dict[str, str],
    batch_size: int,
) -> tuple[list[str], int]:
    """Rewrite flagged lines. A failed or timed-out line keeps its NLLB draft."""
    revised = list(drafts)
    failed = 0
    done = 0
    total = len(indices)
    stopped = False
    print(f" 진행 중: [0/{total}] 완료", flush=True)

    def ask(group: list[int], timeout: float, strict: bool, temperature: float = 0.2) -> list[str | None] | None:
        """Return translations, or None when the request itself times out."""
        try:
            return request_batch(
                [texts[i] for i in group],
                [],
                [],
                model=model,
                lang=lang,
                glossary=glossary,
                strict=strict,
                temperature=temperature,
                notes=[line_note(i, texts, revised) for i in group],
                revise=True,
                num_ctx=POLISH_NUM_CTX,
                timeout=timeout,
            )
        except (httpx.TimeoutException, TimeoutError):
            return None
        except Exception as e:
            print(f"\n⚠️ 요청 실패 ({e})")
            return [None] * len(group)

    for start in range(0, total, batch_size):
        group = indices[start : start + batch_size]
        last = done + len(group)
        print(f"교정 요청 중: {done + 1}–{last}/{total}줄", flush=True)
        results = ask(group, BATCH_TIMEOUT, strict=True)
        if results is None:
            failed += total - done
            print(f"\n⚠️ 교정 요청이 {BATCH_TIMEOUT}초를 넘겼습니다. 남은 {total - done}줄은 NLLB 초안을 유지합니다.")
            stopped = True
            break

        for index, text in zip(group, results):
            fixed = _accept(text, texts[index], lang)
            if fixed is None:
                print(f"교정 요청 중: {done + 1}/{total}줄", flush=True)
                single = ask([index], LINE_TIMEOUT, strict=False, temperature=0.5)
                if single is None:
                    failed += total - done
                    print(f"\n⚠️ 한 줄 교정이 {LINE_TIMEOUT}초를 넘겼습니다. 남은 {total - done}줄은 NLLB 초안을 유지합니다.")
                    stopped = True
                    break
                fixed = _accept(single[0], texts[index], lang)
            if fixed:
                revised[index] = fixed
            else:
                failed += 1
            done += 1
            print(f" 진행 중: [{done}/{total}] 완료", end="\r", flush=True)
        if stopped:
            break

    return revised, failed


def _ollama_ready(model_name: str) -> bool:
    """Start Ollama if it is off. Return False when correction should be skipped."""
    if not ensure_ollama():
        return False
    try:
        import ollama

        ollama.show(model_name)
    except Exception as e:
        print(f"⚠️ Ollama 모델을 사용할 수 없습니다 ({model_name}): {e}")
        print("💡 'ollama pull qwen3:8b'로 모델을 받은 뒤 다시 실행하면 교정만 이어서 할 수 있습니다.")
        return False
    return True


def run_nllb_draft(texts: list[str], nllb_lang: str, model_key: str, batch_size: int, lang_name: str) -> list[str]:
    """Load NLLB and translate. The child process exits so this memory is released."""
    import gc

    import torch

    from import_srt import NLLB_MODELS, load_model, pick_device, translate_unique

    model_name = NLLB_MODELS.get(model_key.lower(), model_key)
    device = pick_device()
    print(f"⏳ NLLB 초안 모델({model_name})을 로드하고 있습니다...", flush=True)
    tokenizer, model = load_model(model_name, nllb_lang, device)
    print(f"🚀 NLLB 초안 번역 (원문: {lang_name}) · 자막 {len(texts)}개 (고유 {len(set(texts))}개)", flush=True)
    draft_map = translate_unique(texts, tokenizer, model, device, max(1, batch_size))
    del model, tokenizer
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return [draft_map.get(text, text) for text in texts]


def write_draft_file(args: argparse.Namespace, path: str) -> None:
    from import_srt import resolve_src_lang

    nllb_lang, lang_name, lang_code = resolve_src_lang(args.lang)
    cues = prepare_for_translation(args.input, lang_code, clean=not args.no_clean)
    texts = [" ".join(cue.text.split()) for cue in cues]
    for cue, text in zip(cues, texts):
        cue.text = text
    drafts = run_nllb_draft(texts, nllb_lang, args.nllb_model, args.nllb_batch, lang_name)
    payload = {
        "lang": lang_code,
        "cues": [{"start": cue.start, "end": cue.end, "text": cue.text} for cue in cues],
        "drafts": drafts,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    sys.stdout.flush()


def main() -> None:
    args = parse_args()
    if args.write_draft:
        if not args.input or not os.path.exists(args.input):
            print(f"❌ 오류: 파일 '{args.input}'이 없습니다. 경로를 확인해주세요.")
            sys.exit(1)
        write_draft_file(args, args.write_draft)
        return
    if args.draft_file:
        polish_saved_draft(args)
        return
    if not args.input or not os.path.exists(args.input):
        print(f"❌ 오류: 파일 '{args.input}'이 없습니다. 경로를 확인해주세요.")
        sys.exit(1)

    fd, path = tempfile.mkstemp(prefix="hybrid_draft_", suffix=".json")
    os.close(fd)
    command = [
        sys.executable,
        os.path.abspath(__file__),
        "--input",
        args.input,
        "--output",
        args.output,
        "--write-draft",
        path,
        "--nllb-model",
        args.nllb_model,
        "--nllb-batch",
        str(args.nllb_batch),
        "--lang",
        args.lang,
    ]
    if args.no_clean:
        command.append("--no-clean")
    completed = subprocess.run(command)
    if completed.returncode != 0:
        if os.path.exists(path):
            os.remove(path)
        sys.exit(completed.returncode)
    args.draft_file = path
    polish_saved_draft(args)


def polish_saved_draft(args: argparse.Namespace) -> None:
    with open(args.draft_file, encoding="utf-8") as handle:
        payload = json.load(handle)
    os.remove(args.draft_file)
    lang_code = payload["lang"]
    cues = [Cue(item["start"], item["end"], item["text"]) for item in payload["cues"]]
    texts = [cue.text for cue in cues]
    drafts = list(payload["drafts"])
    glossary = load_glossary(args.glossary)

    flagged: list[int] = []
    counts: Counter[str] = Counter()
    for i, (src, dst) in enumerate(zip(texts, drafts)):
        reasons = revision_reasons(i, src, dst, texts, glossary, lang_code)
        if reasons:
            flagged.append(i)
            counts.update(reasons)

    print()
    if not flagged:
        print(f"✨ 교정할 줄이 없습니다. NLLB 초안 {len(texts)}줄을 그대로 저장합니다.")
        polished, failed = drafts, 0
    elif not _ollama_ready(args.model):
        print("⚠️ 교정을 건너뛰고 NLLB 초안을 저장합니다.")
        polished, failed = drafts, 0
    else:
        detail = ", ".join(f"{name} {count}" for name, count in counts.most_common())
        print(f"🔧 교정 대상 {len(flagged)}/{len(texts)}줄 ({detail})")
        print(f"🚀 Ollama 교정 시작 · 모델 {args.model} · 묶음 {args.batch_size}줄")
        polished, failed = polish(
            texts,
            drafts,
            flagged,
            model=args.model,
            lang=lang_code,
            glossary=glossary,
            batch_size=max(1, args.batch_size),
        )

    for cue, text in zip(cues, polished):
        cue.text = wrap_lines(text, 22)
    save_cues(args.output, cues)

    print()
    if failed:
        print(f"{ERROR_MARK} {failed}줄은 교정에 실패해 NLLB 초안을 유지합니다.")
    print(f"🎉 초안+교정 번역 완료! 저장 위치: {args.output}")


if __name__ == "__main__":
    main()
