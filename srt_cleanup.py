#!/usr/bin/env python3
"""Subtitle cleanup shared by the STT and translation steps.

- ASR JSON -> SRT: mlx_qwen3_asr's own SRT writer drops punctuation and cuts a cue
  every 10 word tokens, often mid-sentence or even mid-word. Its JSON output keeps
  the punctuated transcript plus word timestamps, so cues are rebuilt per sentence.
- clean: collapse hallucinated repeats, drop filler/noise-only cues, merge duplicate
  and fragment cues so translators receive whole sentences.

Usage:
  python srt_cleanup.py --input result/video.json --output result/video.srt --lang ja
  python srt_cleanup.py --input old.srt --output old_clean.srt
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import timedelta

import srt

CJK_LANGS = {"ja", "jpn", "japanese", "zh", "chinese", "cantonese", "yue", "ko", "korean"}
SENTENCE_END = "。！？.!?…‥"
SOFT_BREAK = "、，,;；:："
OPENING = "「『（(［[｛{〈《“‘"

FILLER_CHARS_CJK = set("あぁアァうぅウゥえぇエェおぉオォんンっッーはハふフ〜~")
FILLER_KEEP_CJK = {"うん", "ううん", "うんうん", "ええ", "おう", "はあ?", "ふーん"}
FILLER_WORDS_LATIN = {"uh", "um", "umm", "uhm", "hmm", "hm", "mm", "mmm", "mhm", "ah", "oh", "er", "eh", "huh", "ha"}

PUNCT_RE = re.compile(r"[\W_]+", re.UNICODE)
CJK_CHAR_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")


@dataclass
class Limits:
    max_chars: int
    max_duration: float = 7.0
    min_duration: float = 0.7
    sentence_gap: float = 1.0
    merge_gap: float = 0.5
    short_chars: int = 5
    short_duration: float = 1.0
    linger: float = 0.3

    @classmethod
    def for_text(cls, cjk: bool) -> "Limits":
        return cls(max_chars=30, short_chars=5) if cjk else cls(max_chars=84, short_chars=12)


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class Cue:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list, repr=False)


@dataclass
class Stats:
    before: int = 0
    after: int = 0
    repeats: int = 0
    noise: int = 0
    duplicates: int = 0
    fragments: int = 0

    def summary(self) -> str:
        parts = [
            f"반복 축약 {self.repeats}",
            f"잡음 제거 {self.noise}",
            f"중복 병합 {self.duplicates}",
            f"조각 병합 {self.fragments}",
        ]
        return f"자막 {self.before}개 → {self.after}개 ({', '.join(parts)})"


# ---------- language helpers ----------
def is_cjk(lang: str | None, sample: str = "") -> bool:
    if lang and lang.strip().lower() in CJK_LANGS:
        return True
    if lang and lang.strip().lower() not in {"", "auto"}:
        return False
    letters = [c for c in sample if c.isalpha()]
    if not letters:
        return False
    return sum(1 for c in letters if CJK_CHAR_RE.match(c)) / len(letters) > 0.3


def join_parts(parts: list[str], cjk: bool) -> str:
    parts = [p.strip() for p in parts if p and p.strip()]
    if cjk:
        return "".join(parts)
    text = " ".join(parts)
    return re.sub(r"\s+([,.;:!?…])", r"\1", text)


def ends_sentence(text: str) -> bool:
    return bool(text) and text.rstrip(" 」』）)\"'”’")[-1:] in SENTENCE_END


def wrap_lines(text: str, width: int) -> str:
    """Break a long single-line subtitle into two lines at the space nearest the middle."""
    text = text.strip()
    if "\n" in text or len(text) <= width:
        return text
    spaces = [m.start() for m in re.finditer(r" ", text)]
    if not spaces:
        return text
    mid = len(text) / 2
    cut = min(spaces, key=lambda i: abs(i - mid))
    return f"{text[:cut].strip()}\n{text[cut + 1:].strip()}"


# ---------- SRT I/O ----------
def load_cues(path: str) -> list[Cue]:
    with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
        subtitles = list(srt.parse(f.read()))
    return [Cue(s.start.total_seconds(), s.end.total_seconds(), s.content.strip()) for s in subtitles]


def save_cues(path: str, cues: list[Cue]) -> None:
    subtitles = [
        srt.Subtitle(
            index=i,
            start=timedelta(seconds=round(c.start, 3)),
            end=timedelta(seconds=round(c.end, 3)),
            content=c.text,
        )
        for i, c in enumerate(cues, 1)
    ]
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(srt.compose(subtitles, reindex=False))


# ---------- ASR JSON -> cues ----------
def attach_punctuation(text: str, segments: list[dict]) -> list[Word]:
    """Map aligner tokens (no punctuation) back onto the punctuated transcript."""
    words: list[Word] = []
    pos = 0
    prev_end = 0

    def flush_gap(gap: str) -> str:
        # Closing punctuation belongs to the previous word, opening brackets to the next.
        split = next((i for i, ch in enumerate(gap) if ch in OPENING), len(gap))
        closing, opening = gap[:split], gap[split:]
        if words and closing.strip():
            words[-1].text += closing.strip()
        return opening.strip()

    for seg in segments:
        token = str(seg.get("text", "")).strip()
        start = float(seg.get("start", 0.0))
        end = max(float(seg.get("end", start)), start)
        idx = text.find(token, pos) if token else -1
        if idx == -1 or idx - pos > 40:
            words.append(Word(token, start, end))
            continue
        prefix = flush_gap(text[prev_end:idx]) if words else text[:idx].strip()
        words.append(Word(prefix + token, start, end))
        pos = prev_end = idx + len(token)

    if words:
        flush_gap(text[prev_end:])
    return words


def _cue_from_words(words: list[Word], cjk: bool) -> Cue:
    return Cue(words[0].start, words[-1].end, join_parts([w.text for w in words], cjk), list(words))


def _split_long(words: list[Word], cjk: bool, limits: Limits) -> list[list[Word]]:
    text = join_parts([w.text for w in words], cjk)
    duration = words[-1].end - words[0].start
    if len(words) <= 1 or (len(text) <= limits.max_chars and duration <= limits.max_duration):
        return [words]

    total = len(text)
    best_k, best_cost = 1, float("inf")
    running = 0
    for k in range(1, len(words)):
        running += len(words[k - 1].text) + (0 if cjk else 1)
        cost = abs(running - total / 2) / total
        if words[k - 1].text[-1:] in SOFT_BREAK + SENTENCE_END:
            cost -= 0.25
        cost -= min(words[k].start - words[k - 1].end, 1.0) * 0.3
        if cost < best_cost:
            best_k, best_cost = k, cost
    return _split_long(words[:best_k], cjk, limits) + _split_long(words[best_k:], cjk, limits)


def _is_short(text: str, duration: float, limits: Limits) -> bool:
    return len(text) < limits.short_chars or duration < limits.short_duration


def _is_fragment(text: str, duration: float, limits: Limits) -> bool:
    # Without punctuation a short reply ("はい") and a cut-off word tail ("たか") look alike;
    # only treat very short cues in both length and time as fragments.
    return len(text) < limits.short_chars and duration < limits.short_duration / 2


def _merge_short(groups: list[list[Word]], cjk: bool, limits: Limits) -> list[list[Word]]:
    merged: list[list[Word]] = []
    for group in groups:
        if merged:
            prev = merged[-1]
            prev_text = join_parts([w.text for w in prev], cjk)
            cur_text = join_parts([w.text for w in group], cjk)
            combined = join_parts([prev_text, cur_text], cjk)
            if (
                (
                    _is_short(prev_text, prev[-1].end - prev[0].start, limits)
                    or _is_short(cur_text, group[-1].end - group[0].start, limits)
                )
                and group[0].start - prev[-1].end <= limits.merge_gap
                and len(combined) <= limits.max_chars
                and group[-1].end - prev[0].start <= limits.max_duration
            ):
                merged[-1] = prev + group
                continue
        merged.append(group)
    return merged


def build_from_asr_json(data: dict, lang: str | None = None) -> list[Cue]:
    text = re.sub(r"[\u3000\s]+", " ", str(data.get("text", ""))).strip()
    segments = [s for s in data.get("segments") or [] if str(s.get("text", "")).strip()]
    if not segments:
        return []
    cjk = is_cjk(data.get("language") or lang, text)
    limits = Limits.for_text(cjk)

    words = attach_punctuation(text, segments)
    sentences: list[list[Word]] = []
    current: list[Word] = []
    for word in words:
        if current and word.start - current[-1].end >= limits.sentence_gap:
            sentences.append(current)
            current = []
        current.append(word)
        if ends_sentence(word.text):
            sentences.append(current)
            current = []
    if current:
        sentences.append(current)

    groups: list[list[Word]] = []
    for sentence in sentences:
        groups.extend(_split_long(sentence, cjk, limits))
    groups = _merge_short(groups, cjk, limits)
    cues = [_cue_from_words(g, cjk) for g in groups]
    # Word timestamps end exactly when speech stops; linger only here so re-cleaning an SRT is idempotent.
    _fix_timing(cues, limits, linger=limits.linger)
    return cues


# ---------- cleanup ----------
def _collapse_repeats(text: str, cjk: bool) -> tuple[str, bool]:
    original = text
    text = re.sub(r"(.)\1{4,}", r"\1\1\1", text)
    sep = "" if cjk else " "
    text = re.sub(r"(.{2,20}?)(?:[\s、,]*\1){3,}", lambda m: m.group(1) + sep + m.group(1), text)
    return text, text != original


def _is_noise(text: str, cjk: bool) -> bool:
    core = PUNCT_RE.sub("", text)
    if not core:
        return True
    if cjk:
        return core not in FILLER_KEEP_CJK and all(ch in FILLER_CHARS_CJK for ch in core)
    tokens = [PUNCT_RE.sub("", t).lower() for t in text.split()]
    tokens = [t for t in tokens if t]
    return bool(tokens) and all(t in FILLER_WORDS_LATIN for t in tokens)


def _norm_key(text: str) -> str:
    return PUNCT_RE.sub("", text).lower()


def clean_cues(cues: list[Cue], lang: str | None = None) -> tuple[list[Cue], Stats]:
    stats = Stats(before=len(cues))
    sample = " ".join(c.text for c in cues[:200])
    cjk = is_cjk(lang, sample)
    limits = Limits.for_text(cjk)

    kept: list[Cue] = []
    for cue in cues:
        text = join_parts(cue.text.split("\n"), cjk)
        text, changed = _collapse_repeats(text, cjk)
        stats.repeats += changed
        if _is_noise(text, cjk):
            stats.noise += 1
            continue
        cue = Cue(cue.start, max(cue.end, cue.start), text, cue.words)

        if kept:
            prev = kept[-1]
            gap = cue.start - prev.end
            if _norm_key(prev.text) == _norm_key(text) and gap < 1.5:
                prev.end = max(prev.end, cue.end)
                stats.duplicates += 1
                continue
            combined = join_parts([prev.text, text], cjk)
            if (
                not ends_sentence(prev.text)
                and (
                    _is_fragment(prev.text, prev.end - prev.start, limits)
                    or _is_fragment(text, cue.end - cue.start, limits)
                )
                and gap <= 0.3
                and len(combined) <= limits.max_chars
                and cue.end - prev.start <= limits.max_duration
            ):
                prev.text, prev.end = combined, cue.end
                prev.words = prev.words + cue.words
                stats.fragments += 1
                continue
        kept.append(cue)

    _fix_timing(kept, limits)
    if not cjk:
        for cue in kept:
            cue.text = wrap_lines(cue.text, 42)
    stats.after = len(kept)
    return kept, stats


def _fix_timing(cues: list[Cue], limits: Limits, linger: float = 0.0) -> None:
    """Enforce a minimum display time; linger keeps text on screen a bit after speech ends."""
    for i, cue in enumerate(cues):
        next_start = cues[i + 1].start - 0.05 if i + 1 < len(cues) else float("inf")
        target = max(cue.end + linger, cue.start + limits.min_duration)
        cue.end = max(cue.end, min(target, next_start))


def prepare_for_translation(path: str, lang: str | None, clean: bool = True) -> list[Cue]:
    """Load an SRT for translation, cleaning it first unless disabled."""
    cues = load_cues(path)
    if not clean:
        return [c for c in cues if c.text.strip()]
    cues, stats = clean_cues(cues, lang)
    print(f"🧹 자막 정리: {stats.summary()}")
    return cues


# ---------- CLI ----------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild/clean subtitles (ASR JSON or SRT -> SRT).")
    parser.add_argument("--input", "-i", required=True, help="mlx_qwen3_asr JSON (with --timestamps) or SRT")
    parser.add_argument("--output", "-o", required=True, help="Output SRT path")
    parser.add_argument("--lang", "-l", default="auto", help="Source language hint: ja, en, auto (default)")
    parser.add_argument("--no-clean", action="store_true", help="Skip repeat/noise/fragment cleanup")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.input):
        print(f"❌ 오류: 파일 '{args.input}'이 없습니다.")
        sys.exit(1)

    if args.input.lower().endswith(".json"):
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)
        cues = build_from_asr_json(data, args.lang)
        if not cues:
            print("❌ 오류: JSON에 타임스탬프(segments)가 없습니다. --timestamps 옵션으로 다시 실행하세요.")
            sys.exit(1)
        print(f"🧩 문장 단위 자막 재구성: {len(cues)}개 (언어: {data.get('language') or args.lang})")
    else:
        cues = load_cues(args.input)

    if not args.no_clean:
        cues, stats = clean_cues(cues, args.lang)
        print(f"🧹 자막 정리: {stats.summary()}")

    save_cues(args.output, cues)
    print(f"💾 저장: {args.output}")


if __name__ == "__main__":
    main()
