#!/usr/bin/env python3
"""Build a glossary from multiple source SRT + Korean SRT pairs.

Scans aligned cue pairs for repeated Japanese/English terms, groups the Korean
lines they appear in, and suggests Hangul translations. Conflicting suggestions
across files are ranked higher so a human can pick one mapping.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from srt_cleanup import load_cues
from translate_srt import DEFAULT_GLOSSARY, load_glossary

ROOT = Path(__file__).resolve().parent

# Japanese name / term shapes (avoid single kana/kanji noise).
JA_TERM_RE = re.compile(
    r"[\u30a0-\u30ffー]{2,}"  # katakana
    r"|[\u4e00-\u9fff]{2,}"  # kanji run
    r"|[\u4e00-\u9fff]+[\u3040-\u30ff]+"  # kanji + kana
    r"|[\u30a0-\u30ffー]+[\u4e00-\u9fff]+"  # kana + kanji
)
EN_TERM_RE = re.compile(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b")
HANGUL_TOKEN_RE = re.compile(r"[가-힣]{2,}")
SPACE_RE = re.compile(r"\s+")
# Verb/adjective endings and polite scraps that are rarely glossary values.
KO_STOP = {
    "입니다", "습니다", "하세요", "하세요서", "합니다", "해요", "이에요", "예요",
    "었어", "았어", "겠다", "네요", "군요", "거든", "는데", "니까", "면서",
    "오늘도", "잠깐", "씨도", "올래", "불러", "부탁", "부탁드립니다", "수고하셨습니다",
}

# Common particles / fillers that are poor glossary keys.
JA_STOP = {
    "こと", "もの", "ため", "とき", "ところ", "よう", "そう", "これ", "それ", "あれ",
    "ここ", "そこ", "あそこ", "なに", "なん", "だれ", "どこ", "いつ", "どう",
    "今日", "来る", "来て", "疲れ", "様です", "願いします", "呼んでるよ",
}
EN_STOP = {
    "I", "I'm", "I'll", "I've", "I'd", "You", "You're", "We", "We're", "They",
    "He", "She", "It", "The", "A", "An", "And", "Or", "But", "So", "If", "OK",
}


@dataclass
class Occurrence:
    file: str
    index: int
    src: str
    kor: str


@dataclass
class Candidate:
    term: str
    suggested: str
    count: int
    variants: list[str]
    inconsistent: bool
    occurrences: list[Occurrence] = field(default_factory=list)

    @property
    def score(self) -> tuple[int, int, int]:
        # Sort: inconsistent first, then more variants, then more hits.
        return (1 if self.inconsistent else 0, len(self.variants), self.count)


@dataclass
class PairData:
    stem: str
    src_path: Path
    kor_path: Path
    src_lines: list[str]
    kor_lines: list[str]


def flatten(text: str) -> str:
    return SPACE_RE.sub(" ", text.replace("\n", " ")).strip()


def find_pairs(paths: list[Path]) -> list[tuple[Path, Path]]:
    """Resolve (source, korean) pairs from files or directories."""
    files: list[Path] = []
    for path in paths:
        path = path.expanduser().resolve()
        if path.is_dir():
            files.extend(sorted(path.glob("*.srt")))
        elif path.is_file() and path.suffix.lower() == ".srt":
            files.append(path)

    by_name = {f.name: f for f in files}
    pairs: list[tuple[Path, Path]] = []
    seen: set[Path] = set()
    for path in files:
        name = path.name
        if name.endswith("_KOR.srt"):
            continue
        kor = path.with_name(f"{path.stem}_KOR.srt")
        # Also accept sibling already in the list under another folder name.
        kor = by_name.get(kor.name, kor)
        if kor.is_file() and path not in seen:
            pairs.append((path, kor))
            seen.add(path)
    return pairs


def load_pair(src_path: Path, kor_path: Path) -> PairData:
    src_cues = load_cues(str(src_path))
    kor_cues = load_cues(str(kor_path))
    n = min(len(src_cues), len(kor_cues))
    return PairData(
        stem=src_path.stem,
        src_path=src_path,
        kor_path=kor_path,
        src_lines=[flatten(c.text) for c in src_cues[:n]],
        kor_lines=[flatten(c.text) for c in kor_cues[:n]],
    )


def extract_terms(text: str, lang: str) -> list[str]:
    found: list[str] = []
    if lang in ("ja", "auto"):
        for match in JA_TERM_RE.finditer(text):
            term = match.group(0)
            if term not in JA_STOP and len(term) >= 2:
                found.append(term)
    if lang in ("en", "auto"):
        for match in EN_TERM_RE.finditer(text):
            term = match.group(0)
            if term not in EN_STOP and len(term) >= 2:
                found.append(term)
    # Prefer longer terms first when overlapping; keep unique order.
    found.sort(key=len, reverse=True)
    unique: list[str] = []
    for term in found:
        if any(term != other and term in other for other in unique):
            continue
        if term not in unique:
            unique.append(term)
    return unique


def hangul_tokens(text: str) -> list[str]:
    return HANGUL_TOKEN_RE.findall(text)


def _token_score(term: str, token: str, df: int, n_lines: int) -> float:
    if token in KO_STOP or len(token) < 2:
        return -1.0
    # Names are usually short Hangul; long polite phrases are weak glossary values.
    length_penalty = 0.0 if len(token) <= 6 else (len(token) - 6) * 0.8
    if token.endswith(("습니다", "합니다", "입니다", "세요", "네요")):
        length_penalty += 2.5
    # Prefer tokens that appear in many of this term's Korean lines.
    coverage = df / max(1, n_lines)
    # Rough length fit vs Japanese term (kanji≈1–2 Hangul syllables each).
    expected = max(2, min(6, len(term)))
    fit = 1.0 - min(1.0, abs(len(token) - expected) / 6)
    return coverage * 3.0 + fit + df * 0.15 - length_penalty


def suggest_translation(term: str, kor_lines: list[str]) -> tuple[str, list[str], bool]:
    """Pick the Hangul token that co-occurs most with this term's Korean lines."""
    df: Counter[str] = Counter()
    for line in kor_lines:
        for token in set(hangul_tokens(line)):
            df[token] += 1
    if not df:
        return "", [], False
    n_lines = len(kor_lines)
    ranked = sorted(
        df.keys(),
        key=lambda token: (_token_score(term, token, df[token], n_lines), df[token], -len(token)),
        reverse=True,
    )
    ranked = [t for t in ranked if _token_score(term, t, df[t], n_lines) >= 0]
    if not ranked:
        return "", [], False
    top = ranked[0]
    # Competing name-like translations with similar coverage.
    rivals = []
    for token in ranked[1:8]:
        if df[token] < max(1, df[top] * 0.4):
            continue
        if abs(len(token) - len(top)) > 4 and len(token) > 6:
            continue
        rivals.append(token)
    inconsistent = len(rivals) > 0 and df[top] < n_lines
    # Also mark inconsistent when two short name-like tokens both appear often.
    short_rivals = [t for t in [top] + rivals if 2 <= len(t) <= 6]
    if len(set(short_rivals)) >= 2 and n_lines >= 2:
        inconsistent = True
    ordered: list[str] = []
    for item in [top] + rivals:
        if item not in ordered:
            ordered.append(item)
    return top, ordered, inconsistent


def build_candidates(
    pairs: list[PairData],
    *,
    lang: str = "ja",
    min_count: int = 2,
    existing: dict[str, str] | None = None,
) -> list[Candidate]:
    existing = existing or {}
    buckets: dict[str, list[Occurrence]] = defaultdict(list)

    for pair in pairs:
        for index, (src, kor) in enumerate(zip(pair.src_lines, pair.kor_lines)):
            if not src:
                continue
            for term in extract_terms(src, lang):
                buckets[term].append(Occurrence(pair.stem, index + 1, src, kor))

    candidates: list[Candidate] = []
    for term, occurrences in buckets.items():
        if term in JA_STOP or term in EN_STOP:
            continue
        if len(occurrences) < min_count and term not in existing:
            # Keep single-hit proper-name shapes (katakana / multi-kanji).
            if not (
                re.fullmatch(r"[\u30a0-\u30ffー]{3,}", term)
                or re.fullmatch(r"[\u4e00-\u9fff]{2,}", term)
            ):
                continue
        kor_lines = [item.kor for item in occurrences]
        suggested, variants, inconsistent = suggest_translation(term, kor_lines)
        if term in existing:
            suggested = existing[term]
            if existing[term] not in variants:
                variants = [existing[term]] + variants
        if not suggested and not variants:
            continue
        # Drop weak single-occurrence noise unless it looks inconsistent or is in glossary.
        if len(occurrences) == 1 and not inconsistent and term not in existing:
            if len(term) < 3:
                continue
        candidates.append(
            Candidate(
                term=term,
                suggested=suggested,
                count=len(occurrences),
                variants=variants or ([suggested] if suggested else []),
                inconsistent=inconsistent or (term in existing and any(v != existing[term] for v in variants[:3])),
                occurrences=occurrences,
            )
        )

    candidates.sort(key=lambda c: (-c.score[0], -c.score[1], -c.score[2], c.term))
    return candidates


def format_glossary_lines(entries: dict[str, str]) -> str:
    lines = [f"{src} = {dst}" for src, dst in sorted(entries.items(), key=lambda item: item[0])]
    return "\n".join(lines) + ("\n" if lines else "")


def merge_glossary(path: Path, entries: dict[str, str], *, overwrite: bool = True) -> dict[str, str]:
    """Merge entries into a glossary file. Returns the full resulting map."""
    current = load_glossary(str(path)) if path.is_file() else {}
    for src, dst in entries.items():
        if not src or not dst:
            continue
        if src in current and not overwrite:
            continue
        current[src] = dst
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# 용어집 (glossary_builder / GUI에서 생성)\n"
        "# 형식: 원문 = 번역\n"
    )
    body = format_glossary_lines(current)
    path.write_text(header + body, encoding="utf-8")
    return current


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Suggest glossary entries from source + Korean SRT pairs.")
    parser.add_argument("paths", nargs="+", help="SRT files or folders that contain *.srt / *_KOR.srt pairs")
    parser.add_argument("--lang", choices=("ja", "en", "auto"), default="ja")
    parser.add_argument("--min-count", type=int, default=2, help="Minimum occurrences to list (default: 2)")
    parser.add_argument("--glossary", "-g", default=str(DEFAULT_GLOSSARY), help="Existing glossary to merge suggestions into")
    parser.add_argument("--write", action="store_true", help="Write top suggestions (inconsistent + frequent) into the glossary")
    parser.add_argument("--limit", type=int, default=40, help="How many candidates to print")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pair_paths = find_pairs([Path(p) for p in args.paths])
    if not pair_paths:
        print("❌ 원문.srt + 원문_KOR.srt 쌍을 찾지 못했습니다.")
        raise SystemExit(1)
    pairs = [load_pair(src, kor) for src, kor in pair_paths]
    print(f"📂 자막 쌍 {len(pairs)}개 로드")
    existing = load_glossary(args.glossary)
    candidates = build_candidates(pairs, lang=args.lang, min_count=args.min_count, existing=existing)
    print(f"🔎 후보 {len(candidates)}개 (표시 {min(args.limit, len(candidates))}개)\n")
    for item in candidates[: args.limit]:
        mark = "⚠" if item.inconsistent else "·"
        variants = ", ".join(item.variants[:4]) if item.variants else "-"
        print(f"{mark} {item.term}  →  {item.suggested or '?'}  ({item.count}회 · {variants})")
    if args.write:
        chosen = {c.term: c.suggested for c in candidates if c.suggested and (c.inconsistent or c.count >= max(3, args.min_count))}
        if not chosen:
            print("저장할 확정 후보가 없습니다.")
            return
        merged = merge_glossary(Path(args.glossary), chosen, overwrite=True)
        print(f"\n💾 용어집 저장: {args.glossary} ({len(merged)}항목, 이번에 {len(chosen)}개)")


if __name__ == "__main__":
    main()
