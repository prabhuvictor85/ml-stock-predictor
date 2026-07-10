#!/usr/bin/env python3
"""
Local token estimator — run before pasting compressed context into an LLM.
============================================================================
No network calls. Works out of the box with zero dependencies (heuristic
char/word blend); auto-upgrades to tiktoken's cl100k_base if installed
(pip install tiktoken) for a tighter, reproducible count.

Usage:
    python3 scripts/tools/count_tokens.py path/to/file.md
    python3 scripts/tools/count_tokens.py --stdin < notes.txt
    echo "some text" | python3 scripts/tools/count_tokens.py --stdin
"""
from __future__ import annotations
import argparse
import sys


def heuristic_estimate(text: str) -> dict:
    n_chars = len(text)
    n_words = len(text.split())
    # Dense/abbreviated/technical text tokenizes tighter than prose (more
    # digits, hyphens, symbols split into subtokens) -> use a band, not a
    # single number pretending to be exact.
    low  = round(n_chars / 4.5)   # optimistic (prose-like, long words)
    mid  = round(n_chars / 4.0)   # standard English-prose rule of thumb
    high = round(n_chars / 3.3)   # dense/symbolic/compressed notation
    word_based = round(n_words * 1.3)
    return {"chars": n_chars, "words": n_words,
           "low": low, "mid": mid, "high": high, "word_based": word_based}


def tiktoken_count(text: str) -> int | None:
    try:
        import tiktoken
    except ImportError:
        return None
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", help="file to count (omit with --stdin)")
    ap.add_argument("--stdin", action="store_true", help="read text from stdin")
    args = ap.parse_args()

    if args.stdin or args.path is None:
        text = sys.stdin.read()
    else:
        with open(args.path, "r", encoding="utf-8") as f:
            text = f.read()

    h = heuristic_estimate(text)
    tk = tiktoken_count(text)

    print(f"chars={h['chars']}  words={h['words']}")
    if tk is not None:
        print(f"tiktoken(cl100k_base) = {tk} tokens  <- most reliable available locally")
    else:
        print("tiktoken not installed (pip install tiktoken for an exact-ish count)")
    print(f"heuristic estimate: low={h['low']}  mid={h['mid']}  high={h['high']}  "
          f"word_based={h['word_based']}")
    best = tk if tk is not None else h["mid"]
    print(f"\nBEST GUESS: ~{best} tokens")


if __name__ == "__main__":
    main()
