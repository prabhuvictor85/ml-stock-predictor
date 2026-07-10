# CLAUDE.md — ml-stock-predictor

Project-level conventions for Claude Code sessions on this repo.

## LLM-memory context compression

When asked to compress conversation context / findings into a dense block for
future LLM ingestion (e.g. cross-session memory, handoff notes), apply these
transformations:

- Remove stop words and prose narration.
- Remove duplicate references (state a fact once; link back to it, don't repeat).
- Replace common terms with fixed abbreviations (define once at top if non-obvious).
- Use `key=value` / `key:value` chains over sentences.
- Collapse repeated entities into one canonical mention.
- Preserve exactly: facts, dates, metrics (with units/sign), IDs (commit hashes,
  ticket/file names), and relationships (X caused Y, A supersedes B).
- No natural-language connective tissue ("this means that...", "as a result...").
- Output as a single compact block, not multiple paragraphs.

**Known tradeoff (measured empirically 2026-07-10):** this style minimizes
*characters*, not necessarily *tokens*. BPE tokenizers (GPT/Claude-family)
give common English words a single token; invented abbreviations, hyphen-chains,
and symbol-dense shorthand often split into MORE subtokens than plain words
would. A 5,137-char compressed block measured at 1,990 tokens via tiktoken —
higher than a naive chars/4 estimate (1,284) would suggest. **Always verify
with the counter below rather than assuming compression = fewer tokens.**

## Token counting before sending context to an LLM

Use `scripts/tools/count_tokens.py` — zero-dependency by default, upgrades
automatically to `tiktoken` (cl100k_base) if installed for a tighter count.

```bash
# from a file
python scripts/tools/count_tokens.py path/to/context.txt

# from stdin (e.g. piped from clipboard or another command)
python scripts/tools/count_tokens.py --stdin < notes.txt
echo "some text" | python scripts/tools/count_tokens.py --stdin
```

Output gives `chars`, `words`, a `tiktoken` count if the library is present,
and a heuristic low/mid/high band as fallback. Treat the tiktoken number as
the reliable one when available (Claude's real tokenizer isn't locally
available, but cl100k_base is typically within ~10-15% of it). Install once
if you want the accurate path every time:

```bash
pip install tiktoken
```

**Workflow:** write the compressed block to a temp file (or pipe it via
`--stdin`) → run the counter → report the `tiktoken` line (or the `mid`
heuristic if tiktoken isn't installed) as the token count, not a guess.

## Other project tools (scripts/tools/)

- `validate_watchlist_forward.py` — independent, read-only watchlist
  forward-return grader. Downloads/caches prices once per unique symbol,
  grades against a benchmark, never discards near-cutoff records (uses
  available-window fallback). See file docstring for the near-cutoff
  metric caveat (avail-window mixes variable hold lengths — use the
  fixed-horizon columns for trustworthy comparisons).
- `compare_watchlist_series.py` — head-to-head grader for N labeled
  watchlist series (e.g. causal vs frozen-recipe, pure-ML vs composite)
  on identical forward-return logic.

See `PROTOCOL.md` for the experiment ledger and one-shot lockbox rules,
and `docs/MODEL_*_PREREGISTRATION.md` for frozen pre-run experiment specs.
