---
title: "Appendix A · Prompt-engineering secret sauce"
chapter: appendix-a
slug: appendix-a-prompt-secrets
est_read_min: 11
---

# Appendix A · Prompt-engineering secret sauce

> The real reason LightRAG beats vanilla RAG and GraphRAG on the EMNLP'25 benchmarks is **not its code structure** — it's the prompt design. This appendix explains *why* those choices win, so you keep the win even if you re-implement in another language.

---

## 1. Why entity extraction is delimiter-based, not JSON

Upstream pickets extraction output as flat tuples separated by `<|#|>` and `<|>` — *not* JSON. First reaction is "wouldn't JSON be more structured?" But delimiter format has three LLM-era advantages:

1. **Shorter format-error tail**. LLMs occasionally drop a quote or add a stray comma, blowing up the entire `json.Unmarshal`. With delimiters the parser is line-oriented; one bad line drops one tuple — the other 99% still flow into the graph.
2. **`<|#|>` essentially never appears in natural text** — much more robust than `\n` or `,`. s09's parser is one `strings.Split`.
3. **Lower token cost**. JSON's `{"name": "X", "type": "PERSON", "description": "..."}` is 30+ tokens; the delimiter version `entity<|>X<|>PERSON<|>...` is ~12. Across hundreds of extraction calls per document that's a ~50% cost cut.

**Trade-off**: delimiter format can't use OpenAI's `response_format={"type": "json_object"}` strict mode. But LightRAG aims at *generic* LLM backends (OpenAI / Anthropic / Bedrock / Ollama) where strict mode is not universally available — delimiters are the more portable lowest common denominator.

s09's `extraction_prompt.go` keeps this design verbatim in spirit.

---

## 2. The gleaning loop: why N rounds beats one big prompt

Naive approach: one prompt asking the LLM for *all* entities at once. Problem: LLMs have "early-stop bias" on extraction — given a long output budget they still stop at entity 5-10 thinking "that's enough".

LightRAG's gleaning: round 1 extracts; we feed the result back and ask "any you missed? same format". Round 2 typically adds 30-60% new entities (per the paper's ablations). Round 3 sees diminishing returns.

**Why continuation beats one big prompt**:
- One big prompt suffers attention dilution over long inputs. Continuation framing concentrates the model on "what's left", not "what's there".
- Continuation cost is bounded by `MaxGleaningRounds` (s09 defaults to 1; research workloads set 2-3).

**Threshold**: past 3 rounds the new-entity count flatlines — you keep paying tokens for ~0 yield. s09's default `MaxGleaningRounds=1` is the economical pick; research mode can crank to 2-3.

---

## 3. The summarization threshold heuristic

s10's decision tree isn't fancy — it's the **avoid redundant LLM calls** engineering rule:

```
if len(descriptions) < 6 AND total_tokens < 500:
    return strings.Join(descriptions, "\n\n")  // free
else:
    return llm_summarize(descriptions)         // ~$0.001/call
```

**Where 6 comes from**: upstream's empirical finding that "≤5 related descriptions read fine concatenated; ≥6 starts repeating". s10 inherits the same threshold.

**Why LLM-merge instead of truncation** beyond threshold:
- Truncation drops information (the last two chunks might be the key evidence).
- Concatenating without merging blows tokens at query time, distracting the LLM during generation.
- LLM-merge compresses redundancy and preserves complementarity. This is the single biggest lever on context quality at query time.

**LLM-merge vs extractive summary**: upstream picks LLM-merge because entity descriptions often *conflict* ("Eleanor born 1834" vs "Eleanor born in 1832") — the merge prompt asks the model to flag conflicts rather than silently pick one. That's why LightRAG holds up on historical-figure corpora.

---

## 4. Why dual-level keywords trivialize mode routing

LightRAG's "high-level vs low-level keywords" is the core ablation in the EMNLP paper.

**high-level keywords**: macro themes / concepts. For "why was the Tarvin Highlands trip cancelled" the high-level set = `["explorations", "patronage", "fire incident"]`.
**low-level keywords**: concrete entities / proper nouns. Same query's low-level = `["Tarvin Highlands", "Scrooge", "Eleanor"]`.

Why splitting them makes mode routing trivial:
- **Local mode** = use low-level over vdbEntities → hits entity nodes by exact name.
- **Global mode** = use high-level over vdbRelations → hits cross-entity relations by theme word.
- **Hybrid mode** = run both + dedup.

Without the split, the same query would hit overlapping chunks in both indices — top-K is highly redundant — and the dual-level benefit collapses. With the split, the two sides cover different axes, and hybrid's dedup actually does work.

**Interesting failure case**: if the corpus is *only proper nouns* (a genealogy dataset), high-level keywords are hard to extract and global mode degrades to noise. LightRAG has no measurable edge there — that's its hidden boundary.

---

## 5. Cost-quality trade-off: when to crank gleaning

s09 defaults to `MaxGleaningRounds=1`. Full extraction over a 100-chunk document:
- 1 round: 100 LLM calls × 1 = 100 calls
- 2 rounds (gleaning=2): 100 × 3 = 300 calls
- 3 rounds (gleaning=3): 100 × 4 = 400 calls

At GPT-4o-mini pricing ($0.15/M input + $0.6/M output) one 100-chunk extraction is roughly $0.30. Cranking gleaning to 3 spikes to $1.20 — for ~10-15% recall improvement.

**Decision tree**:
- One-off research (medical / legal docs, every entry valuable) → gleaning = 2 or 3.
- Continuous ingest of mass documents (web crawl) → gleaning = 1, lean on coverage not recall.
- Real-time pipeline → default 1, with a separate batch re-ingest at gleaning=3 for high-value docs.

**The appendix's takeaway**: "LightRAG's code looks like a generic RAG framework", yet every prompt-design choice was tuned by cost-quality experimentation. Reading `prompt.py` gives you a deeper grasp of the project than reading `operate.py`.

---

## 6. What we deliberately omit

- **Reranker (cross-encoder)**: a slot between retrieval and LLM running an ms-marco-style reranker. Adds 5-10% recall but needs extra GPU; s11 leaves a hook, not implemented.
- **Streaming response**: s02's `CompleteRequest.Stream` field is reserved; the Phase G multi-model addendum only needs to flip `stream=true` on the OpenAI end and assemble SSE. Other backends do their own.
- **Self-RAG style citation verification**: have the LLM re-scan its output to check citations are in the retrieved set. Improves factuality but doubles cost — research mode only.

None of these are LightRAG's "secret"; they're general RAG enhancements. After reading Appendix A you should be able to judge which "advanced RAG tricks" are worth adding for your scenario.

---

**Back to the 11 chapters of code**: every design point this appendix discusses lives in s09's `extraction_prompt.go`, s10's `summarize_prompt.go`, and s11's `keywords.go`. When porting to another language, **port the prompts faithfully first**, then think about the code structure.
