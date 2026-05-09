# Quality report: learn-lightrag

Generated: 2026-05-09T14:20:41Z
Repo: https://github.com/Ding-Ye/learn-lightrag
Total commits: 14
CI status: last 10 runs success=10 failure=0 cancelled=0 (all green)

## Summary

- P0 issues: 0
- P1 issues: 1
- P2 issues: 1

## P0 issues (must fix)

None — all P0 checks passed.

Detail of each P0 check:

- **P0.1 Bilingual heading parity** — every `docs/zh/sNN-*.md` matched its `docs/en/sNN-*.md` `##` count exactly (output of the loop was empty, no `P0 PARITY:` lines).
- **P0.2 Six-section spine** — every `docs/{zh,en}/sNN-*.md` (excluding s_full / appendix-* / multi-model) contains all six required headings (Problem, Solution, How It Works, What Changed, Try It, Upstream Source). The full sweep produced no `P0 SPINE:` output.
- **P0.3 No cross-session imports** — each `agents/sNN/` module only imports from its own `learn-lightrag/sNN` module path. Sweep produced no `P0 CROSS-IMPORT:` output.
- **P0.4 Upstream citation realism** — spot-checked 5 random `lightrag/<file>:<line>` references with `curl` against `raw.githubusercontent.com/HKUDS/LightRAG/main/`. All five returned HTTP 200 and the cited line was within range:
  ```
  lightrag/base.py:759: HTTP=200 total_lines=960 cited=759 OK
  lightrag/lightrag.py:2884: HTTP=200 total_lines=4507 cited=2884 OK
  lightrag/operate.py:3164: HTTP=200 total_lines=5197 cited=3164 OK
  lightrag/kg/nano_vector_db_impl.py:245: HTTP=200 total_lines=430 cited=245 OK
  lightrag/llm/openai.py:986: HTTP=200 total_lines=1076 cited=986 OK
  ```
- **P0.5 Tests pass for every session** — `go vet`, `go build`, and `go test -count=1 ./...` all pass for every session. Each module reports a single `ok` package:
  ```
  ok  	learn-lightrag/s01	0.422s
  ok  	learn-lightrag/s02	1.671s
  ok  	learn-lightrag/s03	0.511s
  ok  	learn-lightrag/s04	0.504s
  ok  	learn-lightrag/s05	0.517s
  ok  	learn-lightrag/s06	1.554s
  ok  	learn-lightrag/s07	0.490s
  ok  	learn-lightrag/s08	0.519s
  ok  	learn-lightrag/s09	2.093s
  ok  	learn-lightrag/s10	1.374s
  ok  	learn-lightrag/s11	0.788s
  ```
- **P0.6 CI status** — last 10 GitHub Actions runs all `success` (10 success / 0 failure / 0 cancelled). Most recent commit `f320149` (`feat(M): multi-model addendum …`) shows both `web` and `docs` workflows green.

## P1 issues (should fix)

### P1.3 — README curriculum table marks all 15 entries with ✅ but `web/lib/curriculum.ts` has 3 entries flagged `available: false`

- File: `/Users/yeding/learn-lightrag/README.md` (lines 86–88) and `/Users/yeding/learn-lightrag/web/lib/curriculum.ts` (entries `s_full-integration`, `appendix-a-prompt-secrets`, `appendix-b-upstream-map`)
- Detail:
  ```
  README.md ✅ count: 15
  curriculum.ts available: true count: 12
  curriculum.ts available: false count: 3   (s_full, appendix A, appendix B)
  ```
  The README's `状态` column says all 15 chapters are shipped, but the doc-viewer site still treats those three as unavailable placeholders, so users following links from the live site will see "available: false" routing while the README claims they exist. The docs themselves (`docs/{zh,en}/s_full-integration.md`, `appendix-a-prompt-secrets.md`, `appendix-b-upstream-map.md`) DO exist and are fully written, so the README is right and `curriculum.ts` is stale.
- Suggested fix: in `web/lib/curriculum.ts`, flip `available: false` to `available: true` for the three trailing entries (`s_full-integration`, `appendix-a-prompt-secrets`, `appendix-b-upstream-map`).

Other P1 checks all passed cleanly:

- **P1.1 Web build** — `npm run typecheck` and `npm run build` both succeed cleanly. Build output ends with the standard Next.js stats block (no error, no warning surfacing in the tail-10).
- **P1.2 README links resolve** — every `[text](docs/...)`, `[text](agents/...)`, and `[text](upstream-readings/...)` link in `README.md` and `README.en.md` resolves to a file that exists on disk (sweep returned no `BROKEN:` output).
- **P1.4 Glossary terms in research-notes + chapter docs** — all spot-checked terms appear in `.learn/research-notes.md` and ≥1 chapter doc:
  ```
  gleaning             research-notes=YES chapter-docs=10
  doc-status           research-notes=YES chapter-docs=13
  high-level keywords  research-notes=YES (as "high-level vs low-level keywords") chapter-docs=3
  QueryParam           research-notes=YES chapter-docs=8
  FilterMissing        research-notes=YES (as filter_keys) chapter-docs=8
  tiktoken             research-notes=YES chapter-docs=7
  PENDING              research-notes=YES chapter-docs=5
  ```
  Note: research-notes uses upstream Python casing (`filter_keys`, `high-level`/`low-level keywords`); the chapter docs use the Go-side identifiers (`FilterMissing`, "high-level keywords"). This is intentional and correct.
- **P1.5 Diff narrative fidelity (s05/s07/s09)** — spot-checked the "What Changed" tables in `docs/en/s05-kv-store.md`, `s07-vector-store.md`, and `s09-extraction.md` against `diff -rq` output between the previous and current `agents/sNN-*/` directories:
  - s05's table cites `kv_json_store.go` → confirmed file exists; cites `FilterMissing` API → confirmed in `kv_store.go`.
  - s07's table cites `CosineIndex` struct + three indices (chunks/entities/relations) → `cosine_index.go` exists; `vector_store.go` is the dispatcher.
  - s09's "data origin / LLM calls / cache" table matches the new `extraction.go`, `gleaning.go`, `cache.go`, `parser.go`, `extraction_prompt.go` files.
  No fabrications detected in the spot check.

## P2 issues (nice to have)

### P2.1 — `testdata/expected.txt` exists for s01 only; the other 10 sessions have no golden output file

- File: `/Users/yeding/learn-lightrag/agents/s{02..11}-*/testdata/`
- Detail:
  ```
  OK: /Users/yeding/learn-lightrag/agents/s01-minimum-loop/
  MISSING: /Users/yeding/learn-lightrag/agents/s02-provider/ testdata/expected.txt
  MISSING: /Users/yeding/learn-lightrag/agents/s03-doc-status/ testdata/expected.txt
  MISSING: /Users/yeding/learn-lightrag/agents/s04-chunking/ testdata/expected.txt
  MISSING: /Users/yeding/learn-lightrag/agents/s05-kv-store/ testdata/expected.txt
  MISSING: /Users/yeding/learn-lightrag/agents/s06-embeddings/ testdata/expected.txt
  MISSING: /Users/yeding/learn-lightrag/agents/s07-vector-store/ testdata/expected.txt
  MISSING: /Users/yeding/learn-lightrag/agents/s08-graph-store/ testdata/expected.txt
  MISSING: /Users/yeding/learn-lightrag/agents/s09-extraction/ testdata/expected.txt
  MISSING: /Users/yeding/learn-lightrag/agents/s10-summarization/ testdata/expected.txt
  MISSING: /Users/yeding/learn-lightrag/agents/s11-query-modes/ testdata/expected.txt
  ```
  Since each session's `*_test.go` already covers correctness in-process, golden files are not strictly needed; this is purely an unfulfilled convention from s01.
- Suggested fix: either drop `expected.txt` from s01 to match the other 10, or add a minimal golden file to each session's `testdata/` directory.

Other P2 checks passed:

- **P2.2 Each session's README.md is non-empty (>40 lines)** — all 11 are between 44 and 97 lines (s05/s06 are the smallest at 44).
- **P2.3 Multi-model addendum coverage** — `docs/zh/multi-model.md` (164 lines) and `docs/en/multi-model.md` (164 lines) both cover all 8 declared profiles: OpenAI, DeepSeek, Qwen, Moonshot, Groq, OpenRouter, vLLM, Anthropic.

## Strengths

- **Bilingual parity is rock solid.** Heading-count check across 11 chapter pairs returned zero diffs — every `## ` in zh has a matching one in en. The spine check (six required sections per chapter doc) was clean across 22 files (11 zh + 11 en).
- **Build / test discipline is excellent.** All 11 Go modules pass `go vet`, `go build`, and `go test -count=1 ./...` from a clean checkout, and 10/10 most recent GitHub Actions runs are green. CI matrix actually catches regressions.
- **Cross-session isolation is enforced.** No `agents/sNN/` module imports another session's package. The "self-contained module per chapter" claim in the README is real, not aspirational.
- **Upstream citations are accurate.** Five randomly sampled `lightrag/<file>:<line>` references all resolved against the upstream `main` branch and the cited line numbers were within file bounds. The docs aren't fabricating line numbers.
- **Diff narratives are concrete and grounded.** "What Changed" tables in s05/s07/s09 reference specific Go files (`kv_json_store.go`, `cosine_index.go`, `extraction.go`, `gleaning.go`, etc.) that all exist on disk; the prose explains why each new file was added.

## Recommendations

If you're going to ship this:

- **Fix the `curriculum.ts` flags** — flipping `available: true` for `s_full`, `appendix-a`, and `appendix-b` is a 3-line edit. Until that lands, the doc-viewer's chapter list will mislead users about what's actually written.
- **Decide on `testdata/expected.txt` policy** — either delete s01's lone `expected.txt` (since `_test.go` already covers it), or add one to each session for parity. The current state where 1-of-11 has it is just visual noise.
- **Optional: bump session READMEs slightly** — s05 and s06 are at the 44-line floor (still passing P2.2's "non-empty" test) while s10 and s11 have 90+. Not a blocker.
