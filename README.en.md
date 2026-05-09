# learn-lightrag

> Build [LightRAG](https://github.com/HKUDS/LightRAG) (HKUDS, EMNLP'25) from scratch in Go, session by session — from a minimum RAG loop all the way to dual-level retrieval across four query modes, with the upstream Python source on the side.

[中文版](README.md)

---

## What this is

LightRAG is an open-source implementation in the GraphRAG family: ingest documents by extracting entities/relations into a knowledge graph, then answer queries by combining vector retrieval over chunks with graph traversal over the KG. This repo is **not** about *using* LightRAG — it's about **how it grows from scratch**.

11 Go chapters + end-to-end integration + 2 appendices + multi-model guide. Each chapter is its own self-contained module — no cross-imports. After 11 chapters you can sketch LightRAG's full architecture from memory.

---

## Quickstart

```bash
git clone https://github.com/Ding-Ye/learn-lightrag.git
cd learn-lightrag/agents/s01-minimum-loop

# Real OpenAI (requires OPENAI_API_KEY)
export OPENAI_API_KEY=sk-...
go run . -q "Who is Scrooge?"

# Fully offline (default mock provider + mock embedder; what CI uses)
go run . -provider mock -q "Who is Scrooge?"

# Run tests
go test -v ./...
```

Go ≥ 1.22 required.

---

## Doc viewer

```bash
cd web
npm install
npm run dev    # http://localhost:3000
```

Bilingual Markdown rendering with an upstream source pane on the side.

---

## Curriculum

| # | Chapter | Mechanism taught | Status |
|---|---|---|---|
| M | [Multi-model guide](docs/en/multi-model.md) | OpenAI / Anthropic / Bedrock / Ollama swap | ⏳ |
| s01 | [Minimum RAG loop](docs/en/s01-minimum-loop.md) | End-to-end 5-stage pipeline (chunk → embed → store → retrieve → complete) | ✅ |
| s02 | Provider interface | OpenAI chat completion + retry + mock provider | ⏳ |
| s03 | Document status state machine | PENDING → PROCESSING → PROCESSED/FAILED | ⏳ |
| s04 | Token-based chunking | tiktoken + 1200/100 sliding window | ⏳ |
| s05 | KV store with filter_keys | JSON persistence + per-key locking + FilterMissing | ⏳ |
| s06 | Embedding provider with batching | 128 batch + retry + Dim() introspection | ⏳ |
| s07 | Cosine-similarity vector store | Three indices (chunks/entities/relations) + thresholding | ⏳ |
| s08 | Adjacency graph store + BFS | Undirected graph + degree-priority BFS subgraph | ⏳ |
| s09 | Entity/relation extraction + gleaning | Delimiter parsing + multi-round continuation | ⏳ |
| s10 | Map-reduce summarization | Threshold decision + recursive LLM merge | ⏳ |
| s11 | Dual-level retrieval, four modes | naive / local / global / hybrid | ⏳ |
| s_full | End-to-end integration | 16-step trace + full-stack diagram | ⏳ |
| App. A | Prompt-engineering secret sauce | Gleaning + dual-level keywords + merge threshold | ⏳ |
| App. B | Upstream source-reading map | Every upstream file → which chapter touches it | ⏳ |

✅ shipped; ⏳ planned.

---

## Layout

```
learn-lightrag/
├── agents/sNN-<slug>/      # One Go module per chapter, fully self-contained
│   └── s01-minimum-loop/
├── docs/{zh,en}/           # Bilingual docs, paired .md per chapter, heading count auto-checked
├── upstream-readings/      # Annotated upstream excerpts
├── web/                    # Next.js 15 doc site
├── .github/workflows/      # CI: go matrix / docs parity / web build
├── go.work                 # Multi-module workspace
└── .learn/                 # Metadata (research-notes / plan / state)
```

---

## Why Go

Upstream LightRAG is Python; we rewrite in Go not for "better performance" but for **more explicit semantics**:

- **Types replace dicts**: upstream's `dict[str, Any]` becomes a named Go struct with every field declared.
- **Interfaces replace callables**: upstream's `llm_model_func: Callable` becomes a `Provider` interface — swap implementations at compile time.
- **goroutine + context replace asyncio**: cancellation is explicit and compile-checked, no `PipelineCancelledException`-as-control-flow.
- **stdlib-first**: nothing beyond `pkoukk/tiktoken-go` (s04) — maximizes code readability.

---

## Credits

- **[HKUDS/LightRAG](https://github.com/HKUDS/LightRAG)** — upstream implementation + EMNLP'25 paper. MIT licensed.
- **[shareAI-lab/learn-claude-code](https://github.com/shareAI-lab/learn-claude-code)** — pedagogy: the six-section spine (Problem → Solution → How It Works → What Changed → Try It → Upstream Source Reading) is borrowed directly.
- **`learn-repo-generator` skill** — toolchain that automates building this repo.

---

## License

MIT — see [LICENSE](LICENSE). Upstream LightRAG is also MIT, so this derivative is fully compliant.
