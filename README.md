# learn-lightrag

> 用 Go 从零渐进构建一个 [LightRAG](https://github.com/HKUDS/LightRAG)（HKUDS, EMNLP'25）。每节加一个机制——从最小 RAG 闭环到双层检索四种模式——并对照上游 Python 源码。
>
> Build [LightRAG](https://github.com/HKUDS/LightRAG) (HKUDS, EMNLP'25) from scratch in Go, session by session — from a minimum RAG loop all the way to dual-level retrieval across four query modes, with the upstream Python source on the side.

[English version below](#english) · [README.en.md](README.en.md)

---

## 这是什么 / What this is

LightRAG 是 GraphRAG 家族的开源实现：先从文档抽实体/关系建知识图谱，再用「向量检索 + 图遍历」双层并取做问答。这个仓库不是教你「**用** LightRAG」，而是教你「**它是怎么从零长出来的**」。

11 节 Go 实现 + 端到端集成 + 2 篇附录 + 多模型指南，每节单独成模块，互不依赖。看完 11 节你能徒手画出 LightRAG 的全栈架构图。

LightRAG is an open-source GraphRAG: extract entities/relations into a knowledge graph during ingestion, then answer queries by combining vector retrieval over chunks with graph traversal over the KG. This repo is **not** about *using* LightRAG — it's about **how it grows from scratch**.

11 Go chapters + end-to-end integration + 2 appendices + multi-model guide. Each chapter is its own self-contained module — no cross-imports. After 11 chapters you can sketch LightRAG's full architecture from memory.

---

## 快速上手 / Quickstart

```bash
git clone https://github.com/Ding-Ye/learn-lightrag.git
cd learn-lightrag/agents/s01-minimum-loop

# 用真 OpenAI（需要 OPENAI_API_KEY）
export OPENAI_API_KEY=sk-...
go run . -q "Who is Scrooge?"

# 完全离线（默认 mock provider + mock embedder，CI 用的就是这个）
go run . -provider mock -q "Who is Scrooge?"

# 跑测试
go test -v ./...
```

需要 Go ≥ 1.22。/ Go ≥ 1.22 required.

---

## 文档站 / Doc viewer

```bash
cd web
npm install
npm run dev    # http://localhost:3000
```

双语 Markdown 渲染 + 上游源码侧栏对照。

---

## 课程 / Curriculum

| # | 章节 (zh) | Chapter (en) | 教什么机制 | 状态 |
|---|---|---|---|---|
| M | [多模型接入指南](docs/zh/multi-model.md) | [Multi-model guide](docs/en/multi-model.md) | OpenAI / Anthropic / Bedrock / Ollama 一键切换 | ⏳ |
| s01 | [最小 RAG 闭环](docs/zh/s01-minimum-loop.md) | [Minimum RAG loop](docs/en/s01-minimum-loop.md) | 端到端 5 阶段管道（chunk → embed → store → retrieve → complete） | ✅ |
| s02 | [提供方接口](docs/zh/s02-provider.md) | [Provider interface](docs/en/s02-provider.md) | OpenAI 聊天补全 + 重试 + mock provider | ✅ |
| s03 | [文档状态机](docs/zh/s03-doc-status.md) | [Document status state machine](docs/en/s03-doc-status.md) | PENDING → PROCESSING → PROCESSED/FAILED | ✅ |
| s04 | [基于 token 的切分](docs/zh/s04-chunking.md) | [Token-based chunking](docs/en/s04-chunking.md) | tiktoken + 1200/100 滑窗 | ✅ |
| s05 | [键值存储与过滤](docs/zh/s05-kv-store.md) | [KV store with filter_keys](docs/en/s05-kv-store.md) | JSON 持久化 + per-key 锁 + FilterMissing | ✅ |
| s06 | [嵌入提供方与批处理](docs/zh/s06-embeddings.md) | [Embedding provider with batching](docs/en/s06-embeddings.md) | 128 batch + 重试 + Dim() 内省 | ✅ |
| s07 | [余弦相似向量库](docs/zh/s07-vector-store.md) | [Cosine-similarity vector store](docs/en/s07-vector-store.md) | 三个独立索引（chunks/entities/relations）+ 阈值 | ✅ |
| s08 | [邻接图存储与子图](docs/zh/s08-graph-store.md) | [Adjacency graph store + BFS](docs/en/s08-graph-store.md) | 无向图 + degree-priority BFS 子图 | ✅ |
| s09 | [实体关系抽取与 gleaning](docs/zh/s09-extraction.md) | [Entity/relation extraction + gleaning](docs/en/s09-extraction.md) | 分隔符解析 + 多轮续写循环 | ✅ |
| s10 | [描述归并](docs/zh/s10-summarization.md) | [Map-reduce summarization](docs/en/s10-summarization.md) | 阈值决策 + 递归 LLM 归并 | ✅ |
| s11 | 双层检索四种模式 | Dual-level retrieval, four modes | naive / local / global / hybrid | ⏳ |
| s_full | 端到端集成 | End-to-end integration | 16 步执行轨迹 + 全栈架构图 | ⏳ |
| App. A | 提示工程的秘密 | Prompt-engineering secret sauce | gleaning + 双层关键词 + 归并阈值 | ⏳ |
| App. B | 上游源码导读地图 | Upstream source-reading map | 每个上游文件 → 课程哪节 | ⏳ |

✅ = 已完成 / shipped；⏳ = 计划中 / planned。

---

## 仓库结构 / Layout

```
learn-lightrag/
├── agents/sNN-<slug>/      # 每节一个独立 Go 模块（自包含，无跨节 import）
│   └── s01-minimum-loop/
├── docs/{zh,en}/           # 双语文档，每节一对 .md，标题数自动比对
├── upstream-readings/      # 上游源码节选，带注解
├── web/                    # Next.js 15 文档站
├── .github/workflows/      # CI: go vet+build+test 矩阵 / 双语对齐 / web 构建
├── go.work                 # 多模块工作区
└── .learn/                 # 元数据（research-notes / plan / state）
```

---

## 为什么是 Go / Why Go

上游 LightRAG 是 Python，我们用 Go 重写不是为了"性能更好"——而是为了"语义更显式"：

- **类型替代 dict**：上游 `dict[str, Any]` 散落各处，Go 强制 struct 命名所有字段。
- **接口替代 callable**：上游 `llm_model_func: Callable` 在 Go 里是 `Provider` interface，编译期就能换实现。
- **goroutine + context 替代 asyncio**：取消语义显式可见，不再有 `PipelineCancelledException` 这种异常代控制流。
- **stdlib 优先**：除 `pkoukk/tiktoken-go`(s04) 外不引入任何第三方库，最大化代码可读性。

The upstream is Python; we rewrite in Go not for "better performance" but for **more explicit semantics**: types replace dicts, interfaces replace callables, `context.Context` replaces `asyncio.CancelledError`, stdlib-first.

---

## 致谢 / Credits

- **[HKUDS/LightRAG](https://github.com/HKUDS/LightRAG)**：上游实现 + EMNLP'25 paper。MIT licensed.
- **[shareAI-lab/learn-claude-code](https://github.com/shareAI-lab/learn-claude-code)**：教学法启发——「心智模型 → ASCII 图 → 30-60 行核心代码 → 与上一节的 diff → 动手试 → 上游源码导读」六段式骨架完全照搬。
- **`learn-repo-generator` skill**：自动化生成本仓库的工具链。

---

## License

MIT — see [LICENSE](LICENSE). 上游 LightRAG 同为 MIT，本仓库的 derivative 合规。
