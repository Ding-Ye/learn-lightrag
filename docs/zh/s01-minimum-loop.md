---
title: "s01 · 最小 RAG 闭环"
chapter: 1
slug: s01-minimum-loop
est_read_min: 12
---

# s01 · 最小 RAG 闭环

> 教什么：把"插入文档 → 提问 → 拿到带引用的答案"这条最小 RAG 路径，用 ~1000 行 Go 走通一遍。本节是后面 10 节的脚手架——所有 interface（Provider / EmbeddingProvider / VectorStore / KVStore）在这里**先声明**，后面每节挑一个换成「真实」实现。

---

## Problem / 问题

读上游 [HKUDS/LightRAG](https://github.com/HKUDS/LightRAG) 的代码，第一个劝退点是「无从下口」：`lightrag/` 目录里 30+ 个文件，单 `lightrag.py` 就 3000 行，`operate.py` 5000 行，到处是 `dict[str, Any]`、双 sync/async API、60 字段的 `@dataclass`。学习者想问的其实只有一句话：「**这玩意儿到底是怎么从一段文本变成一个能回答问题的系统的？**」

s01 的痛点：在还没引入 token 切分、状态机、知识图谱、抽取这些复杂机制之前，先把**完整闭环**跑通——让你 `go run . -q "..."` 拿到一个真实答案，并看到答案引用了哪几个 chunk。这个闭环就是后面 10 节的对照基线。

## Solution / 解决方案

把 RAG 拆成 5 个最小阶段：**chunk** → **embed** → **store** → **retrieve** → **complete**。每个阶段写一个最 stub 的实现 + 声明对应 interface，让后面每节都能「无痛替换」其中一块。

3 个关键决策：
1. **接口先于实现**：`Provider` / `EmbeddingProvider` / `VectorStore` / `KVStore` 全部先声明 interface，s01 的具体实现就是用 `sync.Map` 和切片扫描。s05 把 KV 换 JSON 持久化，s07 把 VectorStore 换索引——调用方一行不改。
2. **离线 mock 优先**：CI 不能依赖 OpenAI，所以 s01 自带 `MockProvider`（确定性回显）+ `MockEmbedder`（sha256 hash → 单位向量）。本地有 `OPENAI_API_KEY` 才走真接口。
3. **chunkID 是契约**：用 `<docID>::chunk-<index>` 这个串既做 KV key 又做 VDB ID 又做最终 References 输出。后面 10 节都保持这个契约。

## How It Works / 工作原理

```
┌────────────────────────────────────────────────────────────────────────┐
│                     s01 minimum RAG loop (~1000 LOC)                   │
│                                                                        │
│   Insert(docID, text)                                                  │
│         │                                                              │
│         ▼                                                              │
│   ChunkByNewlines  ──→  Embedder.Embed  ──→  VDB.Upsert  +  KV.Upsert  │
│                          (mock|openai)      (in-mem cosine) (sync.Map) │
│                                                                        │
│   Query(q, topK=3)                                                     │
│         │                                                              │
│         ▼                                                              │
│   Embedder.Embed(q) → VDB.Query → KV.GetByIDs → build system prompt    │
│         │                                                              │
│         ▼                                                              │
│   Provider.Complete  ──→  QueryResult{Content, References, Mode}       │
│       (mock|openai)                                                    │
└────────────────────────────────────────────────────────────────────────┘
```

核心 30 行（节选自 [`agents/s01-minimum-loop/pipeline.go`](https://github.com/Ding-Ye/learn-lightrag/blob/main/agents/s01-minimum-loop/pipeline.go)）：

```go
func (p *Pipeline) Query(ctx context.Context, q string, topK int) (QueryResult, error) {
    qVecs, err := p.Embedder.Embed(ctx, []string{q})
    if err != nil { return QueryResult{}, err }

    hits, err := p.VDB.Query(ctx, qVecs[0], topK, -1.0) // -1.0 = no threshold in s01
    if err != nil { return QueryResult{}, err }

    ids := make([]string, len(hits))
    for i, h := range hits { ids[i] = h.ID }
    contents, err := p.KV.GetByIDs(ctx, ids)
    if err != nil { return QueryResult{}, err }

    var ctxBuf strings.Builder
    ctxBuf.WriteString("Context:\n")
    for _, id := range ids {
        body, _ := contents[id]["content"].(string)
        fmt.Fprintf(&ctxBuf, "[chunk %s]\n%s\n\n", id, body)
    }
    system := "You are a helpful assistant. Answer using ONLY the context below. " +
              "Cite chunk IDs in [brackets].\n\n" + ctxBuf.String()

    resp, err := p.Provider.Complete(ctx, CompleteRequest{
        System:   system,
        Messages: []Message{{Role: "user", Content: q}},
    })
    if err != nil { return QueryResult{}, err }
    return QueryResult{Content: resp.Text, References: ids, Mode: ModeNaive}, nil
}
```

**4 个非显然之处**：

1. **threshold = -1 是占位** —— s01 的 VectorStore 不做阈值筛选（拿满 topK 个就行）。s07 才把阈值变成真用法（向量和 query 太不像就丢掉）。这里传 -1.0 是「不要过滤」的语义。
2. **`Pipeline.KV` 用 chunkID 做主键** —— 同一个 ID 既存在 VDB 也存在 KV：VDB 用来按相似度检索，KV 用来拿 chunk 的原文。后面 11 节都保持这个一一对应的设计。
3. **system prompt 里硬编码引用规则** —— "Cite chunk IDs in [brackets]" 让 LLM 输出形如 `…answered in [book::chunk-2]…`。这是 s01 的 mock provider 验证「retrieved 哪几个 chunk」的钩子；真 OpenAI 也照办。
4. **`References` 直接返回 retrieved 顺序** —— 不去重不重排，VDB 给什么就用什么。s11 才会按 token 预算做截断和按相关度做 rerank。

## What Changed / 与上一节的变化

s01 是第一节，没有 s00 可以做 diff。本节相对于「空仓库」**新增了**这些 interface 和契约——后面每节都基于它们演化：

- **`Provider` 接口**：单方法 `Complete(ctx, req) (resp, error)`。s02 把 `OpenAIProvider` 提到独立文件并加重试；Phase G 加 `AnthropicProvider` 等——签名一辈子不变。
- **`EmbeddingProvider` 接口**：`Embed([]string) → [][]float32` + `Dim() int`。s06 实现 batch + retry。
- **`VectorStore` 接口**：`Upsert / Query / Delete / Persist`。s07 实现真索引 + JSON 持久化 + 阈值。
- **`KVStore` 接口**：`Get / GetByIDs / Upsert / FilterMissing / Delete / Persist`。`FilterMissing` 是后面 s09 决定「哪些 chunk 还没抽过实体」的关键原语。
- **`chunkID = <docID>::chunk-<idx>`** 命名约定：所有节都用这个 ID 做 cross-store join。
- **`QueryResult{Content, References, Mode}`** 输出形态：所有节的最终返回值都是这个 struct。

语义上的差别：s01 是**单进程同步管道**；s05 之后会出现真磁盘持久化，s09 引入并发受控的实体抽取（带信号量），但**调用方接口形态从 s01 起就锁定**。

## Try It / 动手试一试

```bash
cd agents/s01-minimum-loop

# 1. 全离线 mock，演示「没 API key 也能跑」
go run . -provider mock -q "What did Eleanor patent in 1872?"

# 2. -v 打开 retrieval 调试
go run . -provider mock -v -q "Where did Eleanor's papers end up?"

# 3. 真 OpenAI（需要 OPENAI_API_KEY）
export OPENAI_API_KEY=sk-...
go run . -q "Why was the Tarvin Highlands trip cancelled?"

# 4. 跑测试（CI 矩阵跑这个）
go test -v ./...
```

期望输出形态：

```
[s01] ingested doc "testdata/book.txt" (provider=mock)
=== Answer ===
[mock] retrieved 3 chunks (book::chunk-1, book::chunk-0, book::chunk-2)
preview of top chunk: "Eleanor's most famous patent, granted in 1872, covered ..."

=== References ===
- book::chunk-1
- book::chunk-0
- book::chunk-2

[mode=naive]
```

mock provider 的"答案"是把 system prompt 的前 200 字截一段回显，**确定性**；CI 才能稳定 assert。真 OpenAI 答案会自然语言化并按 system prompt 的指示在答案里嵌 `[book::chunk-N]` 的引用。

## Upstream Source Reading / 上游源码阅读

上游 LightRAG 的等价路径是 `lightrag/lightrag.py:1237` 的 `ainsert()` + `:2622` 的 `aquery()`（+ `:2884` 的 `aquery_llm()` 真正干活）。s01 把这条路径里的 chunking + embed + vdb 部分压成了 ~80 行的 `Pipeline.Insert/Query`，把 doc-status / 实体抽取 / 4 种 mode 全部砍掉留给后面 10 节。

```upstream:lightrag/lightrag.py#L2884-L2940
async def aquery_llm(
    self,
    query: str,
    param: QueryParam = QueryParam(),
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """Asynchronous complete query API: structured retrieval + LLM generation."""
    global_config = asdict(self)

    if param.mode in ["local", "global", "hybrid", "mix"]:
        # 图模式：抽关键词 → 遍历 KG → 收集 chunks。s11 实现这条分支。
        query_result = await kg_query(
            query.strip(),
            self.chunk_entity_relation_graph,    # ← s08 GraphStore
            self.entities_vdb,                   # ← s07 entities 向量索引
            self.relationships_vdb,              # ← s07 relations 向量索引
            self.text_chunks,                    # ← s05 chunk 内容 KV
            param, global_config,
            hashing_kv=self.llm_response_cache,
            system_prompt=system_prompt,
            chunks_vdb=self.chunks_vdb,
        )
    elif param.mode == "naive":
        # 简单分支：仅向量相似度，不碰图。s01 Pipeline.Query 就是这个。
        query_result = await naive_query(
            query.strip(),
            self.chunks_vdb,
            param, global_config,
            hashing_kv=self.llm_response_cache,
            system_prompt=system_prompt,
        )
    elif param.mode == "bypass":
        # 完全不检索，直接问 LLM。诊断用。s01..s11 都没实现，附录 A 提及。
        ...
```

**对照阅读要点**：

- **mode 调度 vs 单一 naive 分支**：上游一个函数四个 mode 分支；s01 只实现 `naive_query` 等价物，把 `Pipeline.Query.Mode` 硬编码为 `ModeNaive`。s11 才把 `kg_query` 那条分支补齐。
- **`global_config = asdict(self)` 反模式**：上游把整个 60 字段的 dataclass 扔成 `dict` 往下传；Go 里这就是 anti-pattern。s01 直接给 `Pipeline` struct 显式字段，s11 也维持。
- **`hashing_kv=self.llm_response_cache` 复用**：上游每次 query 都查一次 LLM cache（避免重复花钱）；s01 没有这个 cache，因为 quickstart 阶段查询很少重复。s09 实现 cache（用同一个 KVStore 做 hash-keyed 存储）。
- **Streaming** —— `is_streaming` 分支返回 `AsyncIterator[str]`；s01 的 `Provider` 接口已留 `Stream bool` 字段但永远传 `false`。Phase G 才把它打通。
- **`naive_query` 的 16 行内核** —— 它就是 s01 `Pipeline.Query` 的 Python 兄弟：embed query → `chunks_vdb.query(top_k)` → `text_chunks.get_by_ids(...)` → 拼 context → 调 LLM。你完全可以把这 16 行和 s01 那 30 行 Go 并排放（[upstream-readings/s01-lightrag.py](../../upstream-readings/s01-lightrag.py) 注解版有详细对照）。

**想读更多**：从 `lightrag/lightrag.py` 的 `LightRAG.ainsert()` (L1237) 入手，跟着 `apipeline_process_enqueue_documents` 进 `lightrag/operate.py` 看真正的 chunking + extraction 循环（约 L100-L500），最后读 `lightrag/operate.py:3164` 的 `kg_query` 看 4-mode 分支。这条线就是 s01 → s04 → s09 → s11 的真实代码地图。

---

**下一节预告**：s02 把 `Provider` 接口正式独立成一节，加重试 / functional options / 三种实现（OpenAI / Mock / Echo）。Phase G 在 s02 的接口上加 Anthropic 和 Bedrock，无需改任何 caller。
