---
title: "s11 · 双层检索四种模式"
chapter: 11
slug: s11-query-modes
est_read_min: 14
---

# s11 · 双层检索四种模式

> 这一节做什么：s01-s10 把整条**入库**链路全部搭好了 — chunk、embedding、KV、向量、图、抽取、归并 — 但查询时一行图代码都没用上。s11 把这个闭环合上。四种 `QueryMode`（naive / local / global / hybrid）各自在 `MaxTotalTokens` 预算内组装上下文，再交给 `Provider.Complete`。其中 hybrid 模式就是 LightRAG 的招牌「双层检索」：同时跑实体中心的 local 检索和关系中心的 global 检索，去重 chunk，再发起**一次**最终合成调用。对应上游 `kg_query`（`lightrag/operate.py:3164-3410`）+ `naive_query`（`:4930-5200`）。

---

## Problem / 问题

到这一节为止，我们存的数据已经很豪华：知识图谱里有实体、关系，三个独立的向量索引（chunks / entities / relations），KV store 里有 chunk 原文。但前面每一节都停在了「写入」。**没有一节真正在「查」什么东西。**

最朴素的回答是「embed 问题、找最近的 chunk、丢给 LLM」 — 这就是 s01 干的事，也是普通 RAG 干的事。**只要问题需要「不止一跳」，它就开始失败。**

朴素向量 RAG 撞墙的三种典型场景：

1. **同义词盲。** 文档说「首席档案员辞职了」，问题问「谁离任了？」。Embedding 余弦把它们算成 0.4 — 低于阈值，不命中。如果图把实体「首席档案员」连到关系「辞职」上，任意一条边都能把这块 chunk 拽出来。

2. **主题问题命中不到实体。** 「围绕这本古抄本起过哪些争议？」需要找的是 conflict / dispute / inquiry 标签的 chunk — **关系**而不是实体。chunk 级余弦相似只会把字面重叠最高的捞上来，那通常不是承载关系的那块。

3. **多跳问题需要 join。** 「Hartwell 发现的那本手稿是谁验证的？」要先 (a) 找到 Hartwell 发现了什么 → 再 (b) 对每个发现找出验证者。「Hartwell」上的向量命中只能给你第一步；第二步只有图能做。

修复方案从 s08 之后就一直摆在那里没用：**查询时遍历图**。但要怎么遍 — 从哪些种子节点出发、用多少预算、最终怎么把 prompt 压到 LLM 能吃下的尺寸？

## Solution / 解决方案

四种模式共享同一个 `Pipeline.Query(ctx, q, param)` 入口。每种模式对应一种问题形状：

- **Naive** — 只在 chunks 上做向量相似度；不碰图。适合答案与问题字面重合度高的情况。s01 的基线，正式化以便做 A/B。
- **Local** — 实体中心。从 query 抽出 LOW-level 关键词（实体名）→ `VDBEntities.Query` → 对每个 top entity 做一跳子图扩展 → 收集这些实体/边引用的 chunk。适合「指名道姓的某个东西，问它怎么样」类问题。
- **Global** — 关系中心。从 query 抽出 HIGH-level 关键词（主题/问法）→ `VDBRelations.Query` → 扩展到关系两端的实体 → 通过关系的 `SourceIDs` 拉 chunk。适合主题型问题。
- **Hybrid** — 同时跑 local + global，对 chunk 去重，按预算拼装（entity + relation + 余下给 chunk），最后**一次** LLM 合成。适合搞不清问题是什么形状的时候。

「双层关键词」是默默无闻的关键设计：一次 LLM 调用就从 query 抽出两个数组 — `high_level_keywords`（驱动 global）和 `low_level_keywords`（驱动 local）。Hybrid 模式只为这一次抽取付一次费，但两个数组都用得上。这是让 local / global 能作为「正交但可组合」的检索维度共存的架构动作。

共享的 `truncateByTokens` 工具（上游分段预算逻辑的端口）按各自的 `MaxEntityTokens` / `MaxRelationTokens` 独立修剪每段；chunk 段拿 `MaxTotalTokens` 的余量。最终 prompt 有界但不会饿死。

## How It Works / 工作原理

### 模式分发

```
Pipeline.Query(ctx, q, param)
  │
  ├── ModeNaive  ──▶ queryNaive   ──▶ VDBChunks ──▶ KV ──▶ Complete
  │
  ├── ModeLocal  ──▶ queryLocal   ──▶ extractKeywords (只用 LL)
  │                                ──▶ VDBEntities ──▶ Graph.GetSubgraph
  │                                ──▶ 收集 chunk IDs ──▶ KV ──▶ Complete
  │
  ├── ModeGlobal ──▶ queryGlobal  ──▶ extractKeywords (只用 HL)
  │                                ──▶ VDBRelations ──▶ 端点实体
  │                                ──▶ 收集 chunk IDs ──▶ KV ──▶ Complete
  │
  └── ModeHybrid ──▶ queryHybrid  ──▶ extractKeywords (一次提两组 HL+LL)
                                   ──▶ retrieveLocal  (不调 LLM)
                                   ──▶ retrieveGlobal (不调 LLM)
                                   ──▶ 合并 + 去重 chunk
                                   ──▶ 一次 Complete
```

### Naive 模式（基线）

```
                     ┌─────────────────┐
   query string ────▶│  Embedder.Embed │
                     └────────┬────────┘
                              ▼
                     ┌─────────────────┐
                     │ VDBChunks.Query │  topK=ChunkTopK, threshold=-1
                     └────────┬────────┘
                              ▼
                     ┌─────────────────┐
                     │   KV.GetByIDs   │
                     └────────┬────────┘
                              ▼
              ┌───────────────────────────────┐
              │  buildSystemPrompt(           │
              │    nil entities,              │
              │    nil relations,             │
              │    chunks,                    │
              │    q, param )                 │
              └───────────┬───────────────────┘
                          ▼
                  Provider.Complete
```

Naive 是普通 RAG 项目的标配。它是我们的下限；其他模式都得跑赢它。

### Local 模式（实体中心）

```
   query
     │
     ├──▶ extractKeywords ──▶ (HL 弃用，LL 保留)
     │
     ▼
   for each LL keyword:
     Embedder.Embed(keyword)
     VDBEntities.Query(vec, TopK)
     │
     ▼
   dedupHits(实体命中)
     │
     ▼
   for each top entity:
     Graph.GetSubgraph(entity_id, depth=1, maxNodes=TopK)
     │
     ▼
   收集 entSet, relSet
     │
     ▼
   chunkIDs = unique(SourceIDs(entSet) ++ SourceIDs(relSet))[:ChunkTopK]
     │
     ▼
   KV.GetByIDs(chunkIDs) ──▶ buildSystemPrompt ──▶ Complete
```

教学要点：低层关键词天然是实体形状，因为 LLM 被要求抽出「具体实体、专有名词、技术术语」。它们 embed 起来便宜（每个关键词一次 Embed），而且它们对 VDBEntities 的索引方式恰好就是实体名 embedding 应该被索引的方式。

### Global 模式（关系中心）

```
   query
     │
     ├──▶ extractKeywords ──▶ (HL 保留，LL 弃用)
     │
     ▼
   for each HL keyword:
     Embedder.Embed(keyword)
     VDBRelations.Query(vec, TopK)
     │
     ▼
   for each relation hit:
     从 metadata 重建 Relationship
     Graph.GetNode(SrcID), Graph.GetNode(TgtID)
     │
     ▼
   收集 relSet, entSet
     │
     ▼
   chunkIDs = unique(SourceIDs(relSet) ++ SourceIDs(entSet))[:ChunkTopK]
     │
     ▼
   KV.GetByIDs(chunkIDs) ──▶ buildSystemPrompt ──▶ Complete
```

关系索引必须在入库时种好 `keywords + description` 的 embedding。这就是为什么 s10 的归并步骤重要 — 喂给 embedder 的描述**已经是**那条归并后的描述，而不是 12 段没合并的零碎。

### Hybrid 模式（招牌）

```
   query
     │
     └──▶ extractKeywords  ──▶  (HL, LL)   ── 一次调用，两个数组
              │
              ├──▶ retrieveLocal  (不调 LLM)  ──▶ entSet_L, relSet_L, chunkIDs_L
              │
              └──▶ retrieveGlobal (不调 LLM)  ──▶ entSet_G, relSet_G, chunkIDs_G
              │
              ▼
          merge entSet_L ∪ entSet_G
          merge relSet_L ∪ relSet_G
          chunkIDs = dedup(chunkIDs_L ++ chunkIDs_G)[:ChunkTopK]
              │
              ▼
          buildSystemPrompt(entities, relations, chunks, q, param)
              │
              ▼
          Provider.Complete   ── 一次调用，一个答案
```

三个不显眼的关键点：

1. **一次合成调用，不是两次。** Hybrid **不是**「先跑 local 再跑 global，最后把两个答案拼起来」。它跑两次**检索**，把合并后的上下文喂给一次 Complete。两个答案拼起来口吻不一致、框架可能矛盾；合并到一个 prompt 里让 LLM 内部协调反而是更稳的做法。
2. **Token 预算先分段、再总封顶。** `MaxEntityTokens` 和 `MaxRelationTokens` 是独立预算。Chunk 拿 `MaxTotalTokens` 减去 entity + relation 之后的余量。如果 `MaxEntityTokens + MaxRelationTokens > MaxTotalTokens` 已经超了，chunk 就分不到了 — 那是用户预算开错的问题。
3. **Chunk 去重按 ID 而非内容。** 两段不同 ID 但内容有重叠的 chunk 都会进（它们是不同文档的不同 chunk）；两次检索同时拉到的 chunk-3 只会进一次。够用了；内容感知去重留作 Phase G 练习。

### `buildSystemPrompt`（先修剪再拼装）

```go
// 来自 context_builder.go
entSection := truncateByTokens(renderEntities(entities), param.MaxEntityTokens)
relSection := truncateByTokens(renderRelations(rels),    param.MaxRelationTokens)

used := tokenCount(entSection) + tokenCount(relSection)
chunkBudget := param.MaxTotalTokens - used
if chunkBudget < 0 { chunkBudget = 0 }
chunkRows := truncateChunksByTokens(chunks, chunkBudget)

// 然后拼接：## Entities / ## Relations / ## Chunks / ## User Query
```

顺序很重要：实体在前、关系在中、chunk 在后。实体在不同问题间复用度最高；chunk 在 token 上最贵。砍 chunk 段尾巴是稳住总预算的代价最低的方式。

## What Changed / 与 s10 的变化

s10 做完了入库的最后一公里：每个实体 47 段描述 → 一段归并摘要，由阈值驱动的决策树判断要不要调 LLM。s10 之后，图里每个实体都有一条干净的单句描述，关系索引里都种好了 `keywords + description` 的 embedding。

s11 终于在查询时**用上**了这一切。三处关键变化：

1. **`Query` 是新的对外 API。** 之前每一节暴露的都是 `Insert` / `Upsert` / `Embed` — 写侧 API。s11 引入 `Query(ctx, q, QueryParam) (QueryResult, error)`，这才是调用方真正会用的接口。`Pipeline` 现在从每个 storage 读；不再写。
2. **双层关键词是路由原语。** 没有高/低层切分的话，`local` / `global` 都打不过 naive 多少。有了之后，我们可以按问题**形状**路由：指名道姓的去 local，主题型的去 global，看不出形状的去 hybrid。
3. **Token 预算是分段制。** 之前每节都没 token 预算的概念（每个函数处理一块 chunk 或一个实体）。s11 的「三段 prompt + 三个预算」是仓库里第一次需要对一个完整 prompt 的尺寸做整体推理。

## Try It / 动手试一试

```bash
cd agents/s11-query-modes

# 同一问题跑四种模式做对比。
go run . -mode all -q "Who discovered the Hartwell Codex?"

# 单独跑一种。
go run . -mode hybrid -q "What conflicts arose around the codex?"
go run . -mode local  -q "Where does Eleanor Hartwell work?"
go run . -mode global -q "What kinds of academic verification has Hartwell pursued?"

# 测试（离线、确定性，约 0.3 秒）。
go test -count=1 -v ./...
```

`-mode all` 期望输出形如：

```
=== s11: query="Who discovered the Hartwell Codex?" ===

[naive]
  references: [chunk-2 chunk-3 ...]
  answer    : Eleanor Hartwell ...
[local]
  references: [chunk-1 chunk-2 chunk-6 chunk-4 chunk-5]
  answer    : Eleanor Hartwell ...
[global]
  references: [chunk-1 chunk-4 chunk-5 ...]
  answer    : Eleanor Hartwell ...
[hybrid]
  references: [chunk-1 chunk-2 chunk-6 chunk-4 chunk-5]
  answer    : Eleanor Hartwell ...
```

对比四种模式的 `references` 列表 — naive 与 local 排序不同，因为 naive 衡量的是 chunk 内容上的余弦相似，而 local 衡量的是「头部实体（及其子图）引用了哪些 chunk」。Hybrid 是两者的并集（去重后）。

## Upstream Source Reading / 上游源码阅读

完整带注释的摘录在 [`upstream-readings/s11-query.py`](../../upstream-readings/s11-query.py)。承重的那一段 — kg_query 的模式分发 — 在此处复刻（≤ 50 LOC）：

```python
# operate.py:3164-3232  (kg_query, 模式分发 + 关键词回退)
async def kg_query(
    query, knowledge_graph_inst, entities_vdb, relationships_vdb,
    text_chunks_db, query_param, global_config, hashing_kv=None,
    system_prompt=None, chunks_vdb=None,
) -> QueryResult | None:
    if not query:
        return QueryResult(content=PROMPTS["fail_response"])

    # 一次 LLM 调用同时抽出高层和低层关键词。
    hl_keywords, ll_keywords = await get_keywords_from_query(
        query, query_param, global_config, hashing_kv,
    )
    if ll_keywords == [] and query_param.mode in ["local", "hybrid", "mix"]:
        logger.warning("low_level_keywords is empty")
    if hl_keywords == [] and query_param.mode in ["global", "hybrid", "mix"]:
        logger.warning("high_level_keywords is empty")
    if hl_keywords == [] and ll_keywords == []:
        if len(query) < 50:
            ll_keywords = [query]              # 用 query 自身做种子
        else:
            return QueryResult(content=PROMPTS["fail_response"])

    ll_keywords_str = ", ".join(ll_keywords) if ll_keywords else ""
    hl_keywords_str = ", ".join(hl_keywords) if hl_keywords else ""

    context_result = await _build_query_context(
        query, ll_keywords_str, hl_keywords_str,
        knowledge_graph_inst, entities_vdb, relationships_vdb,
        text_chunks_db, query_param, chunks_vdb,
    )
    if context_result is None:
        return None

    # [省略：cache、only_need_context、only_need_prompt、流式分支]

    sys_prompt = PROMPTS["rag_response"].format(
        response_type=query_param.response_type or "Multiple Paragraphs",
        user_prompt=query_param.user_prompt or "n/a",
        context_data=context_result.context,
    )
    response = await use_model_func(query, system_prompt=sys_prompt, ...)
    return QueryResult(content=response, raw_data=context_result.raw_data)
```

读这段时要注意三件事：

- **模式分发是隐式的。** `kg_query` 自己里没有 `match query_param.mode:` 那种 switch；分发被藏在 `_build_query_context` 里（operate.py:3700+），它内部按模式分支，产生不同的 `(entities, relations, chunks)` 三元组。Go 端为了易读，把那个分支外提到了 `query_local.go` / `query_global.go` / `query_hybrid.go`。
- **「无关键词」回退。** 当 LLM 两个数组都返回空时，上游回退到把 query **本身**当低层关键词用（query 短的时候）或直接失败（query 长的时候）。Go 端在 `query_local.go` / `query_global.go` 里复制了这个回退（`if len(keys) == 0 { keys = []string{q} }`）。
- **Naive 是平级，不是子级。** `naive_query` 是 operate.py:4953 一个独立的顶层函数；上游没把 naive 设为 kg_query 的子类。Go 端的 `Pipeline.Query` 把它们在 API 层统一起来，更接近一个库的实际使用方式。

`_build_query_context` 在 local / global / hybrid 三种模式下的差异更深的拆解，留到 `s_full` 集成章里走「insert→query 全流程」时再带行号锚点贴上游引用。
