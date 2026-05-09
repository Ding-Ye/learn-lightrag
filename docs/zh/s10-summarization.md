---
title: "s10 · 描述归并 (map-reduce)"
chapter: 10
slug: s10-summarization
est_read_min: 9
---

# s10 · 描述归并 (map-reduce)

> 这一节做什么：当同一个实体在 47 个 chunk 里出现，我们手上就有 47 段独立描述。直接拼接撑爆 embedding 的 token 上限；扔给 LLM 一次又造成 retrieval-time 的语义混乱。s10 镜像上游 `_handle_entity_relation_summary` + `_summarize_descriptions`（lightrag/operate.py:167-385）的「阈值决策 + map-reduce LLM 归并」组合：小数据走拼接，大数据走递归归并。

---

## Problem / 问题

s09 给出的 Extractor 每次只看一个 chunk。它产出的每条 `Entity{Name:"Scrooge", Description:"..."}` 都只携带这块 chunk 里的那段描述。但同一个 doc 里 Scrooge 可能出现在 47 个 chunk，于是 graph store 一不小心就有 47 份 Scrooge 记录，每份描述各自不同。

「合并掉就好了」——具体怎么合？四条死路：

1. **直接拼接：token 爆掉。** 47 段描述每段 30 token，直接拼是 1410 token；OpenAI 的 `text-embedding-3-small` 上限 8192 token 还能扛，但若每段 200 token 就是 9400，挤不进去。即使挤进去了，向量被「噪声」稀释，retrieval 时召回不准。
2. **保留最长那段：丢信息。** "Scrooge 是商人" 和 "Scrooge 在 Christmas Eve 见到 Marley 的鬼魂" 都重要，最长可能是别的句子，丢了关键事实。
3. **Always 用 LLM：成本失控 + 退化。** 每次 ingest 都对每个实体调一次 GPT-4 是浪费——对于「Scrooge 出现 2 次」的小实体，拼接就够；对「Scrooge 出现 47 次」才需要 LLM 综合。
4. **JSON schema 做合并：LLM 不擅长。** 强行让 LLM 输出 `{"merged":"..."}` 反而比让它写自由文段表现差，原因和 s09 一致：JSON 边界字符容易让 LLM 漏 `}`，整段废掉。

LightRAG 的优雅解：**双阈值决策树**——`(count < threshold AND total < budget) → 拼接，否则 → 递归 LLM 归并`。两个阈值缺一不可：单看 count 时 4 段每段 2k token 也炸；单看 token 时 30 段短描述也乱。

s10 把这套决策树搬到 Go，加一个 `tokenCount()` 桩（whitespace word count，s04 才有真 tiktoken）以保持模块隔离。

## Solution / 解决方案

四件事：

1. **`SummarizeDescriptions(ctx, p Provider, descriptions, budgetTokens, contextSize, countThreshold)` 是单一入口。**返回 `(summary, llmUsed, err)`。Provider 与 s02、s09 共形（同一接口在每节都重新声明，sessions 之间不导入）。
2. **决策树（直译上游 operate.py:222-226）：** `len==0` → `("", false, nil)`；`len==1` → 原样返回；`len < threshold AND total < budget` → `\n\n`-拼接；其它 → 走 map-reduce。
3. **Map 阶段：** 把 descriptions 切成 `contextSize` token 的窗口，**每个窗口至少 2 段**（上游 operate.py:255-263 的「保证收敛」规则——单段窗口直接 pass-through 会死循环）。
4. **Reduce 阶段：** 每个多段窗口调一次 `Provider.Complete(summaryPrompt + JSONL)`；单段窗口透传不走 LLM；最后把 partial summaries 合起来——若仍超 budget 则递归（最多 `DefaultMaxRecursionDepth=3` 层；上游 `while True`，我们设上限避免测试里成本失控）。

`MergeEntities` 与 `MergeRelationships` 是 caller-side 的胶水：按 Name（实体）或 canonical(min,max)(关系)分组、调用 `SummarizeDescriptionsForName` 把片段合并，最后给出一条合并记录，SourceIDs 取并集 + 排序。

## How It Works / 工作原理

```
                     SummarizeDescriptions(ctx, p, descs, budget, ctxSize, threshold)
   ┌─────────────────────────────────────────────────────────────────────────┐
   │                                                                         │
   │   Step 1  edge cases:                                                   │
   │             len(descs) == 0  →  ("", false, nil)                        │
   │             len(descs) == 1  →  (descs[0], false, nil)                  │
   │                                                                         │
   │   Step 2  cheap path:                                                   │
   │             if len(descs) < countThreshold &&                           │
   │                totalTokens(descs) < budgetTokens:                       │
   │                  return strings.Join(descs, "\n\n"), false, nil         │
   │                                                                         │
   │   Step 3  map: chunk descs by contextSize, ensuring each chunk has ≥2   │
   │                                                                         │
   │   Step 4  reduce: for each chunk:                                       │
   │             if len(chunk) == 1: pass-through (no LLM)                   │
   │             else: Provider.Complete(prompt + JSONL of chunk)            │
   │                                                                         │
   │   Step 5  converge:                                                     │
   │             if len(partials) == 1                  →  return            │
   │             if joined fits budget AND under thresh →  return joined     │
   │             if depth+1 >= MaxRecursionDepth         →  one final LLM    │
   │             else: recurse on partials (depth+1)                         │
   │                                                                         │
   └─────────────────────────────────────────────────────────────────────────┘

   MergeEntities/MergeRelationships:
       group by Name (or canonical edge) ──► SummarizeDescriptionsForName ──► merged
       SourceIDs = union(sources), sorted
       Weights summed (relationships only)
```

[`agents/s10-summarization/summarize.go`](https://github.com/Ding-Ye/learn-lightrag/blob/main/agents/s10-summarization/summarize.go) 的核心决策路径（约 30 LOC）：

```go
// Step 1: edge cases.
if len(descriptions) == 0 {
    return "", false, nil
}
if len(descriptions) == 1 {
    return descriptions[0], false, nil
}

// Step 2: heuristic — under threshold AND under budget → naive concat.
total := totalTokens(descriptions)
if len(descriptions) < countThreshold && total < budgetTokens {
    return strings.Join(descriptions, descriptionJoinSeparator), false, nil
}

// Step 3: map — chunk into contextSize-token windows.
chunks := chunkByTokens(descriptions, contextSize)

// Step 4: reduce — single-description chunks pass through.
llmUsed := false
partials := make([]string, 0, len(chunks))
for _, chunk := range chunks {
    if len(chunk) == 1 {
        partials = append(partials, chunk[0])
        continue
    }
    summary, err := callLLMSummary(ctx, p, descriptionType, descriptionName, chunk, budgetTokens)
    if err != nil { return "", llmUsed, fmt.Errorf("summarize chunk: %w", err) }
    partials = append(partials, summary)
    llmUsed = true
}

// Step 5: recurse if still too big.
joined := strings.Join(partials, descriptionJoinSeparator)
if tokenCount(joined) < budgetTokens && len(partials) < countThreshold {
    return joined, llmUsed, nil
}
return summarizeWithName(ctx, p, descriptionType, descriptionName, partials, ..., depth+1)
```

**四个不显然的点：**

1. **Threshold-AND-Budget**——必须**同时**满足才走拼接。单看 count（4 段 200 token 每段也炸）或单看 budget（30 段每段 5 token 也乱）都不对；上游 operate.py:222-226 用 `AND` 把两个条件锁在一起。
2. **每个 chunk 至少 2 段**——上游 operate.py:255-263 的「保证收敛」规则。如果某个 chunk 只有 1 段，reduce 阶段会原样透传，下一轮还是同样的 size，死循环。我们 force append 一段给它。
3. **单段 chunk 不调 LLM**——upstream 的 `if len(chunk) == 1: pass-through` 优化；归并 1 段是恒等操作，不烧钱。
4. **递归深度上限**——上游写 `while True`，理论上 reduce 阶段每轮严格压缩；但若 LLM 返回值碰巧没收缩（mock 测试会出现），我们用 `DefaultMaxRecursionDepth=3` 兜底，最后调一次 LLM 把 partials 砍下来，避免测试无限递归。

## What Changed / 与 s09 的变化

s09 给出的 Extractor 每次产出一个 `Entity` 记录（带 1 个 SourceID）。s10 是消费者：把 N 条同名 Entity 折成 1 条，描述合并、SourceIDs 取并集。

| 维度 | s09（抽取） | s10（归并） |
|---|---|---|
| 输入 | 一段 chunk 文本 | N 段 description 片段（每个来自一个 chunk） |
| 输出 | `[]Entity, []Relationship`（每条 1 个 SourceID） | 1 条合并 Entity/Relationship（SourceIDs 取并集） |
| LLM 触发 | 始终调（initial + N gleaning） | 条件触发：count >= threshold OR total > budget 才调 |
| Map-reduce | 无 | 有——切窗口 → 归并 → 递归 |
| Cache | sha256(chunk + ver) | （委托 caller；上游用 cache_type="summary"） |
| 上游引用 | operate.py:2883-3170 + prompt.py | operate.py:167-385 + prompt.py:185-218 |

**为什么必须 cache（虽然 s10 不实现）？**因为 47 段描述里如果只有 2 段变了，我们应当只重 summarize 这 2 段所属的那个窗口，不是把整个 entity 全部重跑。上游 `cache_type="summary"` 就是为这一步设计的。s10 留作 caller 的功课——`Provider.Complete` 之外裹一层 memoizer 即可。

**为什么 budget 默认 500、threshold 默认 6？**上游的 `summary_max_tokens=500` 与 `force_llm_summary_on_merge=6` 配置项。500 是 OpenAI embedding 实测上下文里描述向量稳定的上限——更大就开始稀释；6 是凭经验：≤5 段一般是「同一事实的不同表述」，直接拼。

## Try It / 动手试一试

```bash
cd /Users/yeding/learn-lightrag/agents/s10-summarization

# 默认 mock provider（CI 用的那条；离线、确定）
go run .

# -provider openai 故意没接，s02 才有真 OpenAI
# go run . -provider openai     # → returns clear error

# 跑测试（6 个 spec 测试 + interface contract + 2 个 sanity）
go test -v ./...
```

Demo 输出（节选）：

```
=== s10 demo: Eleanor Hartwell appears in 12 chunks ===
[a] raw concatenation (what s09 would have stored):
  ... 12 段描述，188 词 ...
[b] s10 summarized version:
  The merged subject ... professional accomplishments ...
[c] llmUsed: true
[d] compression: 188 → 66 tokens (35% of original)

=== fast-path demo: 3 descriptions, plenty of budget ===
llmUsed: false (expect false — under threshold AND under budget)
output is exact concat? true
```

测试矩阵：

| 测试 | 断言 |
|---|---|
| `TestSummarizeShortListSkipsLLM` | 3 段 + budget=500 → llmUsed=false，输出 = `\n\n` 拼接 |
| `TestSummarizeLongListInvokesLLM` | 12 段 + budget=60 → llmUsed=true，MockProvider.Calls > 0 |
| `TestSummarizeRespectsBudget` | 长输入压缩；最终 token < 2× budget |
| `TestSummarizeRecursesUntilFits` | 30 段 + 极紧 budget → MockProvider.Calls > 1（递归） |
| `TestSummarizeEmptyDescriptionsReturnsEmpty` | nil / `[]string{}` → `("", false, nil)` |
| `TestMergeEntityAggregatesSourceIDs` | 3 条 Eleanor 分别带 [c1]/[c2]/[c3] → 合并 1 条，SourceIDs 排序=[c1,c2,c3] |

加 3 个 sanity：`TestProviderInterfaceContract` 编译期 `*MockProvider` 满足 `Provider`、`TestMergeRelationshipsCanonicalizesEdge`（A→B 与 B→A 同一 edge，weight 累加）、`TestSingleDescriptionPassesThrough`（单段输入不调 LLM）。

## Upstream Source Reading / 上游源码阅读

```python
# lightrag/operate.py:167-227 (~50 LOC, 上游函数头部 + 决策树)
async def _handle_entity_relation_summary(
    description_type, entity_or_relation_name, description_list,
    separator, global_config, llm_response_cache=None,
) -> tuple[str, bool]:
    """Decision tree:
       1. If total tokens < context_size AND len < threshold → no LLM
       2. Otherwise summarize with LLM, possibly recursively
    """
    if not description_list:
        return "", False
    if len(description_list) == 1:
        return description_list[0], False

    tokenizer = global_config["tokenizer"]
    summary_context_size = global_config["summary_context_size"]
    summary_max_tokens = global_config["summary_max_tokens"]
    force_llm_summary_on_merge = global_config["force_llm_summary_on_merge"]

    current_list = description_list[:]
    llm_was_used = False

    while True:
        total_tokens = sum(len(tokenizer.encode(d)) for d in current_list)

        if total_tokens <= summary_context_size or len(current_list) <= 2:
            if (len(current_list) < force_llm_summary_on_merge
                and total_tokens < summary_max_tokens):
                # no LLM needed, just join
                return separator.join(current_list), llm_was_used
            else:
                final_summary = await _summarize_descriptions(...)
                return final_summary, True

        # Map phase: chunk descriptions into context_size windows;
        # each chunk has >= 2 descriptions to ensure progress.
        chunks = ...  # see operate.py:240-285

        # Reduce: per chunk, len==1 pass-through, else LLM call.
        new_summaries = []
        for chunk in chunks:
            if len(chunk) == 1:
                new_summaries.append(chunk[0])
            else:
                summary = await _summarize_descriptions(...)
                new_summaries.append(summary)
                llm_was_used = True
        current_list = new_summaries  # loop and re-evaluate
```

**怎么读：**

- `force_llm_summary_on_merge` 这个奇怪的命名其实就是「count threshold」——「合并时强制走 LLM 的最低段数」。Go 这边我们叫 `countThreshold`，更直白。
- `summary_max_tokens` vs `summary_context_size`：前者是**最终**输出的目标 budget（s10 的 `budgetTokens`），后者是**每个 map 窗口**的输入大小（s10 的 `contextSize`）。两个参数职责不同，命名易混。
- `while True` 循环看起来吓人，但每轮 reduce 阶段都严格压缩（多段 → 一段），所以理论上一定收敛。我们的 Go 版本加了 `DefaultMaxRecursionDepth=3` 兜底——mock provider 可能返回固定长度的响应，理论极限下不收敛。
- `cache_type="summary"` 是上游 `llm_response_cache` 的命名空间（与 extract / query 分开）。我们在 s10 不实现 cache（caller 责任），但 cache key 应该包含 `description_name + summaryPromptVersion`——否则改 prompt 不会让旧 cache 失效。

注释版上游摘录 + 阅读地图：[`upstream-readings/s10-summarization.py`](https://github.com/Ding-Ye/learn-lightrag/blob/main/upstream-readings/s10-summarization.py)。

下一节 [s11 双层检索](s11-query-modes.md) 是这一节的下游消费者：local 模式调 `vdbEntities.Query()` 检索的就是 s10 归并出来的 description embedding；没有 s10，那个向量索引会被同名实体的 47 个近似向量稀释，召回不准。
