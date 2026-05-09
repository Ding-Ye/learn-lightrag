---
title: "s09 · 实体关系抽取与 gleaning"
chapter: 9
slug: s09-extraction
est_read_min: 12
---

# s09 · 实体关系抽取与 gleaning

> 这一节做什么：用 LLM 自己当 extractor，给它一个**带分隔符**（不是 JSON）的输出契约——`entity<|#|>name<|#|>type<|#|>desc` 一行一个，关系是 `relation<|#|>src<|#|>tgt<|#|>keywords<|#|>desc`，整段以 `<|COMPLETE|>` 收尾。然后做一个 **gleaning 循环**：把上一轮的对话当 history 再问一遍"还漏了什么"，N 轮叠加。最后用 `sha256(chunk + prompt_version)` 当缓存键，避免对同一个 chunk 重复抽。对照上游 `lightrag/operate.py:2883-3170` 的 `extract_entities` + `lightrag/prompt.py` 的 `entity_extraction_system_prompt`。

---

## Problem / 问题

s08 已经把图存储这块做完了——`UpsertNode` / `UpsertEdge` 都准备好了。但是**图是空的**。给一个 doc，谁来告诉这个图："Scrooge 是个 PERSON，Marley 也是 PERSON，他俩是 partner"？

你可能会想"写一个正则吧"——这条路一秒钟就能走死：

1. **正则不认语义。**"Scrooge was a partner of Marley" 的语义靠介词 `of` + 名词在前——但正则只能匹配字面 token 序列。下一句"Marley, Scrooge's old partner, ..." 整个反过来，正则模式直接不匹配。
2. **Schema 是开放的。**Entity 类型可能是 `PERSON / ORG / EVENT / CONCEPT / ...`，relationship keywords 更是任意短语（"power dynamics" / "conflict resolution" / ...）。穷举不完。
3. **跨段落的关系。**"Scrooge ... [400 字] ... Marley ... [200 字] ... they were partners"——这种长距离指代正则根本接不住，需要语义级别的「整段读完再总结」。
4. **没有黄金数据集训分类器。**就算训也训不出能 zero-shot 处理任何文档的模型。

LightRAG 直接选了把锅甩给 LLM 这条路：用 GPT-4o-mini 当 extractor，一次几分钱，召回率高得离谱。但这又引入两个新坑：

- **JSON 输出脆。**LLM 偶尔会忘掉一个引号、漏掉一个 `}`、把 unicode 字符没转义就吐出来。`json.Unmarshal` 直接 panic，整个 chunk 的抽取数据全废。
- **一遍抽不全。**LLM 会"提早收手"——抽到第 5 个 entity 觉得差不多了就 `<|COMPLETE|>`。但实际文档里可能还有第 6/7/8 个。

s09 一次性把这两件事修掉——分隔符替代 JSON、gleaning 循环替代单遍抽。

## Solution / 解决方案

四步：

1. **`Extractor` 接口持有 `Provider` + `MaxGleaningRounds` + 可选 `Cache` + 可选 `Logger`。** `Extract(ctx, chunk Chunk) ([]Entity, []Relationship, error)` 是唯一入口。Provider 来自 s02 的契约（在 s09 里**重新声明**，不 import——session 自包含）；Chunk 来自 s04 的形状（同样重声明）。
2. **分隔符契约（不是 JSON）。** 上游用 `<|#|>` 当字段分隔符（`tuple_delimiter`）、`\n` 当记录分隔符、`<|COMPLETE|>` 当结束符。Entity 是 4 个字段：`entity<|#|>name<|#|>type<|#|>description`；Relation 是 5 个字段：`relation<|#|>src<|#|>tgt<|#|>keywords<|#|>description`。这套字符串组合**几乎不可能**在自然文本里出现，所以即使 LLM 吐了一段散文当 description，分隔符依然能切对。
3. **Gleaning 循环。** 第一轮抽完之后，把 `(user prompt, assistant response)` 拼成 history，再发一个 continuation user prompt（"上一轮漏了什么？补充输出"），再解析、再 dedup、再合并。`MaxGleaningRounds == 1` 默认抽两遍（初轮 + 1 轮 gleaning），跟上游 `entity_extract_max_gleaning=1` 对齐。提前停止条件：本轮没抽到任何**新** entity（已存在的不算），就 break。
4. **Cache 用 `KVStore` 接口的子集。** Hash key = `sha256(chunk.Content || promptVersion)`——promptVersion 是个 const 字符串，prompt 改了就升号让旧缓存失效。`Cache.Get(key)` 命中就跳过 Provider 调用直接 parse 缓存值；miss 就调用 Provider，再 `Cache.Upsert(key, raw_text)`。CLI 末尾打印命中率。

## How It Works / 工作原理

```
                       Extractor.Extract(ctx, chunk)
   ┌─────────────────────────────────────────────────────────────┐
   │                                                             │
   │   Step 1  cache key = sha256(chunk.Content || version)      │
   │           │                                                 │
   │           ├─ hit  ──► parse cached raw_text  ──► merge      │
   │           │                                                 │
   │           └─ miss ──► Provider.Complete(sysPrompt+chunk)    │
   │                            │                                │
   │                            ▼                                │
   │   Step 2  parser:                                           │
   │     split by "\n" + strip "<|COMPLETE|>"                    │
   │     for each line:                                          │
   │       fields = strings.Split(line, "<|#|>")                 │
   │       if fields[0] == "entity"   && len==4 → Entity         │
   │       if fields[0] == "relation" && len==5 → Relationship   │
   │       else: skip silently                                   │
   │                                                             │
   │   Step 3  gleaning loop (MaxGleaningRounds times):          │
   │     msgs = [user₀, assistant₀, user_continue, ...]          │
   │     resp = Provider.Complete(...)                           │
   │     parse resp → newEnts, newRels                           │
   │     dedup by Name (entity) / (Src,Tgt) (rel)                │
   │     if no NEW entity: break early                           │
   │                                                             │
   │   Step 4  Cache.Upsert(key, raw_text)                       │
   │                                                             │
   │   return entities, relationships, nil                       │
   └─────────────────────────────────────────────────────────────┘

   多 chunk 跨调用合并（Extractor 持有的 cache + 调用方的 dedup）：

       chunk-1 ──► Extract ──► [Entity{X, sources:[chunk-1]}]
       chunk-2 ──► Extract ──► [Entity{X, sources:[chunk-2]}]
                                  │
                                  ▼  调用方负责按 Name 合并
       merged: Entity{X, sources:[chunk-1, chunk-2]}
```

[`agents/s09-extraction/parser.go`](https://github.com/Ding-Ye/learn-lightrag/blob/main/agents/s09-extraction/parser.go) 的解析主循环（关键 ~30 行）：

```go
func parseExtractionOutput(s string) ([]Entity, []Relationship, error) {
    s = strings.ReplaceAll(s, completionDelimiter, "")
    var entities []Entity
    var relationships []Relationship
    for _, raw := range strings.Split(s, "\n") {
        line := strings.TrimSpace(raw)
        if line == "" {
            continue
        }
        // strip optional surrounding parens like "(entity<|#|>...)"
        line = strings.Trim(line, "()")
        fields := strings.Split(line, tupleDelimiter)
        if len(fields) < 2 {
            continue
        }
        kind := strings.ToLower(strings.TrimSpace(fields[0]))
        switch {
        case strings.Contains(kind, "entity") && len(fields) >= 4:
            entities = append(entities, Entity{
                Name:        strings.TrimSpace(fields[1]),
                Type:        strings.ToLower(strings.TrimSpace(fields[2])),
                Description: strings.TrimSpace(fields[3]),
            })
        case strings.Contains(kind, "relation") && len(fields) >= 5:
            relationships = append(relationships, Relationship{
                SrcID:       strings.TrimSpace(fields[1]),
                TgtID:       strings.TrimSpace(fields[2]),
                Keywords:    strings.TrimSpace(fields[3]),
                Description: strings.TrimSpace(fields[4]),
                Weight:      1.0,
            })
        }
    }
    return entities, relationships, nil
}
```

**四个不太显眼的点：**

1. **「记录分隔符」是 `\n` 而不是 `<|#|>`。** 上游用了**两个**分隔符层级：`tuple_delimiter=<|#|>` 切字段，`\n` 切记录，`<|COMPLETE|>` 标整段结束。我们解析时先剥掉 `<|COMPLETE|>`、按 `\n` 切，再每行按 `<|#|>` 切——两个层级独立切，互不干扰。这是为什么 description 字段里能塞下完整一句话却不会被错切。
2. **「entity」前缀只看 `Contains` 不看 `==`。** 上游 `_handle_single_entity_extraction` 用 `"entity" in record_attributes[0]`——容忍 LLM 输出 `("entity"<|#|>...)` 这种带括号、带引号的污染。我们的 `strings.Contains(kind, "entity")` 一行做掉同样的事。`relation` / `relationship` 也一样兼容（上游注释说"treat interchangeable"）。
3. **少字段不报错，直接 skip。** Robustness 第一原则：LLM 偶尔少一个字段，宁可丢这一行也不能 panic 整个 chunk。`len(fields) < 4` 走 default 分支不进 case，这一行就没了。**测试** `TestExtractionMalformedOutputFallsBack` 就专门验证这条。
4. **Gleaning 是 history-based，不是 prompt-stuffing。** 上游 continuation prompt 短到只有"补漏"四字——长度优势是因为上一轮的 user prompt + assistant response **都在 history 里**，LLM 看得到上轮抽了什么，自然知道补什么。我们 Go 端用 `Messages: append(prev, userContinue)` 还原同样的语义，每轮都把上一轮的 user/assistant 对话叠回 messages 里。

## What Changed / 与 s08 的变化

s08 把图存储做完了但**图是空的**——`UpsertNode` 只在 main.go 的 demo 里被手工调用。s09 是第一个**自动**填图的环节：把 chunk 喂进去，LLM 把实体/关系吐出来，调用方再去喂 `UpsertNode` / `UpsertEdge`。

| 维度 | s08（图存储） | s09（实体抽取） |
|---|---|---|
| 数据来源 | 调用方手工传 Entity / Relationship 字面值 | 从 chunk 文本通过 LLM 抽 |
| LLM 调用次数 | 0 | 每个 chunk 1 + N 次（N = MaxGleaningRounds） |
| 输出契约 | Go struct（Name, Type, ...） | 分隔符文本 → 解析成同样的 struct |
| 缓存 | 无（图本身就是状态） | 有——`sha256(chunk+ver)` 命中跳过 LLM |
| 容错策略 | 图不容错（边失效就崩） | 解析容错（少字段、错前缀都 skip） |
| 测试关注点 | BFS 子图、tie-break | 解析正确、gleaning 累加、cache 命中 |
| 上游对应 | `kg/networkx_impl.py` | `operate.py:2883-3170 + prompt.py` |

**为什么 cache 必须有？** 一个 100-chunk 的文档，MaxGleaningRounds=1 就是 200 次 LLM 调用。如果只是改了图的 BFS 算法重跑 ingestion，理应不重抽——因为 chunk 没变、prompt 没变，结果就**应该**完全一样。这是 LightRAG 在大文档上能多次迭代调试的关键基础设施。

**为什么 gleaning 默认就开（rounds=1）？** 上游 `entity_extract_max_gleaning=1` 是默认值，不是可选。原因是**单轮召回率太低**——LLM 抽到 5-6 个 entity 就开始想 `<|COMPLETE|>` 收尾，但 chunk 里实际可能有 10 个。一轮 gleaning 把召回率从 ~60% 拉到 ~85%（论文实测）。第二轮收益骤降到 +5%，所以默认就停在 1。

## Try It / 动手试一试

```bash
cd /Users/yeding/learn-lightrag/agents/s09-extraction

# 1. 跑 mock provider 抽 3 chunk（不联网，CI 用的就是这个）
go run . -provider mock -rounds 1 -doc ./testdata/sample.txt

# 2. 看看 gleaning 怎么累加（rounds=2 比 rounds=1 多抽一轮）
go run . -provider mock -rounds 2 -doc ./testdata/sample.txt

# 3. 跑同一个 doc 两次——第二次 cache 全部命中
go run . -provider mock -rounds 1 -doc ./testdata/sample.txt
go run . -provider mock -rounds 1 -doc ./testdata/sample.txt   # cache hit ratio = 100%

# 4. 跑测试（6 个 + 接口契约 1 个）
go test -v ./...
```

demo 输出（节选）：

```
=== chunk 1: extracted 3 entities, 2 relations  (cache MISS) ===
  entity   Scrooge       type=person       "a miserly businessman..."
  entity   Marley        type=person       "Scrooge's deceased business partner"
  entity   Christmas     type=event        "the central holiday in the story"
  relation Scrooge -- Marley   keywords=partnership   "...were business partners"

=== chunk 2: extracted 2 entities, 1 relation  (cache MISS) ===
  ...

cache hit ratio: 0/3 chunks (0%)   ← first run

=== second run on same doc ===
cache hit ratio: 3/3 chunks (100%)   ← all served from sha256-keyed cache
```

测试矩阵：

| 测试 | 断言 |
|---|---|
| `TestExtractorInterfaceContract` | 编译期：`*Extractor` 满足 internal 接口契约 |
| `TestExtractionParsesDelimitedTuples` | 给定一段已知分隔符文本，parser 输出 entities + relations 全对 |
| `TestExtractionGleaningAddsNewEntities` | rounds=2，第一轮 2 个、第二轮 1 新 → 累计 3 个 unique |
| `TestExtractionCacheAvoidsDoubleCall` | 同 chunk 调两次 Extract，provider 调用次数 = 1 |
| `TestExtractionMalformedOutputFallsBack` | 缺字段 / 错前缀 / 空行 → parser 返回空，不 panic |
| `TestExtractionMergesSourceIDsAcrossChunks` | chunk-1 和 chunk-2 都抽到 entity X → 合并后 sources=[chunk-1, chunk-2] |
| `TestExtractionRespectsContextCancel` | provider sleep + ctx cancel → Extract 返回 context.Canceled |

## Upstream Source Reading / 上游源码阅读

```python
# lightrag/operate.py:2949-3050（节选；完整 extract_entities ~290 行）
# 把 system prompt 格式化好、调 LLM、解析、跑 gleaning、合并——主路径就这几十行。

# Get initial extraction
entity_extraction_system_prompt = PROMPTS[
    "entity_extraction_system_prompt"
].format(**context_base)
entity_extraction_user_prompt = PROMPTS["entity_extraction_user_prompt"].format(
    **{**context_base, "input_text": content}
)
entity_continue_extraction_user_prompt = PROMPTS[
    "entity_continue_extraction_user_prompt"
].format(**{**context_base, "input_text": content})

final_result, timestamp = await use_llm_func_with_cache(
    entity_extraction_user_prompt,
    use_llm_func,
    system_prompt=entity_extraction_system_prompt,
    llm_response_cache=llm_response_cache,
    cache_type="extract",
    chunk_id=chunk_key,
)

history = pack_user_ass_to_openai_messages(
    entity_extraction_user_prompt, final_result
)

# 初轮：分隔符解析
maybe_nodes, maybe_edges = await _process_extraction_result(
    final_result, chunk_key, timestamp, file_path,
    tuple_delimiter=context_base["tuple_delimiter"],
    completion_delimiter=context_base["completion_delimiter"],
)

# Gleaning loop（实际上游写法只跑 1 轮，不是 N 轮——参数名是
# entity_extract_max_gleaning 但语义就是「再跑这一轮还是跳过」）
if entity_extract_max_gleaning > 0:
    glean_result, timestamp = await use_llm_func_with_cache(
        entity_continue_extraction_user_prompt,
        use_llm_func,
        system_prompt=entity_extraction_system_prompt,
        llm_response_cache=llm_response_cache,
        history_messages=history,
        cache_type="extract",
        chunk_id=chunk_key,
    )
    glean_nodes, glean_edges = await _process_extraction_result(
        glean_result, chunk_key, timestamp, file_path,
        tuple_delimiter=context_base["tuple_delimiter"],
        completion_delimiter=context_base["completion_delimiter"],
    )
    # 合并：新 entity 直接加，已有 entity 比较 description 长度取长的
    for entity_name, glean_entities in glean_nodes.items():
        if entity_name in maybe_nodes:
            if glean_desc_len > original_desc_len:
                maybe_nodes[entity_name] = list(glean_entities)
        else:
            maybe_nodes[entity_name] = list(glean_entities)
```

**怎么读这段：**

- `pack_user_ass_to_openai_messages(user_prompt, response)` 是把 `(user, assistant)` 一对儿打包成 messages list 给下一轮。Go 端我们直接 `[]Message{{Role:"user", ...}, {Role:"assistant", ...}}` append——两边语义完全一致，只是 Python 的 helper 是 1 行 dict、Go 的是 2 行 struct。
- `use_llm_func_with_cache` 的 `cache_type="extract"` 是 cache 的命名空间——LightRAG 的 KV cache 同时缓存 extract / summarize / query 三类调用，用 namespace 隔离。我们 Go 端只缓存 extract，所以省掉这个字段，hash 直接进根 namespace。
- 合并策略「比较 description 长度，留长的」是个细节：gleaning 轮可能给同一个 entity 一个**更详细**的 description（因为 LLM 二刷时心里有上下文）。上游用长度做代理。我们的实现里第一版采用「先到先得」（`if !exists` 才插入），把长度比较留作扩展练习——最高效是对比测试时引入。
- `_process_extraction_result` 的容错是出名的：split by `\n`，再用 split-by-multi-markers 修复 LLM 把 `<|#|>` 错当记录分隔符、修复 `entity` 前缀缺失等等。我们 Go 端的 `parser.go` 抓重点：split + len-check + skip——不试图修复，而是「能 parse 就 parse、不能就丢」，Robustness 通过宽容路径而非修复路径实现。

带注解的版本+阅读地图见 [`upstream-readings/s09-extraction.py`](https://github.com/Ding-Ye/learn-lightrag/blob/main/upstream-readings/s09-extraction.py)。

下一节 [s10 描述归并](s10-summarization.md) 会处理"同一个 entity 在 47 个 chunk 里出现"的问题——把那 47 段 description 用 LLM 二次归并成一段。再下一节 s11 的 local 模式会把 s09 抽出来的 entity 当种子做 `vdbEntities.Query`。s09 是「图层从空到有」的拐点。
