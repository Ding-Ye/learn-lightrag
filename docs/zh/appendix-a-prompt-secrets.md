---
title: "附录 A · 提示工程的秘密"
chapter: appendix-a
slug: appendix-a-prompt-secrets
est_read_min: 11
---

# 附录 A · 提示工程的秘密

> LightRAG 在 EMNLP'25 跑赢 vanilla RAG 和 GraphRAG 的真正原因，**不是它的代码结构**——是它对提示词的设计。这个附录解释**为什么**那些选择会赢，从而即使你换语言重写也不会丢掉它的精髓。

---

## 1. 为什么实体抽取用分隔符而不是 JSON

上游用 `<|#|>` 和 `<|>` 分隔符把抽取结果拍平成一行行 tuple，**不是** JSON。第一次看会觉得反直觉——JSON 不更结构化吗？

但分隔符格式有三个 LLM 时代特有的优势：

1. **格式错误尾部更短**。LLM 偶尔会在 JSON 中漏个引号或多个逗号，整段 JSON `json.Unmarshal` 失败，前面 99% 的输出全部丢。分隔符格式按行解析，一行坏掉只丢这一行——剩下的实体仍然能进 graph。
2. **`<|#|>` 几乎不可能在自然文本里出现**。比 `\n` 或 `,` 这种容易冲突的分隔符鲁棒得多。s09 的 parser 用 `strings.Split` 一行就解决。
3. **token 占用更少**。JSON 的 `{"name": "X", "type": "PERSON", "description": "..."}` 至少 30+ token；分隔符版 `entity<|>X<|>PERSON<|>...` 只要 ~12 token。乘以一个文档几百次抽取调用，省 50% token 成本。

**取舍**：分隔符格式不能用 OpenAI 的 `response_format={"type": "json_object"}` strict mode。但 LightRAG 的目标是**通用** LLM 后端（OpenAI / Anthropic / Bedrock / Ollama），strict mode 不是普遍可用，分隔符是更稳的最低公分母。

s09 的 `extraction_prompt.go` 完整保留这个设计。

---

## 2. Gleaning 循环：为什么 N 轮比一次大 prompt 强

朴素做法：一次 prompt 就让 LLM 输出**所有**实体。问题是 LLM 在抽取任务上有"早停偏差"——给定足够长的输出预算它仍会停在第 5-10 个实体，认为"够了"。

LightRAG 的 gleaning：第 1 轮抽完，把结果反馈给 LLM 同时问"还有遗漏吗？格式同上"。LLM 在第 2 轮通常能补出 30-60% 新实体（论文实验数据）。第 3 轮收益骤减。

**为什么续问比一次大 prompt 好**：
- 一次大 prompt 受 LLM 注意力衰减影响，对长输入分散关注。续问时 LLM 看到自己刚出过哪些，注意力被"剩下哪些没说"框住。
- 续问的成本可控——可以按 `MaxGleaningRounds` 截断（s09 默认 1 轮，研究场景可设 2-3 轮）。

**临界点**：超过 3 轮续问，新增实体几乎为 0，但仍要付 token 钱。s09 默认 `MaxGleaningRounds=1` 是经济选择；研究场景可以拉到 2-3。

---

## 3. 描述归并的阈值法则

s10 的 decision tree 不是花哨技巧——它是**避免冗余 LLM 调用**的工程经验：

```
if len(descriptions) < 6 AND total_tokens < 500:
    return strings.Join(descriptions, "\n\n")  // free
else:
    return llm_summarize(descriptions)         // ~$0.001/call
```

**6 这个数字怎么来的**：上游靠经验得出"≤5 个相关描述拼起来读不累；≥6 个开始重复"。我们 s10 沿用同一阈值。

**为什么超过阈值要 LLM 归并**而不是简单截断：
- 截断丢信息（最后两个 chunk 可能才是关键证据）。
- 拼接后塞 LLM 但不归并，LLM 在生成阶段又会被冗余信息分散注意力。
- 归并把"重复的"压缩、"互补的"保留。这是查询时上下文质量的最大杠杆。

**LLM 归并 vs 提取式摘要**：上游选 LLM 归并因为实体描述常包含**冲突信息**（"Eleanor 1834 年生" vs "Eleanor 出生于 1832 年"），归并 prompt 要求 LLM 标记 conflicts 而不是 silently 选一个。这是 LightRAG 在历史人物场景上做得好的原因。

---

## 4. 双层关键词为什么能让 mode 路由变 trivial

LightRAG 的"high-level vs low-level keywords"是 EMNLP 论文的核心 ablation。

**high-level keywords**：宏观主题、概念。例如查询"为什么 Tarvin Highlands 探险被取消"中的 high-level = ["explorations", "patronage", "fire incident"]。
**low-level keywords**：具体实体、专有名词。同一查询的 low-level = ["Tarvin Highlands", "Scrooge", "Eleanor"]。

为什么把这两层分开能让 mode 路由变简单：
- **local mode** = 用 low-level 在 vdbEntities 里找——按实体名直接命中节点。
- **global mode** = 用 high-level 在 vdbRelations 里找——按主题词命中跨实体的关系。
- **hybrid mode** = 都跑 + dedup。

如果不分开，查询"为什么 Tarvin Highlands 探险被取消"会同时在 entities 和 relations 索引里命中相同的 chunk，**返回的 top-K 高度重叠**——失去了 dual-level 的好处。分开后两边覆盖不同维度，hybrid 的 dedup 才有意义。

**有意思的反例**：如果你的语料库**只有人名**（比如族谱数据），high-level 关键词会很难抽出来，global mode 退化成噪声。LightRAG 在这种数据上没有显著优势——这是它的隐藏边界。

---

## 5. 成本-质量权衡：什么时候该开 gleaning

s09 默认 `MaxGleaningRounds=1`。完整抽取一份 100-chunk 的文档：
- 1 轮：100 LLM calls × 1 = 100 calls
- 2 轮 (gleaning=2)：100 × 3 = 300 calls
- 3 轮 (gleaning=3)：100 × 4 = 400 calls

按 GPT-4o-mini 的价格（$0.15/M input + $0.6/M output），100-chunk 抽取一次大约 $0.30。开 gleaning=3 飙到 $1.20。但召回率只多 10-15%。

**经济决策树**：
- 一次性研究项目（医疗 / 法律 docs，每条都珍贵）→ gleaning=2 或 3。
- 持续 ingest 海量文档（爬虫抓的网页）→ gleaning=1，靠覆盖率不靠召回。
- 边写边索引的 pipeline → 默认 1，再做一遍批量 gleaning=3 重 ingest 关键文档。

**附录 A 的精髓**：LightRAG 的"代码长得像一般的 RAG 框架"，但每个 prompt 设计点都是经过"成本-质量"实验取舍出来的。看 prompt.py 的人比看 operate.py 的人能更深刻理解这个项目。

---

## 6. 我们刻意没做的

- **Reranker (cross-encoder)**：在 retrieval 和 LLM 之间插一层 ms-marco 风格的重排器。能再涨 5-10% 召回但要额外 GPU；s11 留 hook，未实现。
- **Streaming response**：s02 的 `CompleteRequest.Stream` 字段已留，Phase G 的 multi-model addendum 只需在 OpenAI 端打开 `stream=true` 拼装 SSE，其他端各自实现。
- **Self-RAG 风格的引文校验**：让 LLM 在生成后再扫一遍引用是否在 retrieved 里。提升事实性但成本翻倍——研究场景才值得。

这三个都不是 LightRAG 的"秘密"，是 RAG 通用增强。读完附录 A 你应该能判断哪些"高级 RAG 技巧"对你的场景值得加。

---

**回到 11 节代码**：附录 A 解释的所有设计点都活在 s09 的 `extraction_prompt.go`、s10 的 `summarize_prompt.go`、s11 的 `keywords.go` 里。换语言重写时**先把这些 prompt 抄准**，再考虑代码结构。
