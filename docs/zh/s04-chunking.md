---
title: "s04 · 基于 token 的切分"
chapter: 4
slug: s04-chunking
est_read_min: 8
---

# s04 · 基于 token 的切分

> 教什么：把 s01 的「按双换行硬切 + 长段截断」推进到上游 LightRAG 真正在用的 `chunking_by_token_size`——基于 BPE token 数的 1200/100 滑动窗口，每一对相邻 chunk 共享 100 个 token 做 context preservation。引入仓库第一个外部 Go 依赖 `github.com/pkoukk/tiktoken-go`，并配一个纯 stdlib 的 `WhitespaceTokenizer` 兜底，让离线 CI / 测试不需要联网。

---

## Problem / 问题

s01 的 `ChunkByNewlines(text, maxRunes)` 在三个具体场景上会出问题：

1. **rune count 不等于 token count**：OpenAI 的 `gpt-4o-mini` 输入限制是 token 数，不是字符数。一段全英文 1200 rune 大概 ~250 token，但同样 1200 rune 的中文/代码可能就是 800-1500 token——按 rune 切完直接喂进 chat completion 偶尔就会炸 8K context。
2. **段落硬切丢上下文**：s01 在长段超过 `maxRunes` 时直接 `runes[start:end]` 一刀，相邻两个 chunk 中间的句子被切掉一半。retrieval 时 top-k chunk 召回的就是「半句话」，无论 LLM 多努力答案都是不完整的。这是为什么上游要保留 100 token overlap——chunk N 的尾巴 100 token 等于 chunk N+1 的开头，跨边界的句子两边都能见到。
3. **chunk 数量不可预测**：`ChunkByNewlines` 的 chunk 数取决于段落数和段落长度，对同一份文档跑两次大概率得到不同的 ChunkOrderIndex 范围（如果稍微改了换行）。这违反了 LightRAG 「chunk_id 是稳定 cite 凭证」的契约——s09 抽实体时 chunk_id 还要塞到 entity 的 source_ids 里持久化，不稳定就完蛋。

上游解法在 `lightrag/operate.py:102-166` 的 `chunking_by_token_size`：先 BPE 编码整篇文档，再固定步长滑窗 → 解码 → 写 `chunk_order_index`。s04 的工作是把这套 65 行 Python 翻译成约 200 行 Go，外加给它配一个不依赖网络的 fallback tokenizer 用于离线测试。

## Solution / 解决方案

四件事：

1. **`Tokenizer` 是二方法接口**：`Encode(text) []int` + `Decode(tokens) string`。这跟上游 Python 里的 `Tokenizer` Protocol 一一对应——`chunking_by_token_size` 全程只调用这两个方法，所以接口表面就是这两个。
2. **`TiktokenTokenizer` 包 `pkoukk/tiktoken-go` 的 cl100k_base**：这是 OpenAI 给 `text-embedding-3-small` / `gpt-4o-mini` 用的同款 BPE。第一次调用 `tiktoken.GetEncoding("cl100k_base")` 会从 vendor URL 拉一份约 1.6 MB 的 merges 文件缓存到 `~/.tiktoken/`，之后直接读缓存。
3. **`WhitespaceTokenizer` 是 stdlib 兜底**：`Encode = strings.Fields`，`Decode = strings.Join(parts, " ")`，词汇表在内存里递增地分配 ID。round-trip 不是 byte-stable（连续空白会被压缩），但「token 计数」和「相邻 chunk 内容拼接后能再 tokenize」两条性质都成立，这就够离线测试用了。
4. **`ChunkByTokenSize(docID, text, tokenizer, chunkTokenSize, overlapTokenSize, splitByCharacter)`**：默认走「encode → 步长 = chunkTokenSize-overlap 滑窗 → decode → emit」；`splitByCharacter=true` 切 rune（适合 CJK / 无空白长字符串）。输入校验：`chunkTokenSize > overlapTokenSize > 0` 必须成立；空文本返回 `ErrEmptyText`。

`Chunk` 结构是上游 `TextChunkSchema` (lightrag/base.py:71) 的 Go 翻版：`ContentDocID` + `Content` + `Tokens` + `ChunkOrderIndex`。s01 已经声明过这个结构，s04 把 `Tokens` 字段从「rune 计数 stub」换成真正的 BPE token 计数。

## How It Works / 工作原理

```
text = "lorem ipsum dolor sit amet ..."
        │
        ▼  tokenizer.Encode(text)
tokens = [101, 4023, 1991, 3920, 2017, 1232, ...]   len = 714

       ┌─────────────────────────────── chunkTokenSize = 60 ──────────────────────────────┐
window 0: tokens[0:60]                                                         step = 50
                ┌──── overlap 10 ────┐
                ▼                    ▼
window 1:        tokens[50:110]
                                ┌──── overlap 10 ────┐
                                ▼                    ▼
window 2:                       tokens[100:160]                                      ...

每个 window: tokenizer.Decode(slice).TrimSpace() → Chunk{
    ContentDocID:   docID,
    Content:        decoded text,
    Tokens:         min(chunkTokenSize, len(tokens) - start),   ← 末尾窗口可能更短
    ChunkOrderIndex: i,                                          ← 0, 1, 2, ...
}
```

载货核心 30 行（节选自 [`agents/s04-chunking/chunking.go`](https://github.com/Ding-Ye/learn-lightrag/blob/main/agents/s04-chunking/chunking.go)）：

```go
func chunkByTokens(
    docID, text string,
    tokenizer Tokenizer,
    chunkTokenSize, overlapTokenSize int,
) ([]Chunk, error) {
    tokens := tokenizer.Encode(text)
    if len(tokens) == 0 {
        return nil, ErrEmptyText
    }

    step := chunkTokenSize - overlapTokenSize
    out := make([]Chunk, 0, (len(tokens)/step)+1)
    idx := 0
    for start := 0; start < len(tokens); start += step {
        end := start + chunkTokenSize
        if end > len(tokens) {
            end = len(tokens)
        }
        body := strings.TrimSpace(tokenizer.Decode(tokens[start:end]))
        if body == "" { continue } // 防御：decode 出纯空白时跳过
        out = append(out, Chunk{
            ContentDocID:    docID,
            Content:         body,
            Tokens:          end - start,           // == min(chunkTokenSize, len-start)
            ChunkOrderIndex: idx,
        })
        idx++
        if end == len(tokens) { break } // 末尾窗口写完就退出，不让 step 再前进一步重写
    }
    return out, nil
}
```

**4 个非显然之处**：

1. **`step = chunkTokenSize - overlapTokenSize`，不是 `chunkTokenSize`**：滑动而非接力。每次窗口起点前进 `step` 个 token，窗口长度仍然 `chunkTokenSize`——所以最后 `overlapTokenSize` 个 token 会同时出现在 chunk N 的末尾和 chunk N+1 的开头。这是 100 token overlap 的实现细节：不是「在 chunk 之间插入 100 token 桥梁」，而是「每个 chunk 的尾巴 100 token 复用进下一个 chunk 的头」。
2. **末尾 chunk 的 Tokens 是 `end - start`，不是 `chunkTokenSize`**：当 `start + chunkTokenSize > len(tokens)` 时 `end` 会被 clamp 到 `len(tokens)`，所以 `end - start` 等于上游的 `min(chunk_token_size, len(tokens) - start)`。测试 `TestChunkExactlyAtBoundary` 锁住了这个语义。
3. **`break` 在 `end == len(tokens)` 时**：避免最后一个窗口被多写一次。比如 122 token + chunkSize=60 + overlap=10 (step=50)：start ∈ {0, 50, 100, 150}。start=100 时 end=clamp(160, 122)=122——这是末尾窗口；如果继续，start=150 已超出 `len(tokens)` 循环条件直接退出，但 start=150 的写入会变成 `tokens[150:122]` 即空切片，触发 `body == ""` 的 skip 分支。`break` 是显式退出更可读。
4. **接口编译期断言**：文件底部 `var _ Tokenizer = (*TiktokenTokenizer)(nil)` + `var _ Tokenizer = (*WhitespaceTokenizer)(nil)`——任何把 `Encode` 改成 `Tokenize` 这种 rename 都会在 `go build` 阶段就炸掉，不会等到运行时第一次进 tiktoken 路径才发现。

## What Changed / 与 s01 的变化

```diff
 // s01 chunking.go
-func ChunkByNewlines(docID, text string, maxRunes int) []Chunk {
-    if maxRunes <= 0 { maxRunes = 1200 }
-    var out []Chunk
-    idx := 0
-    for _, para := range strings.Split(text, "\n\n") {
-        para = strings.TrimSpace(para)
-        if para == "" { continue }
-        runes := []rune(para)
-        for start := 0; start < len(runes); start += maxRunes {
-            end := start + maxRunes
-            if end > len(runes) { end = len(runes) }
-            out = append(out, Chunk{
-                ContentDocID:    docID,
-                Content:         string(runes[start:end]),
-                Tokens:          end - start,            // ← rune 计数 stub
-                ChunkOrderIndex: idx,
-            })
-            idx++
-        }
-    }
-    return out
-}

+// s04 chunking.go
+func ChunkByTokenSize(
+    docID, text string,
+    tokenizer Tokenizer,
+    chunkTokenSize, overlapTokenSize int,
+    splitByCharacter bool,
+) ([]Chunk, error) {
+    if tokenizer == nil { return nil, fmt.Errorf("s04: tokenizer is nil") }
+    if !(overlapTokenSize > 0 && chunkTokenSize > overlapTokenSize) {
+        return nil, fmt.Errorf("%w: got chunk=%d overlap=%d",
+            ErrInvalidChunkSize, chunkTokenSize, overlapTokenSize)
+    }
+    if strings.TrimSpace(text) == "" { return nil, ErrEmptyText }
+    if splitByCharacter {
+        return chunkByCharacter(docID, text, tokenizer, chunkTokenSize, overlapTokenSize)
+    }
+    return chunkByTokens(docID, text, tokenizer, chunkTokenSize, overlapTokenSize)
+}
```

主要差异：

- **`Tokens` 字段语义变了**：s01 的 `Tokens = end - start` 是 rune 计数；s04 的 `Tokens = end - start` 是 BPE token 计数。同一篇 600 词英文文档，s01 给出的 `Tokens` 值大概 3000+，s04 用 cl100k_base 给出的是 ~700。下游 s07 vector store 的 token-budget 截断要 token 数，所以这个修复是必须的。
- **签名多一个 `tokenizer Tokenizer` 参数 + `splitByCharacter bool`**：依赖注入的 tokenizer 让 CI 用 `WhitespaceTokenizer`、生产用 `TiktokenTokenizer`，调用方一行不改。`splitByCharacter` 是上游同名参数，给非英文长文本兜底。
- **错误返回**：s01 永不出错（最差返回空 slice）；s04 校验输入并返回 `ErrEmptyText` / `ErrInvalidChunkSize`，配合 `errors.Is` 让调用方在 `pipeline.Insert` 里能区分「文档真的空」vs「调用方传错参数」。

s05 的 KV 章节会把 chunks 持久化到 JSON；那时候 s04 的 `Chunk.Tokens` 和 `Chunk.ChunkOrderIndex` 字段会成为 KV value 的一部分，所以这两个字段的语义稳定性现在就要锁死。

## Try It / 动手试一试

```bash
cd agents/s04-chunking

# 默认 cl100k_base + 1200/100，对 600 词的样本只切出 1 个 chunk
# default cl100k_base + 1200/100 — sample doc gives 1 chunk
go run .
# == 1 chunks ==
#   [00] tokens= 714  Eleanor Hartwell was born in 1843...

# 把 chunkSize 调小看滑窗效果
# shrink chunkSize to see windowing in action
go run . -tokenizer ws -size 60 -overlap 10
# == 11 chunks ==
#   [00] tokens=  60  Eleanor Hartwell was born in 1843...
#   [01] tokens=  60  in her father's careful hand, and presented herself...
#   ...相邻 chunk 末尾/开头共享 10 个 word

# 不带 tiktoken 的离线 demo
# offline demo without tiktoken
go run . -tokenizer ws

# splitByCharacter fallback (rune 切分)
go run . -split-by-character -size 200 -overlap 30

# 跑测试 (7 个，全部 WhitespaceTokenizer 离线)
go test -v ./...
# === RUN   TestChunkRespectsTokenLimit          每个 chunk Tokens <= chunkTokenSize
# === RUN   TestChunkOverlapPreserved            相邻 chunk 共享 overlap 个 token
# === RUN   TestChunkOrderIndexMonotonic         indices = 0, 1, 2, ... 连续
# === RUN   TestChunkSplitByCharacterFallback    无空白长字符串能切多个 chunk
# === RUN   TestChunkEmptyTextReturnsEmpty       空输入 → 0 chunks + ErrEmptyText
# === RUN   TestChunkExactlyAtBoundary           chunkSize → 1 chunk; chunkSize+1 → 2
# === RUN   TestTokenizerInterfaceContract       编译期 + 运行期 round-trip 检查
```

第一次跑 `-tokenizer tiktoken` 会触发 `pkoukk/tiktoken-go` 从 vendor URL 下载约 1.6 MB BPE merges 文件，缓存到 `~/.tiktoken/`。完全 air-gap 的环境用 `-tokenizer ws` 即可。

## Upstream Source Reading / 上游源码阅读

上游 `chunking_by_token_size` 的载货 50 行（节选自 `lightrag/operate.py:102-166`，完整注解见 [`upstream-readings/s04-chunking.py`](../../upstream-readings/s04-chunking.py)）：

```python
def chunking_by_token_size(
    tokenizer: Tokenizer,
    content: str,
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    chunk_overlap_token_size: int = 100,
    chunk_token_size: int = 1200,
) -> list[dict[str, Any]]:
    tokens = tokenizer.encode(content)
    results: list[dict[str, Any]] = []
    if split_by_character:
        # ... (节选省略：split_by_character_only 严格模式 + per-piece 长度校验)
        raw_chunks = content.split(split_by_character)
        new_chunks = []
        for chunk in raw_chunks:
            _tokens = tokenizer.encode(chunk)
            if len(_tokens) > chunk_token_size:
                for start in range(
                    0, len(_tokens), chunk_token_size - chunk_overlap_token_size
                ):
                    chunk_content = tokenizer.decode(
                        _tokens[start : start + chunk_token_size]
                    )
                    new_chunks.append(
                        (min(chunk_token_size, len(_tokens) - start), chunk_content)
                    )
            else:
                new_chunks.append((len(_tokens), chunk))
        for index, (_len, chunk) in enumerate(new_chunks):
            results.append({
                "tokens": _len,
                "content": chunk.strip(),
                "chunk_order_index": index,
            })
    else:
        for index, start in enumerate(
            range(0, len(tokens), chunk_token_size - chunk_overlap_token_size)
        ):
            chunk_content = tokenizer.decode(tokens[start : start + chunk_token_size])
            results.append({
                "tokens": min(chunk_token_size, len(tokens) - start),
                "content": chunk_content.strip(),
                "chunk_order_index": index,
            })
    return results
```

**对照阅读要点**：

- **`Tokenizer` 是 Protocol，不是基类**：上游 `Tokenizer` 是 `typing.Protocol`，鸭子类型的接口。Go 里我们用普通 interface（也是结构性的）——一比一翻译。s06 的 `OpenAIEmbedder` 章节会有同样的 Provider Protocol → Go interface 模式。
- **`split_by_character: str | None` 上游是字符串，s04 用 bool**：上游传一个分隔符字符串（默认 `None`），匹配上就走 split-then-window 路径。s04 简化成 `bool` + 内置 rune-based fallback——少一个 API 维度，但「无空白长串可切」的核心需求满足了。如果学习者要复刻精确语义，可以给 `splitByCharacter` 加成 `*string` 类型，nil 走默认路径，否则当分隔符。
- **`split_by_character_only=True` 严格模式 s04 没翻**：上游有个「split 之后任何 piece 超长就 raise」的严格分支；s04 没翻，因为它的语义是「我承诺自己的分隔符切完就够小，超长直接报错」。教学场景这种 ergonomic 校验放到调用方更清晰；课程练习里可以补上。
- **`chunk.strip()` vs `strings.TrimSpace`**：上游每个 chunk 内容做 strip；s04 也做了 `strings.TrimSpace`，且额外 skip `body == ""` 的退化情况。两边等价。
- **末尾 chunk 长度公式**：上游 `min(chunk_token_size, len(tokens) - start)`；s04 `end - start` 其中 `end := min(start+chunkTokenSize, len(tokens))`——代数上一致。
- **`results` 里没有 `full_doc_id` / `chunk_id` 字段**：上游这里只返回内容相关字段，docID / chunk_id 在调用层 `lightrag/lightrag.py:apipeline_process_enqueue_documents` 里拼接进去。s04 把 `ContentDocID` 直接当参数传入——少一层转译，但语义一致。

**想读更多**：上游 `lightrag/lightrag.py:1500-1700` 区段（`apipeline_process_enqueue_documents`）展示了 `chunking_by_token_size` 的真实调用上下文——它在 doc-status 转 PROCESSING 之后、embedding-batch 之前调用，每篇文档调一次拿到 `list[dict]` 之后再 `compute_mdhash_id(content)` 给每个 chunk 打 `chunk_id`。s09（实体抽取）会把这些 chunk 内容喂给 LLM；那时候 s04 的 ChunkOrderIndex 会变成 entity 的 `source_ids` 排序键。

---

**下一节预告**：s05 把 s01 的 `sync.Map` 换成 JSON 持久化的 `JSONKVStore`，多出来一个核心方法 `FilterMissing(ids) []string`——「这批 chunk_id 里哪些还没存？」这是上游 `apipeline_process_enqueue_documents` 决定哪些 chunk 需要嵌入的关键 primitive，也是 s04 切完 chunk 后第一站会经过的存储层。
