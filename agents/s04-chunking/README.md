# s04 · 基于 token 的切分 / Token-based chunking with overlap

> 把 s01 里「按双换行硬切 + 长段截断」的占位换成上游 `chunking_by_token_size` 真正的「先 BPE 编码 → 滑动窗口 + 重叠 → 解码 → 写 ChunkOrderIndex」一套。tokenizer 是个二方法接口；既支持 `pkoukk/tiktoken-go` 的 cl100k_base（OpenAI 同款），也提供纯 stdlib 的 WhitespaceTokenizer 兜底（CI 离线必备）。
> Replace s01's "split-on-newlines + hard-cut long paragraphs" placeholder with the real upstream `chunking_by_token_size`: tokenize → sliding window with overlap → decode → tag with ChunkOrderIndex. The Tokenizer is a two-method interface; both `pkoukk/tiktoken-go` cl100k_base (the same BPE OpenAI uses) AND a stdlib-only `WhitespaceTokenizer` ship — the latter is what offline CI runs.

## 跑起来 / Run

```bash
cd agents/s04-chunking

# 默认 testdata/sample.txt + cl100k_base，1200/100 滑窗
# default testdata/sample.txt + cl100k_base, 1200/100 sliding window
go run .

# 自定义大小 / overlap，看更多 chunks
# tweak size/overlap to see more chunks
go run . -size 200 -overlap 30

# 完全离线 (无 tiktoken 下载)：用 WhitespaceTokenizer
# fully offline (no tiktoken download): use WhitespaceTokenizer
go run . -tokenizer ws -size 60 -overlap 10

# 自定义文档
go run . -doc /path/to/your.txt -size 800 -overlap 80

# rune-based fallback (适合非英文长文本)
# rune-based fallback (good for non-English long inputs)
go run . -split-by-character -size 200 -overlap 30

# 跑测试 (7 个，全部用 WhitespaceTokenizer，离线无网络)
# tests (7 tests, all use WhitespaceTokenizer — fully offline, no network)
go test -v ./...
```

> **关于 tiktoken 首次下载 / About the first-time tiktoken download**：
> `pkoukk/tiktoken-go` 在第一次 `tiktoken.GetEncoding("cl100k_base")` 时会从其 vendor URL 下载约 1.6 MB 的 BPE merges 文件，缓存到 `~/.tiktoken/`。后续调用直接读缓存。GitHub Actions runners 有出网，CI 也能跑通——但本地 air-gap 环境请先 `go run . -tokenizer ws` 验证逻辑。
> The first call to `tiktoken.GetEncoding("cl100k_base")` downloads a ~1.6 MB BPE merges file from the vendor URL and caches it under `~/.tiktoken/`. Subsequent calls hit the cache. GitHub Actions runners have outbound network so CI works; in air-gapped envs use `-tokenizer ws` instead.

## 文件 / Files

| 文件 / file | 作用 / role | LOC |
|---|---|---|
| `tokenizer.go` | `Tokenizer` 接口 + `TiktokenTokenizer` (cl100k_base) + `WhitespaceTokenizer` 兜底 / interface + tiktoken-backed impl + stdlib fallback | ~115 |
| `chunking.go` | `Chunk` 结构 + `ChunkByTokenSize`：默认滑窗 + `splitByCharacter` rune fallback + 输入校验 / struct + main API + character-fallback path + input validation | ~175 |
| `main.go` | CLI demo：读文件、跑切分、按 chunk 打 index/tokens/前 80 字符 / CLI demo: read file, chunk, print index/tokens/first-80-chars per chunk | ~80 |
| `chunking_test.go` | 6 个核心测试 + 1 个接口契约编译期检查，全部用 `WhitespaceTokenizer` 离线 / 6 core tests + 1 compile-time interface contract, all offline | ~210 |
| `testdata/sample.txt` | 约 600 词 Eleanor Hartwell 短传，跟 s01/s03 的人物保持一致 / ~600 words of fictional Eleanor Hartwell biography, same character as s01/s03 | — |

## 关键教学点 / Key teaching points

- **Tokenizer 是二方法接口 / The Tokenizer is a 2-method interface**：`Encode([]int)` + `Decode(string)`。这跟上游 Python 里的 `Tokenizer` Protocol 一比一对应。任何要换成 GPT-2 BPE / SentencePiece / 自训 BPE 的人都只需要再写 80 行就行。
- **滑动窗口的 step 是 `chunkTokenSize - overlapTokenSize` / The step size is `chunkTokenSize - overlapTokenSize`**：默认 1200/100 → step=1100。所以两个相邻 chunk 共享最后/前 100 个 token，这就是上游声称的「context preservation」load-bearing 实现。
- **末尾 chunk 可能更短 / The last chunk may be shorter**：`Tokens` 字段记 `min(chunkSize, len(tokens)-start)`，跟上游 `min(chunk_token_size, len(tokens) - start)` 一致。测试 `TestChunkExactlyAtBoundary` 锁住了 boundary 行为：N=chunkSize → 1 chunk；N=chunkSize+1 → 2 chunks。
- **`splitByCharacter=true` 是 rune-based fallback / `splitByCharacter=true` is the rune-based fallback**：纯 token 切分对没有空白的长字符串没办法（WhitespaceTokenizer 直接给 1 个 token）。fallback 切按 rune 数（默认 1 token ≈ 4 rune）切，再对每个窗口重新 BPE 编码拿真实 token 数。CJK / 代码片段适用。
- **接口编译期断言 / Compile-time interface assertion**：文件底部 `var _ Tokenizer = (*TiktokenTokenizer)(nil)`——任何 rename/重构会在 build 阶段炸出来，不会拖到第一次跑 tiktoken 才发现。

完整章节文档：[docs/zh/s04-chunking.md](../../docs/zh/s04-chunking.md) · [docs/en/s04-chunking.md](../../docs/en/s04-chunking.md)
