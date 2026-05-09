package main

import (
	"errors"
	"fmt"
	"strings"
)

// Chunk mirrors upstream `TextChunkSchema` (lightrag/base.py:71) and is
// the canonical shape every later session re-types.  Compared to s01:
//
//   - `Tokens` is the real BPE-token count (or whitespace-token count
//     in WhitespaceTokenizer mode), NOT the rune-count stub from s01.
//   - `ChunkOrderIndex` is contiguous 0..N-1, set by the chunker; downstream
//     consumers (s07 vector store, s09 extraction) use it for citation
//     ordering.
//   - `ContentDocID` is the parent doc ID — typically the MD5 from s03.
//     s04 doesn't compute it; the caller passes whatever ID is canonical
//     in their pipeline.
type Chunk struct {
	ContentDocID    string
	Content         string
	Tokens          int
	ChunkOrderIndex int
}

// Sentinel errors so callers can distinguish "input was bad" from
// "tokenizer returned something weird."  Use errors.Is(err, ErrXxx).
var (
	ErrEmptyText        = errors.New("s04: text is empty")
	ErrInvalidChunkSize = errors.New("s04: chunkTokenSize must be > overlapTokenSize > 0")
)

// ChunkByTokenSize is the Go port of upstream
// `chunking_by_token_size` (lightrag/operate.py:102-166).  Algorithm:
//
//  1. Tokenize the whole document once.
//  2. If `splitByCharacter` is true, fall back to the rune-based path
//     described below (matches upstream `split_by_character` mode for
//     non-English docs that tiktoken handles poorly — CJK without
//     whitespace, code blocks, etc.).
//  3. Otherwise slide a window of size `chunkTokenSize` across the
//     token stream with step `chunkTokenSize - overlapTokenSize`, decode
//     each window, and emit one Chunk per window with monotonic
//     `ChunkOrderIndex`.
//
// Validation: `chunkTokenSize > overlapTokenSize > 0` MUST hold or we
// return ErrInvalidChunkSize.  Empty `text` returns no chunks AND no
// error — matching upstream's behavior of "tokenize empty -> no
// windows -> empty list".  The brief asks us to error on empty input;
// we DO return ErrEmptyText, leaving "tokenize-then-empty-windows" as
// the legitimate "no chunks" path.
func ChunkByTokenSize(
	docID, text string,
	tokenizer Tokenizer,
	chunkTokenSize, overlapTokenSize int,
	splitByCharacter bool,
) ([]Chunk, error) {
	if tokenizer == nil {
		return nil, fmt.Errorf("s04: tokenizer is nil")
	}
	if !(overlapTokenSize > 0 && chunkTokenSize > overlapTokenSize) {
		return nil, fmt.Errorf("%w: got chunk=%d overlap=%d",
			ErrInvalidChunkSize, chunkTokenSize, overlapTokenSize)
	}
	if strings.TrimSpace(text) == "" {
		return nil, ErrEmptyText
	}

	if splitByCharacter {
		return chunkByCharacter(docID, text, tokenizer, chunkTokenSize, overlapTokenSize)
	}
	return chunkByTokens(docID, text, tokenizer, chunkTokenSize, overlapTokenSize)
}

// chunkByTokens is the default path — the upstream `else:` branch at
// operate.py:155-166.  Slide a fixed window with overlap; the last
// window may be short, which we record as `Tokens = len(tokens)-start`
// matching upstream's `min(chunk_token_size, len(tokens)-start)`.
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
		// Empty body can happen if the decoded slice is pure
		// whitespace (rare with cl100k_base, never with
		// WhitespaceTokenizer).  Skip rather than emit a phantom chunk.
		if body == "" {
			continue
		}
		out = append(out, Chunk{
			ContentDocID:    docID,
			Content:         body,
			Tokens:          end - start,
			ChunkOrderIndex: idx,
		})
		idx++
		if end == len(tokens) {
			break
		}
	}
	return out, nil
}

// chunkByCharacter mirrors upstream's `if split_by_character:` branch.
// We use the rune-based approach as a deterministic, tokenizer-agnostic
// fallback: split the document into rune runs of at most
// `chunkTokenSize * runesPerToken` characters with overlap proportional
// to `overlapTokenSize`.  For pure non-whitespace inputs (the test
// case `TestChunkSplitByCharacterFallback`) this guarantees that even
// if tokenization yields a single huge token, the result is still
// multiple chunks.  For real upstream parity, callers can pre-split on
// a domain-specific separator and pass each piece through the default
// path; the brief asks only for "very long no-whitespace string still
// gets split", which this satisfies.
func chunkByCharacter(
	docID, text string,
	tokenizer Tokenizer,
	chunkTokenSize, overlapTokenSize int,
) ([]Chunk, error) {
	runes := []rune(text)
	if len(runes) == 0 {
		return nil, ErrEmptyText
	}

	// Heuristic: 1 token ~ 4 runes (close to OpenAI's English baseline,
	// over-estimates for CJK so chunks come out smaller — safe).  This
	// keeps the rune-window comparable to the token-window in the
	// default path without needing to re-encode after every step.
	const runesPerToken = 4
	chunkRunes := chunkTokenSize * runesPerToken
	overlapRunes := overlapTokenSize * runesPerToken
	step := chunkRunes - overlapRunes
	if step <= 0 {
		// Defensive: ChunkByTokenSize already validated this in token
		// space, but a degenerate runesPerToken value could still
		// underflow.  Fail loudly rather than infinite-loop.
		return nil, fmt.Errorf("%w: rune-step underflow", ErrInvalidChunkSize)
	}

	out := make([]Chunk, 0, (len(runes)/step)+1)
	idx := 0
	for start := 0; start < len(runes); start += step {
		end := start + chunkRunes
		if end > len(runes) {
			end = len(runes)
		}
		body := strings.TrimSpace(string(runes[start:end]))
		if body == "" {
			continue
		}
		// Re-tokenize each window so the Tokens field reflects the
		// real BPE count, not the rune-budget approximation.  This is
		// the small extra cost split-by-character buys us in exchange
		// for not requiring a special-character corpus.
		toks := tokenizer.Encode(body)
		out = append(out, Chunk{
			ContentDocID:    docID,
			Content:         body,
			Tokens:          len(toks),
			ChunkOrderIndex: idx,
		})
		idx++
		if end == len(runes) {
			break
		}
	}
	if len(out) == 0 {
		return nil, ErrEmptyText
	}
	return out, nil
}

// Compile-time interface check — ensures any future TiktokenTokenizer
// or WhitespaceTokenizer rename can't silently drop a method.  Lives
// here next to the consumer so the failure mode is "chunking won't
// build", which is loud and obvious.
var (
	_ Tokenizer = (*TiktokenTokenizer)(nil)
	_ Tokenizer = (*WhitespaceTokenizer)(nil)
)
